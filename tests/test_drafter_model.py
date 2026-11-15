# SPDX-License-Identifier: Apache-2.0
"""The MLX drafter model must compute what the Metal kernels compute and export what the loader accepts."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
mx = pytest.importorskip("mlx.core")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    from mlx.utils import tree_flatten, tree_unflatten

    from drafter import contract, export
    from drafter.model import DraftConfig, DraftModel, TargetRows, context_mask
finally:
    sys.path.remove(str(ROOT))

# Tests stay off the GPU so that they never disturb a timed run on the same machine.
mx.set_default_device(mx.cpu)

TINY = DraftConfig(
    hidden=32,
    intermediate=48,
    layers=2,
    query_heads=4,
    key_value_heads=2,
    head_dim=8,
    window=6,
    vocabulary=40,
    taps=2,
    selector_rank=6,
    selector_top_k=4,
    mask_token_id=39,
)


def randomised(config: DraftConfig, seed: int = 7) -> tuple[DraftModel, dict]:
    """A model whose every parameter is random, so that identity and zero initial values cannot hide an error."""
    model = DraftModel(config)
    rng = np.random.default_rng(seed)
    values = {}
    for name, value in tree_flatten(model.parameters()):
        spread = 0.5 if "norm" in name or "base_kernel" in name else 0.3
        centre = 1.0 if "norm" in name else 0.0
        values[name] = (centre + spread * rng.standard_normal(value.shape)).astype(
            np.float32
        )
    model.update(tree_unflatten([(k, mx.array(v)) for k, v in values.items()]))
    return model, values


def rms(x, weight, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * weight


def rotate(vector, position, theta):
    half = vector.shape[-1] // 2
    angle = np.float32(position) * theta ** (
        -2.0 * np.arange(half, dtype=np.float32) / vector.shape[-1]
    )
    first, second = vector[:half], vector[half:]
    return np.concatenate(
        [
            first * np.cos(angle) - second * np.sin(angle),
            second * np.cos(angle) + first * np.sin(angle),
        ]
    )


def reference_block(p, c, taps, context_positions, anchor, block):
    """One block, row by row, transcribed from draft_block8 and the kernels in native/kernels/dflash2.metal."""
    rows, hidden = block.shape
    groups = hidden // c.conv_group
    per_group = c.query_heads // c.key_value_heads
    dim = c.head_dim

    def convolve(values, dynamic, base, side):
        out = np.zeros_like(values)
        for row in range(rows):
            for channel in range(hidden):
                group = channel // c.conv_group
                out[row, channel] = values[row, channel] * (
                    base[side, 0, channel] + dynamic[row, (side * 2) * groups + group]
                )
                if row > 0:
                    out[row, channel] += values[row - 1, channel] * (
                        base[side, 1, channel]
                        + dynamic[row, (side * 2 + 1) * groups + group]
                    )
        return out

    fused = rms(taps @ p["fc.weight"].T, p["hidden_norm.weight"], c.rms_eps)
    state = block.copy()
    for layer in range(c.layers):
        pre = f"layers.{layer}."
        normed = rms(state, p[pre + "input_layernorm.weight"], c.rms_eps)
        dynamic = normed @ p[pre + "attention_conv.kernel_projection.weight"].T
        base = p[pre + "attention_conv.base_kernel"]
        conv = convolve(normed, dynamic, base, 0)
        q = conv @ p[pre + "self_attn.q_proj.weight"].T
        k = conv @ p[pre + "self_attn.k_proj.weight"].T
        v = conv @ p[pre + "self_attn.v_proj.weight"].T
        context_k = fused @ p[pre + "self_attn.k_proj.weight"].T
        context_v = fused @ p[pre + "self_attn.v_proj.weight"].T
        q_norm = p[pre + "self_attn.q_norm.weight"]
        k_norm = p[pre + "self_attn.k_norm.weight"]
        attended = np.zeros((rows, c.query_heads * dim), dtype=np.float64)
        for row in range(rows):
            position = anchor + row
            for head in range(c.query_heads):
                group = head // per_group
                span = slice(group * dim, (group + 1) * dim)
                query = rotate(
                    rms(q[row, head * dim : (head + 1) * dim], q_norm, c.rms_eps),
                    position,
                    c.rope_theta,
                )
                scores, values = [], []
                for index, key_position in enumerate(context_positions):
                    if key_position < anchor and position - key_position < c.window:
                        key = rotate(
                            rms(context_k[index, span], k_norm, c.rms_eps),
                            key_position,
                            c.rope_theta,
                        )
                        scores.append(query @ key * dim**-0.5)
                        values.append(context_v[index, span])
                for other in range(rows):
                    key = rotate(
                        rms(k[other, span], k_norm, c.rms_eps),
                        anchor + other,
                        c.rope_theta,
                    )
                    scores.append(query @ key * dim**-0.5)
                    values.append(v[other, span])
                scores = np.array(scores, dtype=np.float64)
                weights = np.exp(scores - scores.max())
                weights /= weights.sum()
                attended[row, head * dim : (head + 1) * dim] = weights @ np.array(
                    values
                )
        projected = attended @ p[pre + "self_attn.o_proj.weight"].T
        residual = state + convolve(projected, dynamic, base, 1)
        normed = rms(residual, p[pre + "post_attention_layernorm.weight"], c.rms_eps)
        dynamic = normed @ p[pre + "mlp_conv.kernel_projection.weight"].T
        base = p[pre + "mlp_conv.base_kernel"]
        conv = convolve(normed, dynamic, base, 0)
        gate = conv @ p[pre + "mlp.gate_proj.weight"].T
        inner = gate / (1 + np.exp(-gate)) * (conv @ p[pre + "mlp.up_proj.weight"].T)
        projected = inner @ p[pre + "mlp.down_proj.weight"].T
        state = residual + convolve(projected, dynamic, base, 1)
    return rms(state, p["norm.weight"], c.rms_eps)


def test_parameter_names_and_shapes_are_the_checkpoint_contract():
    flat = dict(tree_flatten(DraftModel().parameters()))
    assert {
        name: tuple(value.shape) for name, value in flat.items()
    } == contract.tensors()


def test_new_model_convolution_is_a_no_op():
    model = DraftModel(TINY)
    conv = model.layers[0].attention_conv
    hidden = mx.random.normal((3, 8, TINY.hidden))
    prepared, dynamic = conv.prepare(hidden)
    assert mx.allclose(prepared, hidden).item()
    assert mx.allclose(conv.finish(hidden, dynamic), hidden).item()


def test_context_mask_follows_the_window():
    positions = mx.arange(100, 110)
    mask = np.array(context_mask(mx.array([103, 110]), 8, positions, 6))
    assert mask.shape == (2, 8, 10)
    # Anchor 103 has context 100..102; row 4 sits at 107 and its window of six starts at 102.
    assert mask[0, 0].tolist() == [True] * 3 + [False] * 7
    assert mask[0, 4].tolist() == [False, False, True] + [False] * 7
    assert not mask[0, 5].any()
    # Anchor 110 has all ten rows as context; row 0 sees 105..109 and row 5 nothing.
    assert mask[1, 0].tolist() == [False] * 5 + [True] * 5
    assert not mask[1, 5].any()


def test_forward_matches_the_row_by_row_reference():
    model, values = randomised(TINY)
    rng = np.random.default_rng(11)
    context_positions = np.arange(100, 110)
    taps = rng.standard_normal((10, TINY.taps * TINY.hidden)).astype(np.float32)
    anchors = np.array([103, 110, 100])
    blocks = rng.standard_normal((3, TINY.block, TINY.hidden)).astype(np.float32)

    context = model.context(mx.array(taps), mx.array(context_positions))
    found = np.array(
        model.hidden(
            mx.array(blocks), mx.array(anchors), context, mx.array(context_positions)
        )
    )
    for index, anchor in enumerate(anchors):
        wanted = reference_block(
            values, TINY, taps, context_positions, int(anchor), blocks[index]
        )
        assert np.abs(found[index] - wanted).max() < 2e-4, f"anchor {anchor}"


def test_selector_edges_match_the_kernel_formula():
    model, values = randomised(TINY)
    rng = np.random.default_rng(3)
    hidden = rng.standard_normal((2, 7, TINY.hidden)).astype(np.float32)
    previous = rng.integers(0, TINY.vocabulary, (2, 7))
    candidates = rng.integers(0, TINY.vocabulary, (2, 7, TINY.selector_top_k))
    found = np.array(
        model.candidate_selector.edges(
            mx.array(hidden), mx.array(previous), mx.array(candidates), None
        )
    )
    selector = "candidate_selector."
    gate = values[selector + "predecessor_codebook"][previous] * (
        hidden @ values[selector + "hidden_projection.weight"].T
    )
    wanted = np.einsum(
        "bpr,bpkr->bpk", gate, values[selector + "successor_codebook"][candidates]
    )
    assert np.abs(found - wanted).max() < 1e-4


def test_projected_selector_tables_export_as_full_tables():
    config = DraftConfig(**{**TINY.describe(), "selector_tables": False})
    model = DraftModel(config)
    rng = np.random.default_rng(5)
    for name in ("predecessor_projection", "successor_projection"):
        shape = (config.selector_rank, config.hidden)
        getattr(model.candidate_selector, name).weight = mx.array(
            rng.standard_normal(shape).astype(np.float32)
        )
    rows = (config.vocabulary, config.hidden)
    target = TargetRows(
        mx.array(rng.standard_normal(rows).astype(np.float32)),
        mx.array(rng.standard_normal(rows).astype(np.float32)),
    )
    hidden = mx.array(rng.standard_normal((2, 7, config.hidden)).astype(np.float32))
    previous = mx.array(rng.integers(0, config.vocabulary, (2, 7)))
    candidates = mx.array(rng.integers(0, config.vocabulary, (2, 7, 4)))
    projected = np.array(
        model.candidate_selector.edges(hidden, previous, candidates, target)
    )

    tables = DraftModel(TINY)
    written = export.model_tensors(model, target)
    tables.candidate_selector.update(
        {
            "hidden_projection": {
                "weight": model.candidate_selector.hidden_projection.weight
            },
            "predecessor_codebook": mx.array(
                written["candidate_selector.predecessor_codebook"]
            ),
            "successor_codebook": mx.array(
                written["candidate_selector.successor_codebook"]
            ),
        }
    )
    full = np.array(tables.candidate_selector.edges(hidden, previous, candidates, None))
    assert np.abs(projected - full).max() < 1e-3


def test_block_inputs_are_anchor_then_mask():
    model = DraftModel(TINY)
    rows = np.arange(TINY.vocabulary * TINY.hidden, dtype=np.float32).reshape(
        TINY.vocabulary, TINY.hidden
    )
    target = TargetRows(mx.array(rows), mx.array(rows))
    blocks = np.array(model.block_inputs(mx.array([4, 9]), target))
    assert blocks.shape == (2, TINY.block, TINY.hidden)
    assert (blocks[:, 0] == rows[[4, 9]]).all()
    assert (blocks[:, 1:] == rows[TINY.mask_token_id]).all()


def test_bf16_rounds_to_nearest_even_and_refuses_what_the_loader_refuses():
    def as_float(bits):
        return np.array([bits], dtype=np.uint32).view(np.float32)

    assert export.to_bf16(as_float(0x3F800000))[0] == 0x3F80
    # Exactly half way rounds to the even neighbour; just above half way rounds up.
    assert export.to_bf16(as_float(0x3F808000))[0] == 0x3F80
    assert export.to_bf16(as_float(0x3F818000))[0] == 0x3F82
    assert export.to_bf16(as_float(0x3F808001))[0] == 0x3F81
    restored = export.from_bf16(export.to_bf16(np.float32([0.1, -2.5, 3e-12])))
    assert np.allclose(restored, [0.1, -2.5, 3e-12], rtol=4e-3)
    with pytest.raises(ValueError):
        export.to_bf16(np.float32([np.inf]))
    with pytest.raises(ValueError):
        export.to_bf16(np.float32([np.finfo(np.float32).max]))


def test_writer_refuses_tensors_outside_the_contract(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        export.write_checkpoint(tmp_path / "empty", {}, export.config_json())
    assert not (tmp_path / "empty").exists()


def test_full_checkpoint_round_trip_when_asked(tmp_path):
    if not os.environ.get("TOKSPUR_DRAFTER_SLOW"):
        pytest.skip("set TOKSPUR_DRAFTER_SLOW=1 to write a full 1.2 GB checkpoint")
    mx.random.seed(1)
    model = DraftModel()
    tensors = export.model_tensors(model)
    report = export.write_checkpoint(tmp_path / "ckpt", tensors, export.config_json())
    assert report["accepted"] and report["required_parameters"] == 603_065_600
    restored = export.read_checkpoint(tmp_path / "ckpt")
    for name in ("fc.weight", "layers.4.mlp.down_proj.weight", "norm.weight"):
        assert np.allclose(restored[name], tensors[name], rtol=4e-3, atol=1e-30)
