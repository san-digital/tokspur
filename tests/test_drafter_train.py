# SPDX-License-Identifier: Apache-2.0
"""The tap dump reader and the drafter training objective, on small synthetic data and off the GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
mx = pytest.importorskip("mlx.core")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    from mlx import optimizers

    from drafter import taps, train
    from drafter.model import DraftConfig, DraftModel, TargetRows
finally:
    sys.path.remove(str(ROOT))

mx.set_default_device(mx.cpu)

TINY = DraftConfig(
    hidden=32,
    intermediate=48,
    layers=2,
    query_heads=4,
    key_value_heads=2,
    head_dim=8,
    window=16,
    vocabulary=40,
    taps=2,
    selector_rank=6,
    selector_top_k=4,
    mask_token_id=39,
)
WIDTH = 5


def tap_rows(tokens, salt=0.0):
    tokens = np.asarray(tokens, dtype=np.float32)
    return (tokens[:, None] * 0.5 + np.arange(WIDTH)[None] * 0.125 + salt).astype(
        np.float16
    )


def write_dump(path, rows):
    with path.open("wb") as handle:
        for ordinal, (source, first, tokens, salt) in enumerate(rows):
            taps.write_record(
                handle, source, first, np.array(tokens), tap_rows(tokens, salt), ordinal
            )


def test_dump_reader_rebuilds_conversations_and_rewinds(tmp_path):
    path = tmp_path / "taps.bin"
    write_dump(
        path,
        [
            (0, 0, [10, 11, 12, 13, 14, 15], 0.0),
            (2, 6, [16, 17, 18], 0.0),
            # The worker rewound to position 7: rows 7 and 8 are replaced.
            (2, 7, [27, 28], 1.0),
            (1, 9, [19], 0.0),
            (0, 0, [1, 2, 3, 4], 0.0),
        ],
    )
    with path.open("ab") as handle:
        # A record cut short by a stopped worker is ignored.
        handle.write(taps.HEADER.pack(taps.MAGIC, 1, 4, 3, WIDTH, 1, 5, 0) + b"\0" * 7)

    assert [r.source for r in taps.records(path)] == [
        "prefill",
        "verification",
        "verification",
        "step",
        "prefill",
    ]
    payload, found = taps.conversations(path)
    assert [len(c) for c in found] == [10, 4]
    first = found[0]
    assert first.tokens.tolist() == [10, 11, 12, 13, 14, 15, 16, 27, 28, 19]
    wanted = np.concatenate(
        [
            tap_rows([10, 11, 12, 13, 14, 15, 16]),
            tap_rows([27, 28], 1.0),
            tap_rows([19]),
        ]
    )
    assert (first.taps(payload) == wanted).all()
    assert (first.taps(payload, 6, 9) == wanted[6:9]).all()
    assert (found[1].taps(payload) == tap_rows([1, 2, 3, 4])).all()


def test_dump_reader_refuses_gaps_and_foreign_files(tmp_path):
    gap = tmp_path / "gap.bin"
    write_dump(gap, [(0, 0, [1, 2], 0.0), (2, 5, [3], 0.0)])
    with pytest.raises(ValueError, match="gap"):
        taps.conversations(gap)
    foreign = tmp_path / "foreign.bin"
    foreign.write_bytes(b"\0" * 64)
    with pytest.raises(ValueError, match="tap record"):
        taps.records(foreign)


def synthetic(length=70):
    """A conversation whose next tokens follow from the anchor token, with taps that identify each token."""
    rng = np.random.default_rng(1)
    tokens = (np.arange(length) * 3 + 1) % 7
    codes = rng.standard_normal((TINY.vocabulary, TINY.taps * TINY.hidden)).astype(
        np.float32
    )
    rows = (TINY.vocabulary, TINY.hidden)
    target = TargetRows(
        mx.array(rng.standard_normal(rows).astype(np.float32)),
        mx.array(rng.standard_normal(rows).astype(np.float32)),
    )
    return tokens, codes[tokens], target


def test_batch_masks_labels_past_the_end():
    tokens, tap_values, _ = synthetic(20)
    batch = train.build_batch(
        tokens, tap_values, np.array([3, 15, 18]), TINY.block, first_position=100
    )
    assert np.array(batch.anchors).tolist() == [103, 115, 118]
    assert np.array(batch.anchor_tokens).tolist() == tokens[[3, 15, 18]].tolist()
    assert np.array(batch.labels)[0].tolist() == tokens[4:11].tolist()
    assert np.array(batch.valid).sum(axis=1).tolist() == [7, 4, 1]
    with pytest.raises(ValueError):
        train.build_batch(tokens, tap_values, np.array([19]), TINY.block)


def test_training_reduces_the_loss_and_reaches_every_part():
    mx.random.seed(0)
    tokens, tap_values, target = synthetic()
    model = DraftModel(TINY)
    step = train.make_step(model, target, optimizers.Adam(learning_rate=3e-3))
    rng = np.random.default_rng(2)

    before = {
        name: np.array(value)
        for name, value in __import__("mlx.utils").utils.tree_flatten(
            model.parameters()
        )
    }
    history = []
    for _ in range(80):
        anchors = rng.choice(np.arange(1, len(tokens) - 1), size=16, replace=False)
        total, metrics = step(
            train.build_batch(tokens, tap_values, anchors, TINY.block)
        )
        mx.eval(model.parameters(), total)
        history.append((total.item(), metrics["top1"].item()))

    assert history[-1][0] < 0.5 * history[0][0], history[::20]
    assert history[-1][1] > 0.8, history[::20]
    after = dict(__import__("mlx.utils").utils.tree_flatten(model.parameters()))
    moved = {
        name
        for name, value in before.items()
        if not np.array_equal(value, np.array(after[name]))
    }
    for part in (
        "fc.weight",
        "layers.0.attention_conv.kernel_projection.weight",
        "layers.1.mlp_conv.base_kernel",
        "layers.1.self_attn.q_proj.weight",
        "candidate_selector.hidden_projection.weight",
        "candidate_selector.successor_codebook",
        "norm.weight",
    ):
        assert part in moved, part


def test_head_slice_leaves_out_labels_it_cannot_draft():
    mx.random.seed(0)
    tokens, tap_values, target = synthetic(30)
    model = DraftModel(TINY)
    batch = train.build_batch(tokens, tap_values, np.arange(1, 9), TINY.block)
    _, full = train.losses(model, target, batch)
    # Token ids run from 0 to 6; a head restricted to the first five ids cannot draft 5 or 6.
    _, sliced = train.losses(model, target, batch, head_rows=5)
    labels = np.array(batch.labels)
    assert full["positions"].item() == labels.size
    assert sliced["positions"].item() == int((labels < 5).sum())


def test_expected_accepted_length_counts_only_unbroken_runs():
    mx.random.seed(0)
    tokens, tap_values, target = synthetic(40)
    model = DraftModel(TINY)
    batch = train.build_batch(tokens, tap_values, np.arange(1, 17), TINY.block)
    _, metrics = train.losses(model, target, batch)
    expected, accepted = metrics["expected_accepted"].item(), metrics["accepted"].item()
    # Both count drafts per anchor, so they lie between zero and the seven drafts of a block.
    assert 0.0 <= expected <= 7.0 and 0.0 <= accepted <= 7.0
    plain, _ = train.losses(model, target, batch)
    weighted, _ = train.losses(model, target, batch, eal_weight=2.0)
    assert abs((plain - weighted).item() - 2.0 * expected) < 1e-4

    # Training on the term alone raises the simulated greedy acceptance.
    step = train.make_step(
        model, target, optimizers.Adam(learning_rate=3e-3), eal_weight=1.0
    )
    rng = np.random.default_rng(5)
    for _ in range(60):
        anchors = rng.choice(np.arange(1, len(tokens) - 1), size=16, replace=False)
        total, _ = step(train.build_batch(tokens, tap_values, anchors, TINY.block))
        mx.eval(model.parameters(), total)
    _, after = train.losses(model, target, batch)
    assert after["accepted"].item() > accepted + 1.0
