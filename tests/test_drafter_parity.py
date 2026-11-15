# SPDX-License-Identifier: Apache-2.0
"""The round-by-round comparison must agree with a dump made from the same model and notice a different one."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
mx = pytest.importorskip("mlx.core")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    from mlx.utils import tree_flatten, tree_unflatten

    from drafter import parity, taps
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
PROMPT, BUDGET = 6, 30


def randomised(seed: int) -> DraftModel:
    """A model whose every parameter is random. A new model barely changes its inputs, so two new models draft the same tokens."""
    model = DraftModel(TINY)
    rng = np.random.default_rng(seed)
    values = [
        (
            name,
            mx.array(
                (("norm" in name) + 0.4 * rng.standard_normal(value.shape)).astype(
                    np.float32
                )
            ),
        )
        for name, value in tree_flatten(model.parameters())
    ]
    model.update(tree_unflatten(values))
    return model


def write_generation(handle, model, target, codes, rng, ordinal):
    """Write one conversation the way the worker would: prefill, then rounds whose accepted drafts are this model's own drafts."""
    tokens = list(rng.integers(0, 30, PROMPT))
    taps.write_record(handle, 0, 0, np.array(tokens), codes[tokens], ordinal)
    ordinal += 1
    generated, accepted_total = 0, 0
    anchor = int(rng.integers(0, 30))
    while generated < BUDGET:
        base = PROMPT + generated
        tokens.append(anchor)
        if BUDGET - generated < 2:
            taps.write_record(
                handle, 1, base, np.array([anchor]), codes[[anchor]], ordinal
            )
            ordinal += 1
            break
        width = min(7, BUDGET - generated - 1)
        drafts = parity.proposals(
            model, target, np.array(tokens), codes[tokens], np.array([base])
        )[0][:width]
        accepted = int(rng.integers(0, width + 1))
        committed = [anchor] + [int(t) for t in drafts[:accepted]]
        tokens.extend(committed[1:])
        taps.write_record(
            handle, 2, base, np.array(committed), codes[committed], ordinal
        )
        ordinal += 1
        generated += len(committed)
        accepted_total += accepted
        # The target rejects the next draft: its own token becomes the next anchor.
        rejected = int(drafts[accepted]) if accepted < width else -1
        anchor = next(t for t in rng.permutation(30) if t != rejected)
    return ordinal, accepted_total


def test_comparison_agrees_with_its_own_model_and_notices_another(tmp_path):
    rng = np.random.default_rng(4)
    model = randomised(3)
    rows = (TINY.vocabulary, TINY.hidden)
    target = TargetRows(
        mx.array(rng.standard_normal(rows).astype(np.float32)),
        mx.array(rng.standard_normal(rows).astype(np.float32)),
    )
    codes = rng.standard_normal((TINY.vocabulary, TINY.taps * TINY.hidden)).astype(
        np.float16
    )
    path = tmp_path / "taps.bin"
    ordinal, accepted = 0, 0
    with path.open("wb") as handle:
        for _ in range(2):
            ordinal, gained = write_generation(
                handle, model, target, codes, rng, ordinal
            )
            accepted += gained

    report = parity.compare(path, model, target, [BUDGET, BUDGET], set(), 7)
    assert report["rounds"] > 6 and report["agree"] == report["rounds"]
    assert report["accepted_by_worker"] == report["accepted_by_model"] == accepted
    assert [c["prompt_tokens"] for c in report["conversations"]] == [PROMPT, PROMPT]

    other = randomised(99)
    different = parity.compare(path, other, target, [BUDGET, BUDGET], set(), 7)
    assert different["agree"] < different["rounds"] and different["disagreements"]

    with pytest.raises(ValueError, match="budgets"):
        parity.compare(path, model, target, [BUDGET], set(), 7)
