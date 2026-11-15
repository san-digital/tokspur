# SPDX-License-Identifier: Apache-2.0
"""The dataset layer and the training loop, on a small synthetic dump and off the GPU."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
mx = pytest.importorskip("mlx.core")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    from mlx.utils import tree_flatten

    from drafter import fit, taps
    from drafter.data import TraceSet
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
PROMPT, GENERATED, CONVERSATIONS = 10, 50, 6


def write_dump(path: Path, codes: np.ndarray) -> None:
    """Conversations as the worker dumps them: one prefill record for the prompt, then verification rounds."""
    rng = np.random.default_rng(8)
    ordinal = 0
    with path.open("wb") as handle:
        for conversation in range(CONVERSATIONS):
            tokens = (np.arange(PROMPT + GENERATED) * 3 + conversation) % 7
            taps.write_record(
                handle, 0, 0, tokens[:PROMPT], codes[tokens[:PROMPT]], ordinal
            )
            ordinal += 1
            position = PROMPT
            while position < len(tokens):
                rows = min(int(rng.integers(1, 9)), len(tokens) - position)
                chunk = tokens[position : position + rows]
                taps.write_record(handle, 2, position, chunk, codes[chunk], ordinal)
                ordinal += 1
                position += rows


@pytest.fixture
def setting(tmp_path):
    rng = np.random.default_rng(1)
    codes = rng.standard_normal((TINY.vocabulary, TINY.taps * TINY.hidden)).astype(
        np.float16
    )
    dump = tmp_path / "run" / "taps.bin"
    dump.parent.mkdir()
    write_dump(dump, codes)
    rows = (TINY.vocabulary, TINY.hidden)
    target = TargetRows(
        mx.array(rng.standard_normal(rows).astype(np.float32)),
        mx.array(rng.standard_normal(rows).astype(np.float32)),
    )
    return dump, target, tmp_path


def test_traces_take_anchors_from_generated_tokens_only(setting):
    dump, _, _ = setting
    traces = TraceSet.load([dump.parent])
    assert len(traces) == CONVERSATIONS and traces.tokens() == CONVERSATIONS * (
        PROMPT + GENERATED
    )
    assert {trace.prompt_tokens for trace in traces.traces} == {PROMPT}
    first = traces.traces[0].anchors()
    assert first[0] == PROMPT and first[-1] == PROMPT + GENERATED - 2

    training, validation = traces.split(0.2, seed=3)
    assert len(training) == 5 and len(validation) == 1
    plan = training.plan(
        np.random.default_rng(0), anchors_per_sequence=24, anchors_per_step=8
    )
    assert len(plan) == 5 * 3 and all(len(anchors) == 8 for _, anchors in plan)
    index, anchors = plan[0]
    batch = training.batch(index, anchors, TINY.block, TINY.window)
    positions = np.array(batch.context_positions)
    # Only the window the earliest anchor can see is loaded, and positions stay absolute.
    assert positions[0] == max(0, anchors.min() - (TINY.window - 1))
    assert np.array(batch.anchors).tolist() == anchors.tolist()
    assert anchors.min() >= PROMPT


def test_sidecar_sets_the_prompt_length(setting):
    dump, _, _ = setting
    dump.with_name("conversations.json").write_text(
        json.dumps([{"prompt_tokens": 20}] * CONVERSATIONS)
    )
    traces = TraceSet.load([dump])
    assert traces.traces[0].anchors()[0] == 20
    dump.with_name("conversations.json").write_text(json.dumps([{"prompt_tokens": 20}]))
    with pytest.raises(ValueError, match="describes"):
        TraceSet.load([dump])


def test_learning_rate_warms_up_and_decays():
    options = fit.FitOptions(learning_rate=1.0, warmup_steps=10)
    assert fit.learning_rate_at(0, 110, options) == pytest.approx(0.1)
    assert fit.learning_rate_at(9, 110, options) == pytest.approx(1.0)
    assert fit.learning_rate_at(60, 110, options) == pytest.approx(0.5)
    assert fit.learning_rate_at(110, 110, options) == pytest.approx(0.0)


def test_fit_learns_checkpoints_and_resumes_the_same_run(setting):
    dump, target, tmp_path = setting
    training, validation = TraceSet.load([dump]).split(0.2, seed=3)
    options = fit.FitOptions(
        epochs=4,
        anchors_per_sequence=24,
        anchors_per_step=8,
        learning_rate=3e-3,
        warmup_steps=5,
        checkpoint_every=10,
        log_every=5,
        validation_steps=2,
        max_steps=30,
    )
    mx.random.seed(0)
    model = DraftModel(TINY)
    out = tmp_path / "fit"
    summary = fit.fit(model, target, training, validation, options, out)
    assert summary["steps"] == 30 and summary["resumed_from"] == 0
    assert summary["log"][-1]["loss"] < 0.6 * summary["log"][0]["loss"]
    assert (
        summary["validation"]["anchors"] == 16
        and summary["validation"]["accepted"] >= 0
    )
    progress = json.loads((out / "state/progress.json").read_text())
    assert progress["step"] == 30
    rows = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows if r["event"] == "train"] == [5, 10, 15, 20, 25, 30]
    assert [r["step"] for r in rows if r["event"] == "validation"] == [10, 20, 30]

    # A second process picks the run up from its saved state and takes the next steps of the same plan.
    options.max_steps = 45
    mx.random.seed(123)
    resumed = DraftModel(TINY)
    again = fit.fit(resumed, target, training, validation, options, out, resume=True)
    assert again["resumed_from"] == 30 and again["steps"] == 45
    assert json.loads((out / "state/progress.json").read_text())["step"] == 45
    restored = dict(tree_flatten(resumed.parameters()))
    assert all(np.isfinite(np.array(value)).all() for value in restored.values())
    with pytest.raises(FileExistsError):
        fit.fit(DraftModel(TINY), target, training, validation, options, out)
