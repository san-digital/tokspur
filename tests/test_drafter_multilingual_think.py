# SPDX-License-Identifier: Apache-2.0
"""Multilingual + <think> corpus diagnostics and stratified anchor planning for DFlash2 training."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
mx = pytest.importorskip("mlx.core")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    from drafter import corpus, export, fit, parity, taps, train, vocab
    from drafter.data import TraceSet
    from drafter.model import DraftConfig, DraftModel, TargetRows
finally:
    sys.path.remove(str(ROOT))

mx.set_default_device(mx.cpu)


def _write_qwen_think_cjk_dump(path: Path) -> None:
    """Write a realistic Qwen3.8 Flash-Next tap dump with English <think> reasoning followed by Japanese Kanji prose."""
    width = 16
    rng = np.random.default_rng(42)
    prompt = np.array(
        [151_644, 872, 198, 45_200, 178_900, 248_046, 198], dtype=np.uint32
    )
    think_span = np.arange(1_200, 1_240, dtype=np.uint32)
    jp_prose = np.array(
        [
            42_100,
            48_500,
            162_300,
            189_400,
            51_200,
            174_800,
            205,
            198_100,
            44_300,
            152_900,
        ]
        * 3
        + [corpus.IM_END_ID],
        dtype=np.uint32,
    )
    generated = np.concatenate(
        [
            np.array([corpus.THINK_OPEN_ID], dtype=np.uint32),
            think_span,
            np.array([corpus.THINK_CLOSE_ID], dtype=np.uint32),
            jp_prose,
        ]
    )
    full = np.concatenate([prompt, generated])
    tap_rows = rng.standard_normal((len(full), width)).astype(np.float16)
    with path.open("wb") as handle:
        taps.write_record(handle, 0, 0, prompt, tap_rows[: len(prompt)], 0)
        taps.write_record(handle, 2, len(prompt), generated, tap_rows[len(prompt) :], 1)


def test_multilingual_think_corpus_report_and_validation(tmp_path):
    dump = tmp_path / "jp_think" / "taps.bin"
    dump.parent.mkdir()
    _write_qwen_think_cjk_dump(dump)

    traces = TraceSet.load([dump.parent])
    report = corpus.token_band_report(traces)
    assert report["conversations"] == 1
    assert report["prompt_tokens"] == 7
    assert report["think_open_count"] == 1 and report["think_close_count"] == 1
    assert (
        report["think_conversations"] == 1
        and report["balanced_think_conversations"] == 1
    )
    assert report["think_span_tokens"] == 40
    assert report["think_ascii_fraction"] == pytest.approx(1.0)
    assert report["post_think_tokens"] == 31
    assert report["cjk_multilingual_tokens"] == 27
    assert report["shinjitai_compound_tokens"] == 15
    assert report["control_tokens"] == 3
    assert report["post_think_cjk_fraction"] == pytest.approx(27 / 31)

    assert (
        corpus.validate_multilingual_think_corpus(traces, min_cjk_fraction=0.25) == []
    )
    low_cjk = corpus.validate_multilingual_think_corpus(traces, min_cjk_fraction=0.90)
    assert len(low_cjk) == 1 and "CJK/multilingual fraction" in low_cjk[0]

    assert (
        corpus.main(
            ["--dumps", str(dump.parent), "--min-cjk-fraction", "0.25", "--strict"]
        )
        == 0
    )
    assert (
        corpus.main(
            ["--dumps", str(dump.parent), "--min-cjk-fraction", "0.90", "--strict"]
        )
        == 1
    )


def test_unbalanced_think_trace_is_flagged(tmp_path):
    dump = tmp_path / "broken" / "taps.bin"
    dump.parent.mkdir()
    prompt = np.array([10, 11, 12], dtype=np.uint32)
    generated = np.array(
        [corpus.THINK_OPEN_ID, 500, 501, 45_000, 160_000], dtype=np.uint32
    )
    tap_rows = np.ones((8, 8), dtype=np.float16)
    with dump.open("wb") as handle:
        taps.write_record(handle, 0, 0, prompt, tap_rows[:3], 0)
        taps.write_record(handle, 2, 3, generated, tap_rows[3:], 1)

    traces = TraceSet.load([dump])
    problems = corpus.validate_multilingual_think_corpus(traces)
    assert any("unbalanced <think>" in item for item in problems)
    assert (
        corpus.validate_multilingual_think_corpus(traces, require_balanced_think=False)
        == []
    )


def test_capture_manifest_is_deterministic_and_validates_inputs():
    specs = [
        {
            "id": "jp-msa-01",
            "language": "ja",
            "domain": "legal",
            "messages": [{"role": "user", "content": "契約書を審査してください"}],
        },
        {
            "id": "rust-borrow-02",
            "language": "en",
            "domain": "code",
            "messages": [{"role": "user", "content": "Explain Pin<&mut T>"}],
        },
    ]
    first = corpus.capture_manifest(specs, seed=19)
    second = corpus.capture_manifest(specs, seed=19)
    assert first == second and {row["id"] for row in first} == {
        "jp-msa-01",
        "rust-borrow-02",
    }
    with pytest.raises(ValueError, match="unique"):
        corpus.capture_manifest([specs[0], specs[0]])
    with pytest.raises(ValueError, match="messages"):
        corpus.capture_manifest([{"id": "bad", "messages": []}])


def test_stratified_anchor_weights_preserve_uniform_rng_and_enrich_cjk_and_think(
    tmp_path,
):
    dump = tmp_path / "strat" / "taps.bin"
    dump.parent.mkdir()
    _write_qwen_think_cjk_dump(dump)
    traces = TraceSet.load([dump])
    trace = traces.traces[0]

    # Default (1.0, 1.0) must draw identical anchors in identical order to unweighted sampling.
    plan_a = traces.plan(
        np.random.default_rng(77), anchors_per_sequence=16, anchors_per_step=8
    )
    plan_b = traces.plan(
        np.random.default_rng(77),
        anchors_per_sequence=16,
        anchors_per_step=8,
        cjk_weight=1.0,
        think_boundary_weight=1.0,
    )
    assert [(i, a.tolist()) for i, a in plan_a] == [(i, a.tolist()) for i, a in plan_b]

    weights = trace.anchor_weights(block=8, cjk_weight=4.0, think_boundary_weight=6.0)
    eligible = trace.anchors()
    close_pos = int(np.where(trace.tokens == corpus.THINK_CLOSE_ID)[0][0])
    # Anchor right before </think> spans both </think> (x6) and Japanese CJK labels immediately after it (x4) -> 24.0.
    idx_before_close = int(np.where(eligible == close_pos - 1)[0][0])
    assert weights[idx_before_close] == pytest.approx(24.0)
    # Early English <think> interior anchor far from </think> with ASCII-only labels keeps weight 1.0.
    idx_interior_think = int(np.where(eligible == trace.prompt_tokens + 5)[0][0])
    assert weights[idx_interior_think] == pytest.approx(1.0)

    uniform_hits = 0
    stratified_hits = 0
    for trial in range(200):
        u_steps = traces.plan(
            np.random.default_rng(1000 + trial),
            anchors_per_sequence=12,
            anchors_per_step=6,
        )
        s_steps = traces.plan(
            np.random.default_rng(1000 + trial),
            anchors_per_sequence=12,
            anchors_per_step=6,
            block=8,
            cjk_weight=4.0,
            think_boundary_weight=8.0,
        )
        u_all = np.concatenate([a for _, a in u_steps])
        s_all = np.concatenate([a for _, a in s_steps])
        uniform_hits += int(np.sum((u_all >= close_pos - 7) & (u_all <= close_pos)))
        stratified_hits += int(np.sum((s_all >= close_pos - 7) & (s_all <= close_pos)))
    assert stratified_hits > int(uniform_hits * 1.8)

    with pytest.raises(ValueError, match="above zero"):
        trace.anchor_weights(cjk_weight=0.0)
    with pytest.raises(ValueError, match="above zero"):
        traces.plan(np.random.default_rng(0), 8, 4, think_boundary_weight=-1.0)


def test_fit_runs_with_stratified_multilingual_think_weights(tmp_path):
    tiny = DraftConfig(
        hidden=32,
        intermediate=48,
        layers=2,
        query_heads=4,
        key_value_heads=2,
        head_dim=8,
        window=16,
        vocabulary=64,
        taps=2,
        selector_rank=6,
        selector_top_k=4,
        mask_token_id=63,
    )
    rng = np.random.default_rng(5)
    codes = rng.standard_normal((tiny.vocabulary, tiny.taps * tiny.hidden)).astype(
        np.float16
    )
    dump = tmp_path / "tiny_jp" / "taps.bin"
    dump.parent.mkdir()
    ordinal = 0
    with dump.open("wb") as handle:
        for conv in range(4):
            prompt = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint32)
            think_body = np.array(
                [7 + (i + conv) % 6 for i in range(18)], dtype=np.uint32
            )
            cjk_body = np.array(
                [24 + (i * 3 + conv) % 16 for i in range(20)] + [56], dtype=np.uint32
            )
            generated = np.concatenate(
                [
                    np.array([58], dtype=np.uint32),
                    think_body,
                    np.array([59], dtype=np.uint32),
                    cjk_body,
                ]
            )
            taps.write_record(handle, 0, 0, prompt, codes[prompt], ordinal)
            ordinal += 1
            taps.write_record(
                handle, 2, len(prompt), generated, codes[generated], ordinal
            )
            ordinal += 1

    training, validation = TraceSet.load([dump]).split(0.25, seed=2)
    target = TargetRows(
        mx.array(
            rng.standard_normal((tiny.vocabulary, tiny.hidden)).astype(np.float32)
        ),
        mx.array(
            rng.standard_normal((tiny.vocabulary, tiny.hidden)).astype(np.float32)
        ),
    )
    options = fit.FitOptions(
        epochs=3,
        anchors_per_sequence=16,
        anchors_per_step=8,
        learning_rate=3e-3,
        warmup_steps=3,
        checkpoint_every=6,
        log_every=3,
        validation_steps=2,
        max_steps=18,
        cjk_weight=3.0,
        think_boundary_weight=5.0,
        cjk_floor=24,
        control_floor=56,
        think_tokens=(58, 59),
    )
    mx.random.seed(0)
    model = DraftModel(tiny)
    out = tmp_path / "fit_out"
    summary = fit.fit(model, target, training, validation, options, out)
    assert summary["steps"] == 18
    assert summary["log"][-1]["loss"] < 0.75 * summary["log"][0]["loss"]
    assert summary["validation"]["anchors"] == 16


def test_prompt_opened_think_region_is_balanced(tmp_path):
    dump = tmp_path / "chat_think" / "taps.bin"
    dump.parent.mkdir()
    prompt = np.array(
        [151_644, 872, 198, 45_200, 248_046, 198, corpus.THINK_OPEN_ID, 198],
        dtype=np.uint32,
    )
    generated = np.array(
        [
            1_200,
            1_201,
            1_202,
            corpus.THINK_CLOSE_ID,
            198,
            42_100,
            162_300,
            corpus.IM_END_ID,
        ],
        dtype=np.uint32,
    )
    tap_rows = np.ones((len(prompt) + len(generated), 16), dtype=np.float16)
    with dump.open("wb") as handle:
        taps.write_record(handle, 0, 0, prompt, tap_rows[: len(prompt)], 0)
        taps.write_record(handle, 2, len(prompt), generated, tap_rows[len(prompt) :], 1)

    traces = TraceSet.load([dump.parent])
    report = corpus.token_band_report(traces)
    assert report["think_conversations"] == 1
    assert report["balanced_think_conversations"] == 1
    assert report["think_span_tokens"] == 3
    assert report["post_think_tokens"] == 4
    assert report["post_think_cjk_fraction"] == pytest.approx(2 / 4)
    assert (
        corpus.validate_multilingual_think_corpus(traces, min_cjk_fraction=0.20) == []
    )


def test_warm_start_from_checkpoint_tensors_preserves_state_and_trains(tmp_path):
    tiny = DraftConfig(
        hidden=32,
        intermediate=48,
        layers=2,
        query_heads=4,
        key_value_heads=2,
        head_dim=8,
        window=16,
        vocabulary=64,
        taps=2,
        selector_rank=6,
        selector_top_k=4,
        mask_token_id=63,
        selector_tables=True,
    )
    rng = np.random.default_rng(11)
    codes = rng.standard_normal((tiny.vocabulary, tiny.taps * tiny.hidden)).astype(
        np.float16
    )
    dump = tmp_path / "warm_jp" / "taps.bin"
    dump.parent.mkdir()
    with dump.open("wb") as handle:
        for conv in range(3):
            prompt = np.array([1, 2, 3, 58], dtype=np.uint32)
            generated = np.array(
                [7, 8, 9, 59, 25 + conv, 28 + conv, 31 + conv, 56], dtype=np.uint32
            )
            taps.write_record(handle, 0, 0, prompt, codes[prompt], 2 * conv)
            taps.write_record(
                handle, 2, len(prompt), generated, codes[generated], 2 * conv + 1
            )

    mx.random.seed(9)
    base = DraftModel(tiny)
    saved = {
        name: np.array(val.astype(mx.float32))
        for name, val in fit.tree_flatten(base.parameters())
    }
    mx.random.seed(99)
    warm = DraftModel(tiny)
    export.load_tensors(warm, saved)
    mx.eval(warm.parameters())
    for name, val in fit.tree_flatten(warm.parameters()):
        assert np.allclose(np.array(val.astype(mx.float32)), saved[name])

    traces = TraceSet.load([dump])
    target = TargetRows(
        mx.array(
            rng.standard_normal((tiny.vocabulary, tiny.hidden)).astype(np.float32)
        ),
        mx.array(
            rng.standard_normal((tiny.vocabulary, tiny.hidden)).astype(np.float32)
        ),
    )
    options = fit.FitOptions(
        epochs=3,
        anchors_per_sequence=6,
        anchors_per_step=6,
        learning_rate=3e-3,
        warmup_steps=2,
        checkpoint_every=9,
        log_every=3,
        validation_steps=1,
        max_steps=9,
        cjk_weight=3.0,
        think_boundary_weight=4.0,
        cjk_floor=24,
        control_floor=56,
        think_tokens=(58, 59),
    )
    summary = fit.fit(
        warm, target, traces, TraceSet([]), options, tmp_path / "warm_out"
    )
    assert summary["log"][-1]["loss"] < summary["log"][0]["loss"]
    scratch = fit.build_initial_model(None)
    assert scratch.config.selector_tables is False


def test_head_gather_matches_prefix_slice_and_trains_noncontiguous_cjk_rows():
    tiny = DraftConfig(
        hidden=32,
        intermediate=48,
        layers=2,
        query_heads=4,
        key_value_heads=2,
        head_dim=8,
        window=16,
        vocabulary=64,
        taps=2,
        selector_rank=6,
        selector_top_k=4,
        mask_token_id=63,
        selector_tables=True,
    )
    rng = np.random.default_rng(21)
    target = TargetRows(
        mx.array(rng.standard_normal((64, 32)).astype(np.float32)),
        mx.array(rng.standard_normal((64, 32)).astype(np.float32)),
    )
    tokens = np.array(
        [2, 5, 41, 44, 19, 58, 59, 42, 46, 3, 7, 43, 60, 4, 45, 6], dtype=np.int64
    )
    tap_vals = rng.standard_normal((len(tokens), 64)).astype(np.float32)
    batch = train.build_batch(tokens, tap_vals, np.array([1, 3, 5, 7]), tiny.block)

    mx.random.seed(12)
    model = DraftModel(tiny)
    loss_slice, m_slice = train.losses(model, target, batch, head_rows=32)
    gather_prefix = train.build_head_gather(target.head, np.arange(32, dtype=np.int32))
    loss_gpref, m_gpref = train.losses(model, target, batch, head_rows=gather_prefix)
    mx.eval(loss_slice, loss_gpref, m_slice, m_gpref)
    assert loss_slice.item() == pytest.approx(loss_gpref.item(), rel=1e-5, abs=1e-5)
    assert int(m_slice["positions"].item()) == int(m_gpref["positions"].item())

    # Non-contiguous gather: ASCII 0..15 + CJK 40..47 + control 56..63 (excludes 19).
    noncontig_ids = np.concatenate(
        [np.arange(0, 16), np.arange(40, 48), np.arange(56, 64)]
    ).astype(np.int32)
    gather_cjk = train.build_head_gather(target.head, noncontig_ids)
    loss0, m0 = train.losses(model, target, batch, head_rows=gather_cjk)
    mx.eval(loss0, m0)
    labels_np = np.array(batch.labels)
    valid_np = np.array(batch.valid) & np.isin(labels_np, noncontig_ids)
    assert int(m0["positions"].item()) == int(valid_np.sum())

    from mlx import optimizers

    step = train.make_step(
        model,
        target,
        optimizers.AdamW(learning_rate=4e-3),
        head_rows=gather_cjk,
        eal_weight=0.2,
    )
    for _ in range(30):
        loss_now, _ = step(batch)
        mx.eval(model.parameters(), loss_now)
    assert loss_now.item() < 0.65 * loss0.item()


def test_selector_delta_projections_fold_cleanly_and_resume(tmp_path):
    tiny = DraftConfig(
        hidden=32,
        intermediate=48,
        layers=2,
        query_heads=4,
        key_value_heads=2,
        head_dim=8,
        window=16,
        vocabulary=64,
        taps=2,
        selector_rank=6,
        selector_top_k=4,
        mask_token_id=63,
        selector_tables=True,
    )
    tiny_delta = DraftConfig(**(tiny.describe() | {"selector_delta_projections": True}))
    rng = np.random.default_rng(31)
    target = TargetRows(
        mx.array(rng.standard_normal((64, 32)).astype(np.float32)),
        mx.array(rng.standard_normal((64, 32)).astype(np.float32)),
    )
    mx.random.seed(5)
    base = DraftModel(tiny)
    base_tensors = {
        name: np.array(val.astype(mx.float32))
        for name, val in fit.tree_flatten(base.parameters())
    }

    delta_model = DraftModel(tiny_delta)
    export.load_tensors(delta_model, base_tensors)
    delta_model.candidate_selector.freeze(
        keys=["predecessor_codebook", "successor_codebook"], recurse=False
    )
    mx.eval(delta_model.parameters())

    tokens = np.array([1, 40, 41, 42, 58, 59, 43, 44, 45, 2, 3, 46], dtype=np.int64)
    tap_vals = rng.standard_normal((len(tokens), 64)).astype(np.float16)
    batch = train.build_batch(
        tokens, tap_vals.astype(np.float32), np.array([1, 3, 5]), tiny.block
    )

    l_base, _ = train.losses(base, target, batch)
    l_delta0, _ = train.losses(delta_model, target, batch)
    mx.eval(l_base, l_delta0)
    assert l_base.item() == pytest.approx(l_delta0.item(), rel=1e-6, abs=1e-6)

    dump = tmp_path / "delta_dump" / "taps.bin"
    dump.parent.mkdir()
    with dump.open("wb") as handle:
        taps.write_record(handle, 0, 0, tokens[:3], tap_vals[:3], 0)
        taps.write_record(handle, 2, 3, tokens[3:], tap_vals[3:], 1)
    traces = TraceSet.load([dump])
    gather_spec = train.build_head_gather(
        target.head,
        np.concatenate([np.arange(0, 16), np.arange(40, 64)]).astype(np.int32),
    )
    opts_part1 = fit.FitOptions(
        epochs=4,
        anchors_per_sequence=6,
        anchors_per_step=3,
        learning_rate=4e-3,
        warmup_steps=2,
        checkpoint_every=2,
        log_every=2,
        max_steps=4,
        selector_delta_projections=True,
        freeze_selector_tables=True,
    )
    run_dir = tmp_path / "delta_run"
    fit.fit(
        delta_model,
        target,
        traces,
        TraceSet([]),
        opts_part1,
        run_dir,
        head_spec=gather_spec,
    )
    # Frozen codebooks must remain untouched while delta projections move.
    assert np.allclose(
        np.array(delta_model.candidate_selector.predecessor_codebook),
        base_tensors["candidate_selector.predecessor_codebook"],
    )
    assert not np.allclose(
        np.array(delta_model.candidate_selector.delta_successor_projection.weight), 0.0
    )

    # Resume for 4 more steps.
    resumed = DraftModel(tiny_delta)
    resumed.candidate_selector.freeze(
        keys=["predecessor_codebook", "successor_codebook"], recurse=False
    )
    opts_part2 = fit.FitOptions(**(fit.asdict(opts_part1) | {"max_steps": 8}))
    summary2 = fit.fit(
        resumed,
        target,
        traces,
        TraceSet([]),
        opts_part2,
        run_dir,
        resume=True,
        head_spec=gather_spec,
    )
    assert summary2["resumed_from"] == 4 and summary2["steps"] == 8

    # Fold delta projections into plain codebooks and verify exact forward equivalence.
    folded = DraftModel(tiny)
    folded.update(
        fit.tree_unflatten(
            [
                (
                    k,
                    mx.array(v)
                    if k.endswith("_codebook")
                    else dict(fit.tree_flatten(resumed.parameters()))[k],
                )
                for k, v in [
                    (
                        "candidate_selector.predecessor_codebook",
                        np.array(resumed.candidate_selector.predecessor_codebook)
                        + np.array(target.embedding)
                        @ np.array(
                            resumed.candidate_selector.delta_predecessor_projection.weight
                        ).T,
                    ),
                    (
                        "candidate_selector.successor_codebook",
                        np.array(resumed.candidate_selector.successor_codebook)
                        + np.array(target.head)
                        @ np.array(
                            resumed.candidate_selector.delta_successor_projection.weight
                        ).T,
                    ),
                ]
            ]
        )
    )
    for k, v in fit.tree_flatten(resumed.parameters()):
        if "delta_" not in k and not k.endswith("_codebook"):
            folded.update(fit.tree_unflatten([(k, v)]))
    mx.eval(folded.parameters())

    l_res, _ = train.losses(resumed, target, batch, head_rows=gather_spec)
    l_fold, _ = train.losses(folded, target, batch, head_rows=gather_spec)
    mx.eval(l_res, l_fold)
    assert l_res.item() == pytest.approx(l_fold.item(), rel=1e-5, abs=1e-5)

    p_res = parity.proposals(
        resumed,
        target,
        tokens,
        tap_vals,
        np.array([2, 4]),
        gather_ids=gather_spec.ids,
    )
    p_fold = parity.proposals(
        folded,
        target,
        tokens,
        tap_vals,
        np.array([2, 4]),
        gather_ids=gather_spec.ids,
    )
    assert np.array_equal(p_res, p_fold)


def test_vocab_gather_parser_and_tokenizer_json_reader(tmp_path):
    ids = vocab.parse_vocab_gather("0-255,500-531", 1024)
    assert len(ids) == 288 and ids[0] == 0 and ids[255] == 255 and ids[-1] == 531
    ctrl = vocab.resolve_vocab_gather("control32k", 248_320)
    assert len(ctrl) == 32_768 and 248_068 in ctrl and 248_069 in ctrl
    with pytest.raises(ValueError):
        vocab.parse_vocab_gather("0-254", 1024)

    tok_file = tmp_path / "tokenizer.json"
    tok_file.write_text(
        '{"model": {"vocab": {"a": 0, "b": 1}}, "added_tokens": [{"id": 2, "content": "<think>"}]}',
        encoding="utf-8",
    )
    assert vocab.read_vocabulary_tokens(tok_file) == ["a", "b", "<think>"]
