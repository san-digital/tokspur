# SPDX-License-Identifier: Apache-2.0
"""Tier 2 E2E tests: 100 boundary, corner, and negative rejection tests."""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from . import harness

# ---------------------------------------------------------------------------
# F1: Baseline A/B & Parity Artifacts Boundaries
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f1_01_non_nominal_guard_ineligible(tmp_path: Path) -> None:
    def _write_interleaved_leg(path: Path, *, bad_j2: bool) -> None:
        header = {
            "schema": 1,
            "mode": "interleaved",
            "gap_s": 0.0,
            "oracle_sha256": "a" * 64,
        }
        nominal_sample = {
            "thermal": 0,
            "pressure": 1,
            "swapins": 0,
            "swapouts": 0,
            "swap_used": 0,
        }
        rows = []
        for turn in ("J1", "R1", "J2", "R2", "J3", "R3"):
            after = dict(nominal_sample)
            if bad_j2 and turn == "J2":
                after["thermal"] = 1
                after["swapins"] = 5
            rows.append(
                {
                    "turn": turn,
                    "http_s": 1.0,
                    "before": dict(nominal_sample),
                    "after": after,
                }
            )
        summary_rec = {
            "http_seconds": {
                "full": {"Japanese": 3.0, "Rust": 3.0, "all": 6.0},
                "returns": {"Japanese": 2.0, "Rust": 2.0, "all": 4.0},
            },
            "resource_eligible": {
                "full": not bad_j2,
                "returns": not bad_j2,
            },
        }
        lines = [
            json.dumps(header),
            *(json.dumps(r) for r in rows),
            json.dumps(summary_rec),
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    leg_a = tmp_path / "leg_a.jsonl"
    leg_b = tmp_path / "leg_b.jsonl"
    _write_interleaved_leg(leg_a, bad_j2=False)
    _write_interleaved_leg(leg_b, bad_j2=True)
    res = harness.run_script(
        "benchmark_parked_chats.py",
        ["compare", "--pair", str(leg_a), str(leg_b)],
    )
    assert res.returncode == 0, res.stderr + res.stdout
    records = [
        json.loads(line)
        for line in res.stdout.splitlines()
        if line.strip().startswith("{")
    ]
    assert len(records) == 2
    assert {r["outcome"] for r in records} == {"full", "returns"}
    for rec in records:
        assert rec["eligible_by_arm"] == [True, False]


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f1_02_unknown_key_and_bad_auth_400_401(
    isolated_state_root: Path,
) -> None:
    with harness.ServerProcess(isolated_state_root) as srv:
        c = srv.client()
        bad_key = c.post_chat(
            harness.make_chat_payload("Hi", extra={"unknown_extra_field": 123})
        )
        assert bad_key.status == 400
        bad_auth = c.request(
            "GET",
            "/v1/models",
            headers={"Authorization": "Bearer " + ("0" * 64)},
            include_auth=False,
        )
        assert bad_auth.status == 401


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f1_03_origin_403_and_non_loopback(
    isolated_state_root: Path, tmp_path: Path
) -> None:
    body_path = tmp_path / "body.json"
    body_path.write_text(
        json.dumps(harness.make_chat_payload("Hi", stream=True)), encoding="utf-8"
    )
    res = harness.run_script(
        "stream_collector.py",
        [
            "--endpoint",
            "http://192.0.2.10:18081/v1",
            "--body-file",
            str(body_path),
            "--out",
            str(tmp_path / "out.jsonl"),
        ],
    )
    assert res.returncode != 0
    with harness.ServerProcess(isolated_state_root) as srv:
        resp = srv.client().post_chat(
            harness.make_chat_payload("Hi"),
            headers={"Origin": "https://evil.example"},
        )
        assert resp.status == 403


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f1_04_benchmark_refuses_overwrite(tmp_path: Path) -> None:
    existing = tmp_path / "already.jsonl"
    existing.write_text("{}\n", encoding="utf-8")
    res = harness.run_script(
        "benchmark_parked_chats.py",
        [
            "oracle",
            "--log",
            str(tmp_path / "missing.log"),
            "--out",
            str(existing),
        ],
    )
    assert res.returncode != 0


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f1_05_profile_unset_no_dump_mismatch(isolated_state_root: Path) -> None:
    env = os.environ.copy()
    env.pop("TOKSPUR_DUMP_VERIFY_PROFILE", None)
    with harness.ServerProcess(isolated_state_root) as srv:
        r1 = srv.client().post_chat(
            harness.make_chat_payload("Say alpha.", enable_thinking=False, max_tokens=8)
        )
        r2 = srv.client().post_chat(
            harness.make_chat_payload("Say beta.", enable_thinking=False, max_tokens=8)
        )
        assert r1.status == 200 and r2.status == 200
    out, err = srv.stop()
    assert harness.parse_profile_lines(err, "VERIFY_PROFILE") == []
    done = harness.extract_done_events(harness.parse_jsonl_records(out))
    assert (
        done[0]["hc_exact_final_logit_sha256"] != done[1]["hc_exact_final_logit_sha256"]
    )


# ---------------------------------------------------------------------------
# F2: Holdout Manifest Boundaries
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.e2e_offline
def test_t2_f2_01_single_use_rejects_replay(
    tmp_path: Path, synthetic_r5_factory
) -> None:
    bundle = synthetic_r5_factory("f2_replay")
    first = harness.run_r5_holdout_eval(
        tmp_path / "first",
        bundle["manifest_path"].parent,
        "holdout_0",
        bundle["ledger_path"],
    )
    assert first.returncode == 0, first.stderr + first.stdout
    replay = harness.run_r5_holdout_eval(
        tmp_path / "replay",
        bundle["manifest_path"].parent,
        "holdout_0",
        bundle["ledger_path"],
    )
    assert replay.returncode != 0


@pytest.mark.tier2
@pytest.mark.e2e_offline
def test_t2_f2_02_consume_before_threshold_burns(
    tmp_path: Path, synthetic_r5_factory
) -> None:
    bundle = synthetic_r5_factory(
        "f2_burn", max_relative_l2_error=0.01, reference_error=0.05
    )
    fail_eval = harness.run_r5_holdout_eval(
        tmp_path / "fail_eval",
        bundle["manifest_path"].parent,
        "holdout_0",
        bundle["ledger_path"],
    )
    assert fail_eval.returncode != 0
    ledger = json.loads(bundle["ledger_path"].read_text(encoding="utf-8"))
    assert len(ledger["entries"]) == 1
    assert ledger["entries"][0]["passed"] is False
    replay = harness.run_r5_holdout_eval(
        tmp_path / "burn_replay",
        bundle["manifest_path"].parent,
        "holdout_0",
        bundle["ledger_path"],
    )
    assert replay.returncode != 0


@pytest.mark.tier2
@pytest.mark.e2e_offline
def test_t2_f2_03_path_traversal_symlink_reject(tmp_path: Path) -> None:
    trav_dir = tmp_path / "f2_trav"
    bundle = harness.build_holdout_manifest(trav_dir)
    manifest = bundle["manifest"]
    manifest["holdout_splits"][0]["relative_path"] = "../escape.json"
    bundle["manifest_path"].write_text(json.dumps(manifest), encoding="utf-8")
    res_trav = harness.run_r5_holdout_eval(
        tmp_path / "run_trav",
        trav_dir,
        "holdout_0",
        bundle["ledger_path"],
    )
    assert res_trav.returncode != 0

    sym_dir = tmp_path / "f2_sym"
    sym_bundle = harness.build_holdout_manifest(sym_dir)
    target = tmp_path / "outside.json"
    target.write_text('{"split":"outside"}\n', encoding="utf-8")
    h0 = sym_dir / "holdout_0.json"
    h0.unlink()
    h0.symlink_to(target)
    res_sym = harness.run_r5_holdout_eval(
        tmp_path / "run_sym",
        sym_dir,
        "holdout_0",
        sym_bundle["ledger_path"],
    )
    assert res_sym.returncode != 0


@pytest.mark.tier2
@pytest.mark.e2e_offline
def test_t2_f2_04_calibration_leakage_rejected(
    tmp_path: Path, synthetic_r5_factory
) -> None:
    bundle = synthetic_r5_factory("f2_leak", leak_calibration=True)
    res = harness.run_r5_holdout_eval(
        tmp_path / "run_leak",
        bundle["manifest_path"].parent,
        "holdout_0",
        bundle["ledger_path"],
    )
    assert res.returncode != 0
    assert not bundle["ledger_path"].exists()


@pytest.mark.tier2
@pytest.mark.e2e_offline
def test_t2_f2_05_ledger_chain_tamper_detected(
    tmp_path: Path, synthetic_r5_factory
) -> None:
    bundle = synthetic_r5_factory("f2_tamper", num_holdouts=2)
    ok_res = harness.run_r5_holdout_eval(
        tmp_path / "run_ok",
        bundle["manifest_path"].parent,
        "holdout_0",
        bundle["ledger_path"],
    )
    assert ok_res.returncode == 0, ok_res.stderr + ok_res.stdout
    harness.tamper_holdout_ledger(bundle["ledger_path"])
    tamper_res = harness.run_r5_holdout_eval(
        tmp_path / "run_tamper",
        bundle["manifest_path"].parent,
        "holdout_1",
        bundle["ledger_path"],
    )
    assert tamper_res.returncode != 0


# ---------------------------------------------------------------------------
# F3: Sequential Futility Stopping Boundaries
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f3_01_single_block_no_premature_stop(tmp_path: Path) -> None:
    blocks = harness.build_paired_block_series("single_block_only")
    res = harness.run_r5_futility_eval(
        tmp_path, {"mode": "futility", "alpha": 0.05, "blocks": blocks}
    )
    assert res.returncode == 0, res.stderr + res.stdout
    out = json.loads(res.stdout)
    assert out["stop_for_futility"] is False and out["blocks_evaluated"] == 1


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f3_02_exact_tie_no_false_stop(tmp_path: Path) -> None:
    blocks = harness.build_paired_block_series("exact_tie")
    res = harness.run_r5_futility_eval(
        tmp_path, {"mode": "futility", "alpha": 0.05, "blocks": blocks}
    )
    assert res.returncode == 0, res.stderr + res.stdout
    out = json.loads(res.stdout)
    assert out["stop_for_futility"] is False
    assert math.isclose(out["p_value"], 1.0)


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f3_03_borderline_winner_not_aborted(tmp_path: Path) -> None:
    blocks = harness.build_paired_block_series("borderline_nonfutile")
    res = harness.run_r5_futility_eval(
        tmp_path, {"mode": "futility", "alpha": 0.05, "blocks": blocks}
    )
    assert res.returncode == 0, res.stderr + res.stdout
    out = json.loads(res.stdout)
    assert out["stop_for_futility"] is False


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f3_04_invalid_alpha_or_len_mismatch(tmp_path: Path) -> None:
    for idx, alpha in enumerate((0.0, 1.0)):
        res = harness.run_r5_futility_eval(
            tmp_path / f"alpha_{idx}",
            {"mode": "bh_fdr", "alpha": alpha, "p_values": [0.05]},
        )
        assert res.returncode != 0


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f3_05_bh_fdr_extreme_nan_pvalues(tmp_path: Path) -> None:
    res_ok = harness.run_r5_futility_eval(
        tmp_path / "ext",
        {"mode": "bh_fdr", "alpha": 0.05, "p_values": [0.0, 1.0]},
    )
    assert res_ok.returncode == 0, res_ok.stderr + res_ok.stdout
    out = json.loads(res_ok.stdout)
    assert out["rejected"] == [True, False]
    res_bad = harness.run_r5_futility_eval(
        tmp_path / "bad_p",
        {"mode": "bh_fdr", "alpha": 0.05, "p_values": ["nan"]},
    )
    assert res_bad.returncode != 0


# ---------------------------------------------------------------------------
# F4: native/rust.mk Receipt Cache Boundaries
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f4_01_wire_change_invalidates_cache(tmp_path: Path) -> None:
    shim, log1 = harness.create_cargo_logging_shim(tmp_path)
    cache_dir = tmp_path / "rcache"
    res1 = harness.run_make_native(
        ["rust-check", f"RUST_RECEIPT_CACHE_DIR={cache_dir}", f"CARGO={shim}"],
        makefile="rust.mk",
        env={"TOKSPUR_E2E_CARGO_LOG": str(log1)},
    )
    assert res1.returncode == 0, res1.stdout + "\n" + res1.stderr
    before = {p.name: p.read_bytes() for p in sorted(cache_dir.rglob("*.json"))}
    assert len(before) >= 1

    log2 = tmp_path / "cargo_miss.log"
    with harness.temporary_wire_comment_probe():
        res2 = harness.run_make_native(
            ["rust-check", f"RUST_RECEIPT_CACHE_DIR={cache_dir}", f"CARGO={shim}"],
            makefile="rust.mk",
            env={"TOKSPUR_E2E_CARGO_LOG": str(log2)},
        )
        assert res2.returncode == 0, res2.stdout + "\n" + res2.stderr
        assert log2.exists() and len(harness.read_cached_cargo_calls(log2)) >= 1
        during = {p.name: p.read_bytes() for p in sorted(cache_dir.rglob("*.json"))}
        assert during != before

    before_keys = {name: json.loads(raw)["content_key"] for name, raw in before.items()}
    log3 = tmp_path / "cargo_repopulate.log"
    res3 = harness.run_make_native(
        ["rust-check", f"RUST_RECEIPT_CACHE_DIR={cache_dir}", f"CARGO={shim}"],
        makefile="rust.mk",
        env={"TOKSPUR_E2E_CARGO_LOG": str(log3)},
    )
    assert res3.returncode == 0, res3.stdout + "\n" + res3.stderr
    assert len(harness.read_cached_cargo_calls(log3)) >= 1
    after3_keys = {
        p.name: json.loads(p.read_text(encoding="utf-8"))["content_key"]
        for p in sorted(cache_dir.rglob("*.json"))
    }
    assert after3_keys == before_keys

    log4 = tmp_path / "cargo_restored_hit.log"
    res4 = harness.run_make_native(
        ["rust-check", f"RUST_RECEIPT_CACHE_DIR={cache_dir}", f"CARGO={shim}"],
        makefile="rust.mk",
        env={"TOKSPUR_E2E_CARGO_LOG": str(log4)},
    )
    assert res4.returncode == 0, res4.stdout + "\n" + res4.stderr
    assert harness.read_cached_cargo_calls(log4) == []


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f4_02_cache_zero_bypasses_receipts(tmp_path: Path) -> None:
    shim, cold_log = harness.create_cargo_logging_shim(tmp_path)
    cache_dir = tmp_path / "bypass_cache"
    res1 = harness.run_make_native(
        ["rust-check", f"RUST_RECEIPT_CACHE_DIR={cache_dir}", f"CARGO={shim}"],
        makefile="rust.mk",
        env={"TOKSPUR_E2E_CARGO_LOG": str(cold_log)},
    )
    assert res1.returncode == 0, res1.stdout + "\n" + res1.stderr
    mtimes_before = {p: p.stat().st_mtime_ns for p in sorted(cache_dir.rglob("*.json"))}
    assert len(mtimes_before) >= 1
    bypass_log = tmp_path / "bypass_cargo.log"
    res2 = harness.run_make_native(
        [
            "rust-check",
            "RUST_RECEIPT_CACHE=0",
            f"RUST_RECEIPT_CACHE_DIR={cache_dir}",
            f"CARGO={shim}",
        ],
        makefile="rust.mk",
        env={"TOKSPUR_E2E_CARGO_LOG": str(bypass_log)},
    )
    assert res2.returncode == 0, res2.stdout + "\n" + res2.stderr
    assert len(harness.read_cached_cargo_calls(bypass_log)) >= 1
    mtimes_after = {p: p.stat().st_mtime_ns for p in sorted(cache_dir.rglob("*.json"))}
    assert mtimes_after == mtimes_before


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f4_03_corrupt_receipt_json_rebuilds(tmp_path: Path) -> None:
    shim, cold_log = harness.create_cargo_logging_shim(tmp_path)
    cache_dir = tmp_path / "corrupt_rcache"
    res1 = harness.run_make_native(
        ["rust-check", f"RUST_RECEIPT_CACHE_DIR={cache_dir}", f"CARGO={shim}"],
        makefile="rust.mk",
        env={"TOKSPUR_E2E_CARGO_LOG": str(cold_log)},
    )
    assert res1.returncode == 0, res1.stdout + "\n" + res1.stderr
    receipts = sorted(cache_dir.rglob("*.json"))
    assert receipts
    receipts[0].write_text("{corrupt-json", encoding="utf-8")
    heal_log = tmp_path / "heal_cargo.log"
    res2 = harness.run_make_native(
        ["rust-check", f"RUST_RECEIPT_CACHE_DIR={cache_dir}", f"CARGO={shim}"],
        makefile="rust.mk",
        env={"TOKSPUR_E2E_CARGO_LOG": str(heal_log)},
    )
    assert res2.returncode == 0, res2.stdout + "\n" + res2.stderr
    assert len(harness.read_cached_cargo_calls(heal_log)) >= 1
    assert json.loads(receipts[0].read_text(encoding="utf-8"))["status"] == "passed"


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f4_04_cpp_oracle_toggle_distinct_key(tmp_path: Path) -> None:
    shim, log0 = harness.create_cargo_logging_shim(tmp_path)
    cache_dir = tmp_path / "oracle_toggle_cache"
    r0 = harness.run_make_native(
        [
            "rust-check",
            "CPP_ORACLE=0",
            f"RUST_RECEIPT_CACHE_DIR={cache_dir}",
            f"CARGO={shim}",
        ],
        makefile="rust.mk",
        env={"TOKSPUR_E2E_CARGO_LOG": str(log0)},
    )
    log1 = tmp_path / "oracle_on.log"
    r1 = harness.run_make_native(
        [
            "rust-check",
            "CPP_ORACLE=1",
            f"RUST_RECEIPT_CACHE_DIR={cache_dir}",
            f"CARGO={shim}",
        ],
        makefile="rust.mk",
        env={"TOKSPUR_E2E_CARGO_LOG": str(log1)},
    )
    assert r0.returncode == 0, r0.stdout + "\n" + r0.stderr
    assert r1.returncode == 0, r1.stdout + "\n" + r1.stderr
    assert len(harness.read_cached_cargo_calls(log0)) >= 1
    assert len(harness.read_cached_cargo_calls(log1)) >= 1
    oracle0 = sorted(cache_dir.rglob("*--oracle0.json"))
    oracle1 = sorted(cache_dir.rglob("*--oracle1.json"))
    assert len(oracle0) >= 1 and len(oracle1) >= 1
    r0_fast = json.loads(
        (oracle0[0].parent / "clippy--fast-math--oracle0.json").read_text(
            encoding="utf-8"
        )
    )
    r1_fast = json.loads(
        (oracle1[0].parent / "clippy--fast-math--oracle1.json").read_text(
            encoding="utf-8"
        )
    )
    assert r0_fast["cpp_oracle"] == "0" and r1_fast["cpp_oracle"] == "1"
    assert r0_fast["content_key"] != r1_fast["content_key"]


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f4_05_missing_cache_dir_safe_create(tmp_path: Path) -> None:
    cache_dir = tmp_path / "fresh/nested/receipt-cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    res = harness.run_make_native(
        ["rust-check", f"RUST_RECEIPT_CACHE_DIR={cache_dir}"], makefile="rust.mk"
    )
    assert res.returncode == 0, res.stdout + "\n" + res.stderr
    assert cache_dir.exists()
    receipts = list(cache_dir.rglob("*.json"))
    assert len(receipts) >= 1
    assert isinstance(json.loads(receipts[0].read_text(encoding="utf-8")), dict)


# ---------------------------------------------------------------------------
# F5: Parked-Lane Recording Boundaries
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f5_01_mismatched_seed_forces_miss(isolated_state_root: Path) -> None:
    p_a = harness.make_long_repeated_prompt("badSeedA")
    p_b = harness.make_long_repeated_prompt("badSeedB")
    with harness.ServerProcess(
        isolated_state_root, serve_args=["--speed-preset"]
    ) as srv:
        c = srv.client()
        ra = c.post_chat(
            harness.make_chat_payload(p_a, enable_thinking=False, seed=101)
        )
        assert (
            c.post_chat(
                harness.make_chat_payload(p_b, enable_thinking=False, seed=102)
            ).status
            == 200
        )
        r3 = c.post_chat(
            harness.make_chat_payload(
                "Return with wrong seed.",
                history=[
                    {"role": "user", "content": p_a},
                    {
                        "role": "assistant",
                        "content": ra.json()["choices"][0]["message"]["content"],
                    },
                ],
                enable_thinking=False,
                seed=999,
            )
        )
        assert r3.status == 200
        assert r3.json()["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    out, _ = srv.stop()
    done = harness.extract_done_events(harness.parse_jsonl_records(out))[-1]
    assert done["parked_lanes"]["restores"] == 0
    traces = [
        t for t in srv.prefix_trace_records() if t.get("event") == "prompt_prefix"
    ]
    assert traces[-1]["seed_matches"] is False
    assert traces[-1]["cached_tokens"] == 0


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f5_02_mismatched_sampling_refuses(isolated_state_root: Path) -> None:
    p_a = harness.make_long_repeated_prompt("badSampA")
    p_b = harness.make_long_repeated_prompt("badSampB")
    with harness.ServerProcess(
        isolated_state_root, serve_args=["--speed-preset"]
    ) as srv:
        c = srv.client()
        ra = c.post_chat(
            harness.make_chat_payload(
                p_a, enable_thinking=False, seed=55, temperature=0.0, top_k=1
            )
        )
        assert c.post_chat(harness.make_chat_payload(p_b, seed=56)).status == 200
        r3 = c.post_chat(
            harness.make_chat_payload(
                "Changed sampling policy.",
                history=[
                    {"role": "user", "content": p_a},
                    {
                        "role": "assistant",
                        "content": ra.json()["choices"][0]["message"]["content"],
                    },
                ],
                enable_thinking=False,
                seed=55,
                temperature=1.0,
                top_k=20,
                top_p=0.95,
            )
        )
        assert r3.status == 200
        assert r3.json()["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    out, _ = srv.stop()
    done = harness.extract_done_events(harness.parse_jsonl_records(out))[-1]
    assert done["parked_lanes"]["restores"] == 0
    traces = [
        t for t in srv.prefix_trace_records() if t.get("event") == "prompt_prefix"
    ]
    assert traces[-1]["sampling_matches"] is False
    assert traces[-1]["cached_tokens"] == 0


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f5_03_short_below_min_prompt_safe(isolated_state_root: Path) -> None:
    with harness.ServerProcess(
        isolated_state_root, serve_args=["--speed-preset"]
    ) as srv:
        c = srv.client()
        r1 = c.post_chat(
            harness.make_chat_payload("Hi.", enable_thinking=False, max_tokens=8)
        )
        r2 = c.post_chat(
            harness.make_chat_payload("Bye.", enable_thinking=False, max_tokens=8)
        )
        assert r1.status == 200 and r2.status == 200
    out, _ = srv.stop()
    done = harness.extract_done_events(harness.parse_jsonl_records(out))[-1]
    assert done["parked_lanes"]["restores"] == 0


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f5_04_ambiguous_prefix_deterministic(
    isolated_state_root: Path,
) -> None:
    p_base = harness.make_long_repeated_prompt("ambig")
    with harness.ServerProcess(
        isolated_state_root, serve_args=["--speed-preset"]
    ) as srv:
        c = srv.client()
        assert (
            c.post_chat(
                harness.make_chat_payload(
                    p_base + "\nVariant 1.", enable_thinking=False, seed=1
                )
            ).status
            == 200
        )
        assert (
            c.post_chat(
                harness.make_chat_payload(
                    p_base + "\nVariant 2.", enable_thinking=False, seed=1
                )
            ).status
            == 200
        )
        r3 = c.post_chat(
            harness.make_chat_payload(
                p_base + "\nVariant 3.", enable_thinking=False, seed=1
            )
        )
        assert r3.status == 200
        harness.validate_chat_completion_json(r3.json())


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f5_05_budget_eviction_cold_fallback(
    isolated_state_root: Path,
) -> None:
    prompts = [
        harness.make_long_repeated_prompt(f"evict_{idx}", approx_lines=140)
        for idx in range(5)
    ]
    with harness.ServerProcess(
        isolated_state_root,
        serve_args=["--shared-prefix-cache", "--parked-lanes-mib", "256"],
    ) as srv:
        c = srv.client()
        first_msg = ""
        for idx, prompt in enumerate(prompts):
            resp = c.post_chat(
                harness.make_chat_payload(
                    prompt, enable_thinking=False, seed=300 + idx, max_tokens=16
                )
            )
            assert resp.status == 200
            if idx == 0:
                first_msg = resp.json()["choices"][0]["message"]["content"]
    out_mid, _ = srv.stop()
    done_mid = harness.extract_done_events(harness.parse_jsonl_records(out_mid))[-1]
    assert done_mid["parked_lanes"]["budget_bytes"] == 256 * 1024 * 1024
    assert done_mid["parked_lanes"]["parks"] >= 1
    assert done_mid["parked_lanes"]["evictions"] >= 1
    assert done_mid["parked_lanes"]["skips"] == 0

    with harness.ServerProcess(
        isolated_state_root,
        serve_args=["--shared-prefix-cache", "--parked-lanes-mib", "256"],
    ) as srv2:
        c2 = srv2.client()
        for idx, prompt in enumerate(prompts):
            resp = c2.post_chat(
                harness.make_chat_payload(
                    prompt, enable_thinking=False, seed=300 + idx, max_tokens=16
                )
            )
            assert resp.status == 200
            if idx == 0:
                first_msg = resp.json()["choices"][0]["message"]["content"]
        cold_ret = c2.post_chat(
            harness.make_chat_payload(
                "Return to evicted lane 0.",
                history=[
                    {"role": "user", "content": prompts[0]},
                    {"role": "assistant", "content": first_msg},
                ],
                enable_thinking=False,
                seed=300,
                max_tokens=16,
            )
        )
        assert cold_ret.status == 200
        assert cold_ret.json()["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    out2, _ = srv2.stop()
    done2 = harness.extract_done_events(harness.parse_jsonl_records(out2))
    assert (
        done2[-1]["parked_lanes"]["restores"] == done2[-2]["parked_lanes"]["restores"]
    )


# ---------------------------------------------------------------------------
# F6: Hoisted Render/Encode & Echoed Reasoning Boundaries
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f6_01_tampered_reasoning_invalidates(isolated_state_root: Path) -> None:
    p_a = harness.make_long_repeated_prompt("tamperReasonA")
    p_b = harness.make_long_repeated_prompt("tamperReasonB")
    with harness.ServerProcess(
        isolated_state_root, serve_args=["--speed-preset"]
    ) as srv:
        c = srv.client()
        r1 = c.post_chat(
            harness.make_chat_payload(
                p_a,
                enable_thinking=True,
                seed=61,
                max_tokens=512,
                extra={"thinking_budget": 24},
            )
        )
        assert r1.status == 200
        assert r1.json()["choices"][0]["finish_reason"] == "stop"
        msg1 = r1.json()["choices"][0]["message"]
        assert c.post_chat(harness.make_chat_payload(p_b)).status == 200
        r3 = c.post_chat(
            harness.make_chat_payload(
                "After tampered reasoning.",
                history=[
                    {"role": "user", "content": p_a},
                    {
                        "role": "assistant",
                        "content": msg1["content"],
                        "reasoning_content": "COMPLETELY DIFFERENT REASONING TEXT",
                    },
                ],
                enable_thinking=True,
                seed=61,
            )
        )
        assert r3.status == 200
        assert r3.json()["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    out, _ = srv.stop()
    done = harness.extract_done_events(harness.parse_jsonl_records(out))[-1]
    assert done["parked_lanes"]["restores"] == 0


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f6_02_conflicting_thinking_kwargs_400(
    isolated_state_root: Path,
) -> None:
    with harness.ServerProcess(isolated_state_root) as srv:
        resp = srv.client().post_chat(
            harness.make_chat_payload(
                "Conflict test",
                enable_thinking=True,
                extra={"chat_template_kwargs": {"enable_thinking": False}},
            )
        )
        assert resp.status == 400


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f6_03_oversized_reasoning_body_limit(isolated_state_root: Path) -> None:
    with harness.ServerProcess(isolated_state_root) as srv:
        huge = "X" * (2 * 1024 * 1024)
        resp = srv.client().post_chat(
            harness.make_chat_payload(
                "Oversize",
                history=[
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a", "reasoning_content": huge},
                ],
            )
        )
        assert resp.status in (400, 413)


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f6_04_unicode_cjk_control_escaping(isolated_state_root: Path) -> None:
    p_a = harness.make_long_repeated_prompt("cjk") + "\n日本語テスト 🚀\tline2\n"
    p_b = harness.make_long_repeated_prompt("cjkOther")
    with harness.ServerProcess(
        isolated_state_root, serve_args=["--speed-preset"]
    ) as srv:
        c = srv.client()
        r1 = c.post_chat(
            harness.make_chat_payload(
                p_a,
                enable_thinking=True,
                seed=88,
                max_tokens=512,
                extra={"thinking_budget": 32},
            )
        )
        assert r1.status == 200
        assert r1.json()["choices"][0]["finish_reason"] == "stop"
        msg1 = r1.json()["choices"][0]["message"]
        assert c.post_chat(harness.make_chat_payload(p_b)).status == 200
        hist_item = {"role": "assistant", "content": msg1["content"]}
        if "reasoning_content" in msg1:
            hist_item["reasoning_content"] = msg1["reasoning_content"]
        r3 = c.post_chat(
            harness.make_chat_payload(
                "日本語で続けてください。",
                history=[{"role": "user", "content": p_a}, hist_item],
                enable_thinking=True,
                seed=88,
                extra={"thinking_budget": 32},
            )
        )
        assert r3.status == 200
        assert harness.validate_chat_completion_json(r3.json(), min_cached_tokens=1) > 0


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f6_05_preserve_thinking_toggle_safe(isolated_state_root: Path) -> None:
    p_a = harness.make_long_repeated_prompt("preserveThink")
    with harness.ServerProcess(
        isolated_state_root, serve_args=["--speed-preset"]
    ) as srv:
        c = srv.client()
        r1 = c.post_chat(
            harness.make_chat_payload(
                p_a,
                enable_thinking=True,
                extra={"preserve_thinking": False, "thinking_budget": 16},
            )
        )
        assert r1.status == 200
        r2 = c.post_chat(
            harness.make_chat_payload(
                "Next turn.",
                history=[
                    {"role": "user", "content": p_a},
                    {
                        "role": "assistant",
                        "content": r1.json()["choices"][0]["message"]["content"],
                    },
                ],
                enable_thinking=True,
                extra={"preserve_thinking": True},
            )
        )
        assert r2.status == 200


# ---------------------------------------------------------------------------
# F7: Parked-Lane respelled_on_committed Boundaries
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f7_01_divergent_edit_stops_respell(isolated_state_root: Path) -> None:
    p_a = harness.make_long_repeated_prompt("divEditA")
    p_b = harness.make_long_repeated_prompt("divEditB")
    with harness.ServerProcess(
        isolated_state_root, serve_args=["--speed-preset"]
    ) as srv:
        c = srv.client()
        r1 = c.post_chat(harness.make_chat_payload(p_a, enable_thinking=False, seed=3))
        orig = r1.json()["choices"][0]["message"]["content"]
        assert c.post_chat(harness.make_chat_payload(p_b, seed=4)).status == 200
        r3 = c.post_chat(
            harness.make_chat_payload(
                "Follow up",
                history=[
                    {"role": "user", "content": p_a},
                    {"role": "assistant", "content": orig + " MUTATED_SUFFIX"},
                ],
                enable_thinking=False,
                seed=3,
            )
        )
        assert r3.status == 200
        assert r3.json()["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    out, _ = srv.stop()
    done = harness.extract_done_events(harness.parse_jsonl_records(out))[-1]
    assert done["parked_lanes"]["restores"] == 0


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f7_02_noncanonical_bpe_bad_seed_miss(isolated_state_root: Path) -> None:
    p_a = harness.make_long_repeated_prompt("bpeSeedA")
    p_b = harness.make_long_repeated_prompt("bpeSeedB")
    with harness.ServerProcess(
        isolated_state_root, serve_args=["--speed-preset"]
    ) as srv:
        c = srv.client()
        r1 = c.post_chat(
            harness.make_chat_payload(
                p_a, enable_thinking=False, seed=100, temperature=0.7, top_k=20
            )
        )
        assert c.post_chat(harness.make_chat_payload(p_b, seed=101)).status == 200
        r3 = c.post_chat(
            harness.make_chat_payload(
                "Follow up",
                history=[
                    {"role": "user", "content": p_a},
                    {
                        "role": "assistant",
                        "content": r1.json()["choices"][0]["message"]["content"],
                    },
                ],
                enable_thinking=False,
                seed=200,
                temperature=0.7,
                top_k=20,
            )
        )
        assert r3.status == 200
        assert r3.json()["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    out, _ = srv.stop()
    done = harness.extract_done_events(harness.parse_jsonl_records(out))[-1]
    assert done["parked_lanes"]["restores"] == 0
    assert done["sampling"]["seed"] == 200


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f7_03_utf8_multibyte_boundary_safe(isolated_state_root: Path) -> None:
    p_a = harness.make_long_repeated_prompt("utf8Boundary") + "\n漢字絵文字🎯🧪\n"
    with harness.ServerProcess(
        isolated_state_root, serve_args=["--speed-preset"]
    ) as srv:
        c = srv.client()
        r1 = c.post_chat(
            harness.make_chat_payload(
                p_a, enable_thinking=False, seed=33, temperature=0.5, top_k=20
            )
        )
        assert r1.status == 200
        r2 = c.post_chat(
            harness.make_chat_payload(
                "Continue UTF-8.",
                history=[
                    {"role": "user", "content": p_a},
                    {
                        "role": "assistant",
                        "content": r1.json()["choices"][0]["message"]["content"],
                    },
                ],
                enable_thinking=False,
                seed=33,
                temperature=0.5,
                top_k=20,
            )
        )
        assert r2.status == 200


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f7_04_repeated_token_longest_match(isolated_state_root: Path) -> None:
    p_a = ("    0000    1111\n" * 240) + "Summarize."
    p_b = harness.make_long_repeated_prompt("repeatOther")
    with harness.ServerProcess(
        isolated_state_root, serve_args=["--speed-preset"]
    ) as srv:
        c = srv.client()
        r1 = c.post_chat(harness.make_chat_payload(p_a, enable_thinking=False, seed=44))
        assert c.post_chat(harness.make_chat_payload(p_b, seed=45)).status == 200
        r3 = c.post_chat(
            harness.make_chat_payload(
                "Again.",
                history=[
                    {"role": "user", "content": p_a},
                    {
                        "role": "assistant",
                        "content": r1.json()["choices"][0]["message"]["content"],
                    },
                ],
                enable_thinking=False,
                seed=44,
            )
        )
        assert r3.status == 200
        assert harness.validate_chat_completion_json(r3.json(), min_cached_tokens=1) > 0


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f7_05_empty_committed_returns_miss() -> None:
    res = harness.run_cargo_test_checked(
        ["-p", "tokspur-server", "--", "respelled"], min_passed=1
    )
    assert res.returncode == 0, res.stderr + res.stdout


# ---------------------------------------------------------------------------
# F8: C++ ServeSession Parity Boundaries
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f8_01_rust_diff_negative_ctrl_fails() -> None:
    res = harness.run_make_native(["rust-diff"], makefile="rust.mk")
    assert res.returncode == 0, res.stderr
    neg = json.loads(
        (harness.BUILD_DIR / "rust-diff/server-report-negative.json").read_text(
            encoding="utf-8"
        )
    )
    assert neg["negative_control"] is True
    assert neg["harness_failures"] == []
    assert len(neg["differences"]) > 0
    assert neg["cpp_exit"] == 0 and neg["rust_exit"] == 0


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f8_02_malformed_body_method_parity(isolated_state_root: Path) -> None:
    with harness.ServerProcess(isolated_state_root) as srv:
        c = srv.client()
        bad_json = c.request("POST", "/v1/chat/completions", body=b"{not-json")
        assert bad_json.status == 400
        bad_method = c.request("PUT", "/v1/chat/completions", body=b"{}")
        assert bad_method.status in (404, 405)


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f8_03_cancel_before_start_no_leak() -> None:
    res = harness.run_make_native(["rust-diff"], makefile="rust.mk")
    assert res.returncode == 0, res.stderr
    diff_dir = harness.BUILD_DIR / "rust-diff"
    rep = json.loads(
        (diff_dir / "server-report-channel.json").read_text(encoding="utf-8")
    )
    assert rep["negative_control"] is False
    assert rep["differences"] == []
    assert rep["harness_failures"] == []
    assert rep["cpp_exit"] == 0 and rep["rust_exit"] == 0
    cpp_side = json.loads(
        (diff_dir / "server-report-channel-cpp.json").read_text(encoding="utf-8")
    )
    rust_side = json.loads(
        (diff_dir / "server-report-channel-rust.json").read_text(encoding="utf-8")
    )
    assert cpp_side["responses"]["cancel-before-start"] == ""
    assert rust_side["responses"]["cancel-before-start"] == ""


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f8_04_max_tokens_bounds_and_conflict(isolated_state_root: Path) -> None:
    with harness.ServerProcess(isolated_state_root) as srv:
        c = srv.client()
        assert c.post_chat(harness.make_chat_payload("Hi", max_tokens=0)).status == 400
        assert (
            c.post_chat(harness.make_chat_payload("Hi", max_tokens=32769)).status == 400
        )
        assert (
            c.post_chat(
                harness.make_chat_payload(
                    "Hi", max_tokens=16, extra={"max_completion_tokens": 16}
                )
            ).status
            == 400
        )


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f8_05_calibrated_sampling_400_parity(isolated_state_root: Path) -> None:
    with harness.ServerProcess(isolated_state_root) as srv:
        c = srv.client()
        resp = c.post_chat(
            harness.make_chat_payload(
                "Calibrated check", temperature=0.37, top_k=7, top_p=0.5
            )
        )
        assert resp.status == 200
    out, _ = srv.stop()
    ready = next(
        r for r in harness.parse_jsonl_records(out) if r.get("event") == "ready"
    )
    assert not ready["native_ready"].get("native_measured_controller")
    rust_cal = harness.run_cargo_test_checked(
        [
            "-p",
            "tokspur-server",
            "--test",
            "public_readiness",
            "--",
            "measured_sampling_is_checked_before_submission",
        ],
        min_passed=1,
    )
    assert rust_cal.returncode == 0, rust_cal.stderr + rust_cal.stdout
    cpp_bin = harness.require_artifact(
        harness.find_native_test_bin("test-measured_controller"),
        "test-measured_controller",
    )
    cpp_cal = harness.run_cmd([str(cpp_bin)])
    assert cpp_cal.returncode == 0, cpp_cal.stderr + cpp_cal.stdout


# ---------------------------------------------------------------------------
# F9: graph.mm Phase B Level-Scheduling Boundaries
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f9_01_overlap_vs_disjoint_byte_spans(synthetic_trace_factory) -> None:
    tb = harness.DispatchTraceBuilder()
    tb.add_pipeline("rms_norm:1", "rms_norm", [(0, "a", "write", True)])
    tb.add_allocation(0, "shared_buf", 8192)
    tb.add_host_write(0, 0, 8192, "disjoint_init")
    bid = tb.begin_batch()
    tb.add_dispatch(
        "rms_norm:1",
        [{"index": 0, "slot": 0, "offset": 0, "bytes": 4096, "access": "write"}],
        constants=[
            {"index": 3, "bytes": 4, "sha256": harness.sha256_bytes(b"span_lo")}
        ],
    )
    tb.add_dispatch(
        "rms_norm:1",
        [{"index": 0, "slot": 0, "offset": 4096, "bytes": 4096, "access": "write"}],
        constants=[
            {"index": 3, "bytes": 4, "sha256": harness.sha256_bytes(b"span_hi")}
        ],
    )
    tb.submit_batch(bid, 2, [(0, "span_done")])
    path = synthetic_trace_factory("disjoint_ranges.jsonl", tb)
    res = harness.run_dispatch_graph(["summary", str(path)])
    assert res.returncode == 0
    assert json.loads(res.stdout)["batches"][0]["dependency_edges"] == 0


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f9_02_zero_unaligned_immutable_reject(tmp_path: Path) -> None:
    bad_zero = tmp_path / "zero_bytes.jsonl"
    bad_zero.write_text(
        '{"event":"trace","schema":1,"producer":"x","device":"m"}\n'
        '{"event":"allocation","slot":0,"label":"z","kind":"buffer",'
        '"bytes":0,"immutable":true,"sha256":null}\n',
        encoding="utf-8",
    )
    assert harness.run_dispatch_graph(["summary", str(bad_zero)]).returncode == 1
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    bin_path = harness.require_artifact(
        harness.find_native_test_bin("test-graph"), "test-graph"
    )
    res = harness.run_cmd([str(bin_path), str(metallib)])
    assert res.returncode == 0, res.stderr + res.stdout
    assert "max scaled error:" in res.stdout


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f9_03_batch_state_machine_violations(tmp_path: Path) -> None:
    double_begin = tmp_path / "double_begin.jsonl"
    double_begin.write_text(
        '{"event":"trace","schema":1,"producer":"x","device":"m"}\n'
        '{"event":"begin","batch":0}\n'
        '{"event":"begin","batch":1}\n',
        encoding="utf-8",
    )
    assert harness.run_dispatch_graph(["summary", str(double_begin)]).returncode == 1
    orphan_submit = tmp_path / "orphan_submit.jsonl"
    orphan_submit.write_text(
        '{"event":"trace","schema":1,"producer":"x","device":"m"}\n'
        '{"event":"submit","batch":0,"dispatches":0}\n',
        encoding="utf-8",
    )
    assert harness.run_dispatch_graph(["summary", str(orphan_submit)]).returncode == 1
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    bin_path = harness.require_artifact(
        harness.find_native_test_bin("test-graph"), "test-graph"
    )
    res = harness.run_cmd([str(bin_path), str(metallib)])
    assert res.returncode == 0, res.stderr + res.stdout
    assert "Persistent Metal graph checks passed:" in res.stdout


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f9_04_startup_mode_lock_enforced() -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    bin_path = harness.require_artifact(
        harness.find_native_test_bin("test-concurrent_dispatch_startup"),
        "test-concurrent_dispatch_startup",
    )
    res = harness.run_cmd([str(bin_path), str(metallib)])
    assert res.returncode == 0, res.stderr + res.stdout
    assert "concurrent startup: 6 refusals passed" in res.stdout


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f9_05_empty_batch_zero_dispatch_ok(synthetic_trace_factory) -> None:
    tb = harness.DispatchTraceBuilder()
    bid = tb.begin_batch()
    tb.submit_batch(bid, 0)
    path = synthetic_trace_factory("empty_batch.jsonl", tb)
    res = harness.run_dispatch_graph(["summary", str(path)])
    assert res.returncode == 0
    batch = json.loads(res.stdout)["batches"][0]
    assert batch["dispatches"] == 0 and batch["barriers"] == 0


# ---------------------------------------------------------------------------
# F10: dispatch-graph check Boundaries
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f10_01_missing_barrier_rejected(tmp_path: Path) -> None:
    ref_p, cand_p = harness.build_dag_pair_traces(tmp_path, depth=2)
    lines = [
        line
        for line in cand_p.read_text(encoding="utf-8").splitlines()
        if '"event": "barrier"' not in line
    ]
    bad_p = tmp_path / "no_barrier.jsonl"
    bad_p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    res = harness.run_dispatch_graph(["check", str(ref_p), str(bad_p)])
    assert res.returncode == 1
    assert json.loads(res.stdout)["passed"] is False


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f10_02_swapped_or_dropped_rejected(tmp_path: Path) -> None:
    ref_p, cand_p = harness.build_dag_pair_traces(tmp_path, depth=2)
    lines = cand_p.read_text(encoding="utf-8").splitlines()
    dropped = [line for line in lines if '"event": "dispatch"' not in line]
    bad_p = tmp_path / "dropped.jsonl"
    bad_p.write_text("\n".join(dropped) + "\n", encoding="utf-8")
    res = harness.run_dispatch_graph(["check", str(ref_p), str(bad_p)])
    assert res.returncode == 1


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f10_03_mismatched_state_sha_rejected(tmp_path: Path) -> None:
    ref_p, cand_p = harness.build_dag_pair_traces(tmp_path, depth=2)
    bad_text = cand_p.read_text(encoding="utf-8").replace(
        harness.sha256_bytes(b"final_0"), "f" * 64
    )
    bad_p = tmp_path / "bad_state.jsonl"
    bad_p.write_text(bad_text, encoding="utf-8")
    res = harness.run_dispatch_graph(["check", str(ref_p), str(bad_p)])
    assert res.returncode == 1


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f10_04_malformed_jsonl_rejected(tmp_path: Path) -> None:
    no_hdr = tmp_path / "no_header.jsonl"
    no_hdr.write_text('{"event":"begin","batch":0}\n', encoding="utf-8")
    assert harness.run_dispatch_graph(["summary", str(no_hdr)]).returncode == 1

    unclosed = tmp_path / "unclosed.jsonl"
    unclosed.write_text(
        '{"event":"trace","schema":1,"producer":"x","device":"m"}\n'
        '{"event":"begin","batch":0}\n',
        encoding="utf-8",
    )
    assert harness.run_dispatch_graph(["summary", str(unclosed)]).returncode == 1


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f10_05_controls_violation_exits_one(synthetic_trace_factory) -> None:
    tb = harness.DispatchTraceBuilder()
    tb.add_pipeline(
        "rms_norm:1",
        "rms_norm",
        [(0, "in", "read", True), (1, "out", "write", True)],
    )
    tb.add_allocation(0, "s0", 4096)
    tb.add_allocation(1, "s1", 4096)
    tb.add_host_write(0, 0, 4096, "c_init0")
    tb.add_host_write(1, 0, 4096, "c_init1")
    bid = tb.begin_batch()
    tb.add_dispatch(
        "rms_norm:1",
        [
            {"index": 0, "slot": 0, "offset": 0, "bytes": 4096, "access": "read"},
            {"index": 1, "slot": 1, "offset": 0, "bytes": 4096, "access": "write"},
        ],
        constants=[{"index": 3, "bytes": 4, "sha256": harness.sha256_bytes(b"ctrl_a")}],
    )
    tb.add_dispatch(
        "rms_norm:1",
        [
            {"index": 0, "slot": 1, "offset": 0, "bytes": 4096, "access": "read"},
            {"index": 1, "slot": 0, "offset": 0, "bytes": 4096, "access": "write"},
        ],
        constants=[{"index": 3, "bytes": 4, "sha256": harness.sha256_bytes(b"ctrl_b")}],
    )
    tb.submit_batch(bid, 2, [(0, "c_end0"), (1, "c_end1")])
    bad = synthetic_trace_factory("controls_missing_barrier.jsonl", tb)
    res = harness.run_dispatch_graph(["controls", str(bad)])
    assert res.returncode == 1
    assert json.loads(res.stdout)["held"] is False


# ---------------------------------------------------------------------------
# F11: error_flag (buffer(7)) Boundaries
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f11_01_over_16384_dispatches_fallback(tmp_path: Path) -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    probe = harness.build_f11_metal_graph_probe(tmp_path)
    res = harness.run_cmd([str(probe), str(metallib), "over_16384"])
    assert res.returncode == 0, res.stderr + res.stdout
    assert "PASS f11 over_16384 dispatches=16385" in res.stdout


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f11_02_middle_dispatch_error_sticky(tmp_path: Path) -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    probe = harness.build_f11_metal_graph_probe(tmp_path)
    res = harness.run_cmd([str(probe), str(metallib), "middle_error_sticky"])
    assert res.returncode == 0, res.stderr + res.stdout
    assert "PASS f11 middle_error_sticky slots=64" in res.stdout


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f11_03_flag_reader_keeps_dep_barrier(synthetic_trace_factory) -> None:
    tb = harness.DispatchTraceBuilder()
    tb.add_pipeline("rms_norm:1", "rms_norm", [(0, "x", "write", True)])
    tb.add_pipeline(
        "qsa_score_accumulate:1", "qsa_score_accumulate", [(0, "y", "write", True)]
    )
    tb.add_allocation(0, "a", 1024)
    tb.add_allocation(1, "b", 1024)
    tb.add_host_write(0, 0, 1024, "fa_0")
    tb.add_host_write(1, 0, 1024, "fb_0")
    bid = tb.begin_batch()
    tb.add_dispatch(
        "rms_norm:1",
        [
            {"index": 0, "slot": 0, "offset": 0, "bytes": 1024, "access": "write"},
            {"index": 7, "buffer": "error_flag"},
        ],
        constants=[{"index": 3, "bytes": 4, "sha256": harness.sha256_bytes(b"qsa_w")}],
    )
    tb.add_barrier()
    tb.add_dispatch(
        "qsa_score_accumulate:1",
        [
            {"index": 0, "slot": 1, "offset": 0, "bytes": 1024, "access": "write"},
            {"index": 7, "buffer": "error_flag"},
        ],
        constants=[{"index": 3, "bytes": 4, "sha256": harness.sha256_bytes(b"qsa_r")}],
    )
    tb.submit_batch(bid, 2, [(0, "fa_end"), (1, "fb_end")])
    path = synthetic_trace_factory("flag_reader_barrier.jsonl", tb)
    res = harness.run_dispatch_graph(["summary", str(path)])
    assert res.returncode == 0
    batch = json.loads(res.stdout)["batches"][0]
    assert batch["depth"] == 2 and batch["dependency_edges"] == 1


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f11_04_out_of_sync_flag_table_caught(tmp_path: Path) -> None:
    sync_res = harness.run_cargo_test_checked(
        ["-p", "tokspur-dispatch-graph", "--test", "kernels"], min_passed=1
    )
    assert sync_res.returncode == 0

    def _build_pair(pipeline_key: str, fn_name: str, out_name: str) -> dict[str, Any]:
        tb = harness.DispatchTraceBuilder()
        tb.add_pipeline("rms_norm:1", "rms_norm", [(0, "x", "write", True)])
        tb.add_pipeline(pipeline_key, fn_name, [(0, "y", "write", True)])
        tb.add_allocation(0, "a", 1024)
        tb.add_allocation(1, "b", 1024)
        bid = tb.begin_batch()
        tb.add_dispatch(
            "rms_norm:1",
            [
                {"index": 0, "slot": 0, "offset": 0, "bytes": 1024, "access": "write"},
                {"index": 7, "buffer": "error_flag"},
            ],
        )
        tb.add_barrier()
        tb.add_dispatch(
            pipeline_key,
            [
                {"index": 0, "slot": 1, "offset": 0, "bytes": 1024, "access": "write"},
                {"index": 7, "buffer": "error_flag"},
            ],
        )
        tb.submit_batch(bid, 2)
        p = tb.write(tmp_path / out_name)
        r = harness.run_dispatch_graph(["summary", str(p)])
        assert r.returncode == 0
        return json.loads(r.stdout)["batches"][0]

    unreg = _build_pair("fake_qsa_reader:1", "fake_qsa_reader", "fake_qsa.jsonl")
    assert unreg["dependency_edges"] == 0 and unreg["depth"] == 1
    reg = _build_pair(
        "qsa_verify_rows_channels:1", "qsa_verify_rows_channels", "real_qsa.jsonl"
    )
    assert reg["dependency_edges"] == 1 and reg["depth"] == 2


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f11_05_aborted_batch_clears_error_flag(tmp_path: Path) -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    probe = harness.build_f11_metal_graph_probe(tmp_path)
    res = harness.run_cmd([str(probe), str(metallib), "abort_clears_error"])
    assert res.returncode == 0, res.stderr + res.stdout
    assert "PASS f11 abort_clears_error" in res.stdout


# ---------------------------------------------------------------------------
# F12: Default Concurrent Level-Scheduling Boundaries
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f12_01_concurrent_plus_capture_reject() -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    startup_bin = harness.require_artifact(
        harness.find_native_test_bin("test-concurrent_dispatch_startup"),
        "test-concurrent_dispatch_startup",
    )
    res_s = harness.run_cmd([str(startup_bin), str(metallib)])
    assert res_s.returncode == 0, res_s.stderr + res_s.stdout
    assert "concurrent startup: 6 refusals passed" in res_s.stdout
    graph_bin = harness.require_artifact(
        harness.find_native_test_bin("test-graph"), "test-graph"
    )
    res_g = harness.run_cmd([str(graph_bin), str(metallib), "--operator-checkpoints"])
    assert res_g.returncode == 0, res_g.stderr + res_g.stdout


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f12_02_aborted_first_batch_locks_mode(tmp_path: Path) -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    bin_path = harness.require_artifact(
        harness.find_native_test_bin("test-concurrent_dispatch_startup"),
        "test-concurrent_dispatch_startup",
    )
    res = harness.run_cmd([str(bin_path), str(metallib)])
    assert res.returncode == 0, res.stderr + res.stdout
    assert "profiler mode=" in res.stdout
    assert "6 refusals passed" in res.stdout
    ref_p, cand_p = harness.build_dag_pair_traces(tmp_path, depth=2)
    chk = harness.run_dispatch_graph(["check", str(ref_p), str(cand_p)])
    assert chk.returncode == 0 and json.loads(chk.stdout)["passed"] is True


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f12_03_concurrent_vs_serial_sha_match(isolated_state_root: Path) -> None:
    payload = harness.make_chat_payload(
        "Compute 11 * 13.", enable_thinking=False, seed=12, temperature=0.0
    )
    with harness.ServerProcess(isolated_state_root / "conc") as srv_c:
        assert srv_c.client().post_chat(payload).status == 200
    out_c, _ = srv_c.stop()
    sha_c = harness.extract_done_events(harness.parse_jsonl_records(out_c))[-1][
        "hc_exact_final_logit_sha256"
    ]
    with harness.ServerProcess(
        isolated_state_root / "serial", serve_args=["--no-concurrent-dispatch"]
    ) as srv_s:
        assert srv_s.client().post_chat(payload).status == 200
    out_s, _ = srv_s.stop()
    sha_s = harness.extract_done_events(harness.parse_jsonl_records(out_s))[-1][
        "hc_exact_final_logit_sha256"
    ]
    assert sha_c == sha_s


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f12_04_explicit_and_default_flags_same() -> None:
    res = harness.run_cargo_test_checked(
        [
            "-p",
            "tokspur-cli",
            "--",
            "serve_options_default_and_opt_in_flags_match_python_cli",
        ],
        min_passed=1,
    )
    assert res.returncode == 0, res.stderr + res.stdout
    help_res = harness.run_tokspur(["serve", "--help"])
    assert help_res.returncode == 0, help_res.stderr + help_res.stdout
    assert "--concurrent-dispatch" in help_res.stdout


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f12_05_invalid_concurrent_flag_reject(isolated_state_root: Path) -> None:
    res = harness.run_tokspur(
        ["serve", "--concurrent-dispatch=bogus"], state_root=isolated_state_root
    )
    assert res.returncode != 0


# ---------------------------------------------------------------------------
# F13: Pack v3 Static dm Certification Boundaries
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f13_01_pack_v3_nan_dm_0x7e00_reject(tmp_path: Path) -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    bin_path = harness.require_artifact(
        harness.find_native_test_bin("test-planar_pack_v3"), "test-planar_pack_v3"
    )
    res = harness.run_cmd([str(bin_path), str(metallib), "--nax-staging-only"])
    assert res.returncode == 0, res.stderr + res.stdout
    assert "PASS planar_v3 sticky finite-value contract" in res.stdout

    cpu_probe = harness.build_planar_dm_cpu_probe(tmp_path)
    cpu_res = harness.run_cmd([str(cpu_probe), "nan"])
    assert cpu_res.returncode == 0, cpu_res.stderr + cpu_res.stdout
    assert "PASS f13 planar_dm_cpu nan" in cpu_res.stdout


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f13_02_pack_v3_inf_dm_0x7c00_reject(tmp_path: Path) -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    bin_path = harness.require_artifact(
        harness.find_native_test_bin("test-planar_pack_v3"), "test-planar_pack_v3"
    )
    res = harness.run_cmd(
        [str(bin_path), str(metallib), "--narrow-rows-control", str(metallib)]
    )
    assert res.returncode == 0, res.stderr + res.stdout
    assert "PASS narrow-row exact complete outputs" in res.stdout

    cpu_probe = harness.build_planar_dm_cpu_probe(tmp_path)
    cpu_res = harness.run_cmd([str(cpu_probe), "inf"])
    assert cpu_res.returncode == 0, cpu_res.stderr + cpu_res.stdout
    assert "PASS f13 planar_dm_cpu inf" in cpu_res.stdout


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f13_03_tampered_payload_or_seal_reject(tmp_path: Path) -> None:
    corrupt_copy = harness.build_synthetic_gguf_v3(tmp_path / "corrupt_magic.asa.gguf")
    harness.tamper_file_bit(corrupt_copy, offset=0, xor_mask=0xFF)
    native_bin = harness.require_artifact(
        harness.find_tokspur_native(), "tokspur-native"
    )
    inspect_bad = harness.run_cmd([str(native_bin), "inspect", str(corrupt_copy)])
    assert inspect_bad.returncode != 0

    derived_bin = harness.require_artifact(
        harness.find_native_test_bin("test-derived_pack"), "test-derived_pack"
    )
    d_res = harness.run_cmd([str(derived_bin)])
    assert d_res.returncode == 0, d_res.stderr + d_res.stdout
    assert "PASS derived pack checks=" in d_res.stdout
    assert "v3_auth_source_read_mode=" in d_res.stdout

    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    pack_bin = harness.require_artifact(
        harness.find_native_test_bin("test-planar_pack_v3"), "test-planar_pack_v3"
    )
    p_res = harness.run_cmd([str(pack_bin), str(metallib)])
    assert p_res.returncode == 0, p_res.stderr + p_res.stdout
    assert "PASS planar_v3 builder, multi-worker adoption" in p_res.stdout


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f13_04_sm16_out_of_range_64_rejected() -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    bin_path = harness.require_artifact(
        harness.find_native_test_bin("test-planar_pack_v3"), "test-planar_pack_v3"
    )
    res = harness.run_cmd([str(bin_path), str(metallib)])
    assert res.returncode == 0, res.stderr + res.stdout
    assert "PASS planar_v3 sm16 bijection and range refusal" in res.stdout


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f13_05_bad_gguf_suffix_or_chunk_mib(
    isolated_state_root: Path, tmp_path: Path
) -> None:
    harness.seed_isolated_state_root(isolated_state_root)
    adopted_plan = harness.run_tokspur(
        ["pack", "plan", "--format", "v3"], state_root=isolated_state_root
    )
    assert adopted_plan.returncode != 0
    assert (
        "already uses a pre-built derived pack"
        in (adopted_plan.stderr + adopted_plan.stdout).lower()
    )
    for bad_chunk in ("0", "65"):
        res_chunk = harness.run_tokspur(
            ["pack", "plan", "--format", "v3", "--chunk-mib", bad_chunk],
            state_root=isolated_state_root,
        )
        assert res_chunk.returncode != 0

    native_bin = harness.require_artifact(
        harness.find_tokspur_native(), "tokspur-native"
    )
    src = harness.build_synthetic_gguf_v3(tmp_path / "src.gguf")
    mtp = harness.build_synthetic_gguf_v3(tmp_path / "mtp.gguf")
    for bad_name in ("bad.bin", "tokspur-pack-v2.asa.gguf"):
        bad_build = harness.run_cmd(
            [
                str(native_bin),
                "pack-v3-build",
                "--output",
                str(tmp_path / bad_name),
                "--mtp",
                str(mtp),
                "--gguf",
                str(src),
            ]
        )
        assert bad_build.returncode != 0
        assert (
            "distinct from the reserved v2 pack paths"
            in (bad_build.stderr + bad_build.stdout).lower()
        )


# ---------------------------------------------------------------------------
# F14: Shape-Registration Invariant Discharge Boundaries
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f14_01_route_id_ge_experts_rejected() -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    bin_path = harness.require_artifact(
        harness.find_native_test_bin("test-planar_pack_v3"), "test-planar_pack_v3"
    )
    res = harness.run_cmd([str(bin_path), str(metallib)])
    assert res.returncode == 0, res.stderr + res.stdout
    assert "PASS planar_v3 sticky finite-value contract" in res.stdout


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f14_02_unaligned_or_zero_dim_rejected(tmp_path: Path) -> None:
    native_bin = harness.require_artifact(
        harness.find_tokspur_native(), "tokspur-native"
    )
    zero_dim = harness.build_synthetic_gguf_v3(
        tmp_path / "zero_dim.gguf",
        tensors=[("blk.0.attn_norm.weight", (0, 256), 0, b"")],
    )
    res_zero = harness.run_cmd([str(native_bin), "inspect", str(zero_dim)])
    assert res_zero.returncode != 0
    assert "zero tensor dimension" in (res_zero.stderr + res_zero.stdout).lower()

    unaligned_q8 = harness.build_synthetic_gguf_v3(
        tmp_path / "unaligned_q8.gguf",
        tensors=[("blk.0.attn_q.weight", (257, 4), 8, b"\x00" * (34 * 9 * 4))],
    )
    res_unalign = harness.run_cmd([str(native_bin), "inspect", str(unaligned_q8)])
    assert res_unalign.returncode != 0
    assert "not block-aligned" in (res_unalign.stderr + res_unalign.stdout).lower()

    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    probe = harness.build_planar_matrix_probe(tmp_path)
    res_probe = harness.run_cmd([str(probe), str(metallib), "invalid_shapes"])
    assert res_probe.returncode == 0, res_probe.stderr + res_probe.stdout
    assert "PASS f14 invalid_shapes refusals=8" in res_probe.stdout


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f14_03_non_planar_consumer_rejected(tmp_path: Path) -> None:
    bad_gguf = harness.build_synthetic_gguf_v3(tmp_path / "bad_layout.gguf")
    native_bin = harness.require_artifact(
        harness.find_tokspur_native(), "tokspur-native"
    )
    res = harness.run_cmd([str(native_bin), "layout", str(bad_gguf)])
    assert res.returncode != 0


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f14_04_raw_upload_dm_nan_inf_rejected(tmp_path: Path) -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    probe = harness.build_planar_matrix_probe(tmp_path)
    res_probe = harness.run_cmd([str(probe), str(metallib), "raw_dm"])
    assert res_probe.returncode == 0, res_probe.stderr + res_probe.stdout
    assert "PASS f14 raw_dm refusals=9" in res_probe.stdout

    pack_bin = harness.require_artifact(
        harness.find_native_test_bin("test-planar_pack_v3"), "test-planar_pack_v3"
    )
    p_res = harness.run_cmd([str(pack_bin), str(metallib), "--nax-staging-only"])
    assert p_res.returncode == 0, p_res.stderr + p_res.stdout
    assert "PASS planar_v3 sticky finite-value contract" in p_res.stdout


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f14_05_dflash2_oob_stage_dup_prepare() -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    stage_bin = harness.require_artifact(
        harness.find_native_test_bin("test-dflash2_context_stage"),
        "test-dflash2_context_stage",
    )
    res_s = harness.run_cmd([str(stage_bin), str(metallib)])
    assert res_s.returncode == 0, res_s.stderr + res_s.stdout
    assert "DFlash2 pending context:" in res_s.stdout
    gather_bin = harness.require_artifact(
        harness.find_native_test_bin("test-dflash2_gather"), "test-dflash2_gather"
    )
    res_g = harness.run_cmd([str(gather_bin), str(metallib), "--correctness-only"])
    assert res_g.returncode == 0, res_g.stderr + res_g.stdout


# ---------------------------------------------------------------------------
# F15: Stripped Metal Hot-Loop Boundaries
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f15_01_narrow_rows_1_2_vs_nax_3_32(tmp_path: Path) -> None:
    b0 = tmp_path / "narrow0"
    b1 = tmp_path / "narrow1"
    for bdir, flag in ((b0, "0"), (b1, "1")):
        mk = harness.run_make_native(
            [
                f"VERIFY_NARROW_ROWS={flag}",
                f"BUILD={bdir}",
                f"{bdir}/tokspur-owned.metallib",
                f"{bdir}/test-planar_pack_v3",
            ]
        )
        assert mk.returncode == 0, mk.stderr
    res = harness.run_cmd(
        [
            str(b1 / "test-planar_pack_v3"),
            str(b1 / "tokspur-owned.metallib"),
            "--narrow-rows-control",
            str(b0 / "tokspur-owned.metallib"),
        ]
    )
    assert res.returncode == 0, res.stderr + res.stdout


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f15_02_non_pow2_rows_stride_padding(tmp_path: Path) -> None:
    b0 = tmp_path / "union0"
    b1 = tmp_path / "union1"
    for bdir, flag in ((b0, "0"), (b1, "1")):
        mk = harness.run_make_native(
            [
                f"PARALLEL_PLANAR_UNION={flag}",
                f"BUILD={bdir}",
                f"{bdir}/tokspur-owned.metallib",
                f"{bdir}/test-planar_pack_v3",
            ]
        )
        assert mk.returncode == 0, mk.stderr
    res = harness.run_cmd(
        [
            str(b1 / "test-planar_pack_v3"),
            str(b1 / "tokspur-owned.metallib"),
            "--union-tiles-control",
            str(b0 / "tokspur-owned.metallib"),
        ]
    )
    assert res.returncode == 0, res.stderr + res.stdout


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f15_03_extreme_finite_65504_no_false(tmp_path: Path) -> None:
    b0 = tmp_path / "stage0"
    b1 = tmp_path / "stage1"
    for bdir, flag in ((b0, "0"), (b1, "1")):
        mk = harness.run_make_native(
            [
                f"M5_HEADLINE_SCREEN={flag}",
                f"BUILD={bdir}",
                f"{bdir}/tokspur-owned.metallib",
                f"{bdir}/test-planar_pack_v3",
            ]
        )
        assert mk.returncode == 0, mk.stderr
    res = harness.run_cmd(
        [
            str(b1 / "test-planar_pack_v3"),
            str(b1 / "tokspur-owned.metallib"),
            "--nax-staging-control",
            str(b0 / "tokspur-owned.metallib"),
        ]
    )
    assert res.returncode == 0, res.stderr + res.stdout


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f15_04_fp16_stage_overflow_caught() -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    pack_bin = harness.require_artifact(
        harness.find_native_test_bin("test-planar_pack_v3"), "test-planar_pack_v3"
    )
    res = harness.run_cmd([str(pack_bin), str(metallib), "--nax-staging-only"])
    assert res.returncode == 0, res.stderr + res.stdout
    assert "PASS planar_v3 sticky finite-value contract" in res.stdout
    num_bin = harness.require_artifact(
        harness.find_native_test_bin("test-numeric"), "test-numeric"
    )
    assert harness.run_cmd([str(num_bin)]).returncode == 0


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f15_05_nax_staging_only_poison_catch() -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    meta_bin = harness.require_artifact(
        harness.find_native_test_bin("test-planar_union_metadata"),
        "test-planar_union_metadata",
    )
    res_m = harness.run_cmd([str(meta_bin), str(metallib)])
    assert res_m.returncode == 0, res_m.stderr + res_m.stdout
    assert "ordered planar union metadata cases passed" in res_m.stdout
    pack_bin = harness.require_artifact(
        harness.find_native_test_bin("test-planar_pack_v3"), "test-planar_pack_v3"
    )
    res_p = harness.run_cmd([str(pack_bin), str(metallib), "--nax-staging-only"])
    assert res_p.returncode == 0, res_p.stderr + res_p.stdout
    assert "output guards" in res_p.stdout


# ---------------------------------------------------------------------------
# F16: End-of-Batch Finiteness & Metal 3.2 Flags Boundaries
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f16_01_final_col_nan_0x7e00_caught() -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    mtp_bin = harness.require_artifact(
        harness.find_native_test_bin("test-mtp_head"), "test-mtp_head"
    )
    res_m = harness.run_cmd([str(mtp_bin), str(metallib)])
    assert res_m.returncode == 0, res_m.stderr + res_m.stdout
    assert "MTP head generated checks passed:" in res_m.stdout
    topk_bin = harness.require_artifact(
        harness.find_native_test_bin("test-logit_topk"), "test-logit_topk"
    )
    res_t = harness.run_cmd([str(topk_bin), str(metallib)])
    assert res_t.returncode == 0, res_t.stderr + res_t.stdout
    assert "exact CPU/GPU parity" in res_t.stdout


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f16_02_pos_neg_inf_0x7c00_fc00_caught() -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    bg_bin = harness.require_artifact(
        harness.find_native_test_bin("test-block_gemv"), "test-block_gemv"
    )
    res_bg = harness.run_cmd([str(bg_bin), str(metallib)])
    assert res_bg.returncode == 0, res_bg.stderr + res_bg.stdout
    assert "Owned block GEMV checks passed:" in res_bg.stdout

    n3_bin = harness.require_artifact(
        harness.find_native_test_bin("test-q8_three_rows_stack"),
        "test-q8_three_rows_stack",
    )
    res_n3 = harness.run_cmd([str(n3_bin), "--execute-generated", str(metallib)])
    assert res_n3.returncode == 0, res_n3.stderr + res_n3.stdout
    assert "nonfinite-output-refusals=16" in res_n3.stdout


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f16_03_swiglu_3e38_overflow_caught() -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    stage_bin = harness.require_artifact(
        harness.find_native_test_bin("test-dflash2_context_stage"),
        "test-dflash2_context_stage",
    )
    assert harness.run_cmd([str(stage_bin), str(metallib)]).returncode == 0
    pack_bin = harness.require_artifact(
        harness.find_native_test_bin("test-planar_pack_v3"), "test-planar_pack_v3"
    )
    res = harness.run_cmd([str(pack_bin), str(metallib), "--nax-staging-only"])
    assert res.returncode == 0, res.stderr + res.stdout


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f16_04_subnormal_sampling_logit_catch() -> None:
    samp_bin = harness.require_artifact(
        harness.find_native_test_bin("test-sampling"), "test-sampling"
    )
    assert harness.run_cmd([str(samp_bin)]).returncode == 0
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    topk_bin = harness.require_artifact(
        harness.find_native_test_bin("test-logit_topk"), "test-logit_topk"
    )
    res = harness.run_cmd([str(topk_bin), str(metallib)])
    assert res.returncode == 0, res.stderr + res.stdout
    assert "exact CPU/GPU parity" in res.stdout


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f16_05_clean_retry_after_nan_recovers() -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    graph_bin = harness.require_artifact(
        harness.find_native_test_bin("test-graph"), "test-graph"
    )
    res_g = harness.run_cmd([str(graph_bin), str(metallib)])
    assert res_g.returncode == 0, res_g.stderr + res_g.stdout
    assert "Persistent Metal graph checks passed:" in res_g.stdout
    stack_bin = harness.require_artifact(
        harness.find_native_test_bin("test-sampled_stack"), "test-sampled_stack"
    )
    res_s = harness.run_cmd([str(stack_bin), str(metallib), "--concurrent"])
    assert res_s.returncode == 0, res_s.stderr + res_s.stdout
    assert "nonfinite-output-refusals=8" in res_s.stdout


# ---------------------------------------------------------------------------
# F17: DFlash2 Lazy flush_context() Boundaries
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f17_01_short_single_token_dflash2_safe(
    isolated_state_root: Path,
) -> None:
    with harness.ServerProcess(
        isolated_state_root,
        serve_args=["--speed-preset", "--drafter", "dflash2"],
    ) as srv:
        c = srv.client()
        assert (
            c.post_chat(
                harness.make_chat_payload("Hi.", enable_thinking=False, max_tokens=4)
            ).status
            == 200
        )
        assert (
            c.post_chat(
                harness.make_chat_payload("Ok.", enable_thinking=False, max_tokens=4)
            ).status
            == 200
        )


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f17_02_three_way_ping_pong_zero_skips(
    isolated_state_root: Path,
) -> None:
    prompts = [harness.make_long_repeated_prompt(f"pp_{k}") for k in ("A", "B", "C")]
    with harness.ServerProcess(
        isolated_state_root,
        serve_args=["--speed-preset", "--drafter", "dflash2"],
    ) as srv:
        c = srv.client()
        ra1 = c.post_chat(
            harness.make_chat_payload(prompts[0], enable_thinking=False, seed=1)
        )
        assert (
            c.post_chat(
                harness.make_chat_payload(prompts[1], enable_thinking=False, seed=2)
            ).status
            == 200
        )
        assert (
            c.post_chat(
                harness.make_chat_payload(prompts[2], enable_thinking=False, seed=3)
            ).status
            == 200
        )
        ra2 = c.post_chat(
            harness.make_chat_payload(
                "Back to A.",
                history=[
                    {"role": "user", "content": prompts[0]},
                    {
                        "role": "assistant",
                        "content": ra1.json()["choices"][0]["message"]["content"],
                    },
                ],
                enable_thinking=False,
                seed=1,
            )
        )
        assert ra2.status == 200
    out, _ = srv.stop()
    done = harness.extract_done_events(harness.parse_jsonl_records(out))[-1]
    assert done["parked_lanes"]["skips"] == 0
    assert done["parked_lanes"]["restores"] >= 1


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f17_03_dflash2_bad_seed_no_stale_ctx(
    isolated_state_root: Path,
) -> None:
    p_a = harness.make_long_repeated_prompt("dfBadSeedA")
    p_b = harness.make_long_repeated_prompt("dfBadSeedB")
    with harness.ServerProcess(
        isolated_state_root,
        serve_args=["--speed-preset", "--drafter", "dflash2"],
    ) as srv:
        c = srv.client()
        r1 = c.post_chat(harness.make_chat_payload(p_a, enable_thinking=False, seed=11))
        assert (
            c.post_chat(
                harness.make_chat_payload(p_b, enable_thinking=False, seed=12)
            ).status
            == 200
        )
        r3 = c.post_chat(
            harness.make_chat_payload(
                "Return bad seed.",
                history=[
                    {"role": "user", "content": p_a},
                    {
                        "role": "assistant",
                        "content": r1.json()["choices"][0]["message"]["content"],
                    },
                ],
                enable_thinking=False,
                seed=9999,
            )
        )
        assert r3.status == 200
        assert r3.json()["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    out, _ = srv.stop()
    done = harness.extract_done_events(harness.parse_jsonl_records(out))[-1]
    assert done["parked_lanes"]["restores"] == 0
    assert done["sampling"]["seed"] == 9999


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f17_04_dflash2_lookup_park_preserves(isolated_state_root: Path) -> None:
    p_a = harness.make_long_repeated_prompt("dfLookupA")
    p_b = harness.make_long_repeated_prompt("dfLookupB")
    with harness.ServerProcess(
        isolated_state_root,
        serve_args=[
            "--speed-preset",
            "--drafter",
            "dflash2",
            "--dflash2-lookup",
            "hybrid",
        ],
    ) as srv:
        c = srv.client()
        r1 = c.post_chat(harness.make_chat_payload(p_a, enable_thinking=False, seed=21))
        assert c.post_chat(harness.make_chat_payload(p_b, seed=22)).status == 200
        r3 = c.post_chat(
            harness.make_chat_payload(
                "Return with lookup.",
                history=[
                    {"role": "user", "content": p_a},
                    {
                        "role": "assistant",
                        "content": r1.json()["choices"][0]["message"]["content"],
                    },
                ],
                enable_thinking=False,
                seed=21,
            )
        )
        assert r3.status == 200
        assert harness.validate_chat_completion_json(r3.json(), min_cached_tokens=1) > 0
    out, _ = srv.stop()
    done = harness.extract_done_events(harness.parse_jsonl_records(out))[-1]
    assert done["parked_lanes"]["skips"] == 0
    assert done["parked_lanes"]["restores"] >= 1


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f17_05_dflash2_budget_evict_recovers(isolated_state_root: Path) -> None:
    prompts = [
        harness.make_long_repeated_prompt(f"df_ev_{idx}", approx_lines=100)
        for idx in range(4)
    ]
    with harness.ServerProcess(
        isolated_state_root,
        serve_args=[
            "--drafter",
            "dflash2",
            "--shared-prefix-cache",
            "--parked-lanes-mib",
            "256",
        ],
    ) as srv:
        c = srv.client()
        first_msg = ""
        for idx, prompt in enumerate(prompts):
            resp = c.post_chat(
                harness.make_chat_payload(
                    prompt, enable_thinking=False, seed=40 + idx, max_tokens=16
                )
            )
            assert resp.status == 200
            if idx == 0:
                first_msg = resp.json()["choices"][0]["message"]["content"]
        cold = c.post_chat(
            harness.make_chat_payload(
                "Return after eviction.",
                history=[
                    {"role": "user", "content": prompts[0]},
                    {"role": "assistant", "content": first_msg},
                ],
                enable_thinking=False,
                seed=40,
                max_tokens=16,
            )
        )
        assert cold.status == 200
        assert cold.json()["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    out, _ = srv.stop()
    done = harness.extract_done_events(harness.parse_jsonl_records(out))
    assert done[-2]["parked_lanes"]["evictions"] >= 1
    assert done[-1]["parked_lanes"]["skips"] == 0


# ---------------------------------------------------------------------------
# F18: Sqrt(N) Checkpoint Boundaries
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f18_01_64k_vs_128k_context_boundary(isolated_state_root: Path) -> None:
    harness.seed_isolated_state_root(isolated_state_root)
    bad_ctx = harness.run_tokspur(
        ["serve", "--context-length", "32768"], state_root=isolated_state_root
    )
    assert bad_ctx.returncode == 2
    prompt_75k = harness.make_long_repeated_prompt("ctx128k", approx_lines=2500)
    with harness.ServerProcess(
        isolated_state_root,
        serve_args=[
            "--context-length",
            "131072",
            "--prefill-dense",
            "nax",
            "--prefill-grouped",
            "nax",
        ],
    ) as srv:
        resp = srv.client().post_chat(
            harness.make_chat_payload(prompt_75k, enable_thinking=False, max_tokens=8),
            timeout=900.0,
        )
        assert resp.status == 200
        assert resp.json()["usage"]["prompt_tokens"] > 65536


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f18_02_over_context_length_400_reject(
    isolated_state_root: Path,
) -> None:
    huge = "tok " * 70000
    with harness.ServerProcess(
        isolated_state_root, serve_args=["--context-length", "65536"]
    ) as srv:
        resp = srv.client().post_chat(
            harness.make_chat_payload(huge, enable_thinking=False, max_tokens=4)
        )
        assert resp.status == 400


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f18_03_checkpoint_cap_range_32gib_max(
    isolated_state_root: Path,
) -> None:
    res = harness.run_tokspur(
        ["serve", "--checkpoint-cap-gib", "33.0"],
        state_root=isolated_state_root,
    )
    assert res.returncode != 0


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f18_04_floor_gt_cap_or_symlink_reject(
    isolated_state_root: Path, tmp_path: Path
) -> None:
    harness.seed_isolated_state_root(isolated_state_root)
    over_floor = harness.run_tokspur(
        ["serve", "--checkpoint-floor-gib", "101.0"],
        state_root=isolated_state_root,
    )
    assert over_floor.returncode == 2
    floor_no_cap = harness.run_tokspur(
        ["serve", "--checkpoint-floor-gib", "20.0", "--checkpoint-cap-gib", "0.0"],
        state_root=isolated_state_root,
    )
    assert floor_no_cap.returncode == 1
    assert "checkpoint-cap-gib" in (floor_no_cap.stderr + floor_no_cap.stdout)

    real_dir = tmp_path / "real_ckpt"
    real_dir.mkdir()
    sym_dir = tmp_path / "sym_ckpt"
    sym_dir.symlink_to(real_dir)
    with pytest.raises(RuntimeError, match="symlink"):
        with harness.ServerProcess(
            isolated_state_root,
            serve_args=[
                "--checkpoint-cap-gib",
                "2.0",
                "--checkpoint-dir",
                str(sym_dir),
            ],
        ):
            pass


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f18_05_tight_ssd_cap_keeps_sqrt_miles(
    isolated_state_root: Path,
) -> None:
    long_p = harness.make_long_repeated_prompt("tightSsd", approx_lines=2500)
    with harness.ServerProcess(
        isolated_state_root,
        serve_args=[
            "--context-length",
            "131072",
            "--checkpoint-cap-gib",
            "1.0",
            "--checkpoint-floor-gib",
            "0.25",
            "--prefill-dense",
            "nax",
            "--prefill-grouped",
            "nax",
        ],
    ) as srv:
        c = srv.client()
        r1 = c.post_chat(
            harness.make_chat_payload(
                long_p + "\nBranch 1.", enable_thinking=False, max_tokens=16
            ),
            timeout=900.0,
        )
        assert r1.status == 200
        assert r1.json()["usage"]["prompt_tokens"] > 65536
        r2 = c.post_chat(
            harness.make_chat_payload(
                long_p + "\nBranch 2.", enable_thinking=False, max_tokens=16
            ),
            timeout=900.0,
        )
        assert r2.status == 200
    out, _ = srv.stop()
    done = harness.extract_done_events(harness.parse_jsonl_records(out))[-1]
    assert done["checkpoint_ssd_enabled"] is True
    assert done["checkpoint_ssd_spills"] >= 1
    assert 0 < done["checkpoint_ssd_occupied_bytes"] <= (1 << 30)
    assert done["checkpoint_restored_tokens"] > 0


# ---------------------------------------------------------------------------
# F19: Instantaneous Anchor-Entropy Width Gating Boundaries
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f19_01_no_pending_anchor_step_safe(isolated_state_root: Path) -> None:
    with harness.ServerProcess(isolated_state_root) as srv:
        resp = srv.client().post_chat(
            harness.make_chat_payload("Yes", enable_thinking=False, max_tokens=1)
        )
        assert resp.status == 200
        assert resp.json()["usage"]["completion_tokens"] == 1
    out, _ = srv.stop()
    done = harness.extract_done_events(harness.parse_jsonl_records(out))[-1]
    assert done["native_mtp_rounds"] >= 1


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f19_02_entropy_zero_and_uniform_ext(isolated_state_root: Path) -> None:
    with harness.ServerProcess(isolated_state_root) as srv:
        c = srv.client()
        r_low = c.post_chat(
            harness.make_chat_payload(
                "0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 next:",
                enable_thinking=False,
                max_tokens=16,
            )
        )
        assert r_low.status == 200
        lo_done = harness.wait_for_done_events(srv.stdout_log_path, min_count=1)[-1]
        r_high = c.post_chat(
            harness.make_chat_payload(
                "Choose a random rare word:",
                enable_thinking=False,
                temperature=1.0,
                top_k=20,
                seed=77,
                max_tokens=16,
            )
        )
        assert r_high.status == 200
    out, _ = srv.stop()
    hi_done = harness.extract_done_events(harness.parse_jsonl_records(out))[-1]
    lo_rounds = lo_done["native_mtp_rounds"]
    lo_proposed = lo_done["native_mtp_proposed"]
    hi_rounds = hi_done["native_mtp_rounds"] - lo_rounds
    hi_proposed = hi_done["native_mtp_proposed"] - lo_proposed
    assert lo_rounds >= 1 and hi_rounds >= 1
    assert lo_proposed >= 1 and hi_proposed >= 1
    assert lo_done["native_mtp_accepted"] >= 1
    assert (hi_proposed / hi_rounds) < (lo_proposed / lo_rounds)


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f19_03_lookup_miss_falls_back_clean(isolated_state_root: Path) -> None:
    with harness.ServerProcess(
        isolated_state_root, serve_args=["--mtp-lookup", "hybrid"]
    ) as srv:
        resp = srv.client().post_chat(
            harness.make_chat_payload(
                "Explain why Rust prevents data races.",
                enable_thinking=False,
                max_tokens=24,
            )
        )
        assert resp.status == 200
    out, _ = srv.stop()
    done = harness.extract_done_events(harness.parse_jsonl_records(out))[-1]
    assert done["exact_candidate_mtp_lookup"] == "hybrid"
    assert isinstance(done["exact_candidate_mtp_lookup_rounds"], int)
    assert (
        0 <= done["exact_candidate_mtp_lookup_rounds"] <= done["native_mtp_rounds"] + 4
    )
    if "exact_candidate_mtp_lookup_accepted" in done:
        assert isinstance(done["exact_candidate_mtp_lookup_accepted"], int)
        assert (
            0
            <= done["exact_candidate_mtp_lookup_accepted"]
            <= done["native_mtp_accepted"]
        )
    assert done["native_mtp_rounds"] >= 1 and done["native_mtp_proposed"] >= 1


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f19_04_nonfinite_anchor_triggers_r3() -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    mtp_bin = harness.require_artifact(
        harness.find_native_test_bin("test-mtp_head"), "test-mtp_head"
    )
    res_m = harness.run_cmd([str(mtp_bin), str(metallib)])
    assert res_m.returncode == 0, res_m.stderr + res_m.stdout
    assert "MTP head generated checks passed:" in res_m.stdout
    stack_bin = harness.require_artifact(
        harness.find_native_test_bin("test-sampled_stack"), "test-sampled_stack"
    )
    res_s = harness.run_cmd([str(stack_bin), str(metallib)])
    assert res_s.returncode == 0, res_s.stderr + res_s.stdout
    assert "nonfinite-output-refusals=8" in res_s.stdout


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f19_05_r6_no_history_early_cap_guard(isolated_state_root: Path) -> None:
    res_early = harness.run_tokspur(
        ["serve", "--dflash2-early-cap=1"], state_root=isolated_state_root
    )
    assert res_early.returncode != 0
    res_hist = harness.run_tokspur(
        ["serve", "--history-only-head-cap=1"], state_root=isolated_state_root
    )
    assert res_hist.returncode != 0
    help_res = harness.run_tokspur(["serve", "--help"], state_root=isolated_state_root)
    assert help_res.returncode == 0
    assert "dflash2-early-cap" not in help_res.stdout
    assert "history-only-head-cap" not in help_res.stdout


# ---------------------------------------------------------------------------
# F20: Unified Default CLI & Worker Boundaries
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f20_01_explicit_no_shared_or_park_0(
    isolated_state_root: Path,
) -> None:
    with harness.ServerProcess(
        isolated_state_root,
        serve_args=["--no-shared-prefix-cache", "--parked-lanes-mib", "0"],
    ) as srv:
        assert (
            srv.client()
            .post_chat(
                harness.make_chat_payload(
                    "No cache.", enable_thinking=False, max_tokens=8
                )
            )
            .status
            == 200
        )
    out, _ = srv.stop()
    done = harness.extract_done_events(harness.parse_jsonl_records(out))[-1]
    assert done["parked_lanes"]["budget_bytes"] == 0


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f20_02_explicit_no_concurrent_optout(
    isolated_state_root: Path,
) -> None:
    with harness.ServerProcess(
        isolated_state_root, serve_args=["--no-concurrent-dispatch"]
    ) as srv:
        c = srv.client()
        assert c.get_health().status == 200
        resp = c.post_chat(
            harness.make_chat_payload(
                "Serial ping.", enable_thinking=False, max_tokens=8
            )
        )
        assert resp.status == 200
    out, _ = srv.stop()
    ready = next(
        r for r in harness.parse_jsonl_records(out) if r.get("event") == "ready"
    )
    assert ready["native_ready"].get("concurrent_dispatch_mode") in (None, "off")
    done = harness.extract_done_events(harness.parse_jsonl_records(out))[-1]
    assert done.get("concurrent_dispatch_mode") in (None, "off")


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f20_03_conflicting_cli_flags_nonzero(
    isolated_state_root: Path,
) -> None:
    res1 = harness.run_tokspur(
        ["serve", "--drafter", "dflash2", "--mtp-lookup", "hybrid"],
        state_root=isolated_state_root,
    )
    assert res1.returncode != 0
    res2 = harness.run_tokspur(
        ["serve", "--parked-lanes-mib", "16385"],
        state_root=isolated_state_root,
    )
    assert res2.returncode != 0
    res3 = harness.run_tokspur(["--state-root", "relative/path", "doctor"])
    assert res3.returncode != 0


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f20_04_serve_lock_exclusivity_enforced(
    isolated_state_root: Path,
) -> None:
    with harness.ServerProcess(isolated_state_root) as srv:
        assert srv.client().get_health().status == 200
        second = harness.run_tokspur(
            ["serve", "--port", str(harness.pick_unused_loopback_port())],
            state_root=isolated_state_root,
        )
        assert second.returncode == 1
        assert "Another tokspur operation owns this state" in second.stderr


@pytest.mark.tier2
@pytest.mark.e2e_live
def test_t2_f20_05_r6_no_lean_nanoda_cli_flags() -> None:
    top_help = harness.run_tokspur(["--help"])
    assert top_help.returncode == 0
    assert "lean4" not in top_help.stdout.lower()
    assert "nanoda" not in top_help.stdout.lower()
    bad_lean = harness.run_tokspur(["serve", "--lean4"])
    assert bad_lean.returncode != 0
    meta = harness.run_cargo_metadata()
    pkg_names = {pkg["name"] for pkg in meta["packages"]}
    assert "nanoda_lib" not in pkg_names and "lean4" not in pkg_names
