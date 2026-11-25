# SPDX-License-Identifier: Apache-2.0
"""Tier 1 E2E tests: 100 isolated happy-path feature tests (5 per F1-F20)."""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

from . import harness

# ---------------------------------------------------------------------------
# F1: Baseline A/B & Parity Artifacts
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f1_01_health_and_models_baseline(isolated_state_root: Path) -> None:
    with harness.ServerProcess(isolated_state_root) as srv:
        client = srv.client()
        health = client.get_health()
        assert health.status == 200
        assert health.json()["status"] == "ready"
        models = client.get_models()
        assert models.status == 200
        data = models.json()["data"]
        assert len(data) == 1 and data[0]["object"] == "model"


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f1_02_verify_and_draft_profile_capture(
    isolated_state_root: Path,
) -> None:
    # graph.mm:3368 requires serial execution when Metal counter profiling is enabled.
    with harness.ServerProcess(
        isolated_state_root,
        serve_args=["--no-concurrent-dispatch"],
        extra_env={
            "TOKSPUR_DUMP_VERIFY_PROFILE": "1",
            "TOKSPUR_ALLOW_KNOWN_PROFILE": "1",
        },
    ) as srv:
        client = srv.client()
        resp = client.post_chat(
            harness.make_chat_payload("Write a Rust add function.", max_tokens=24)
        )
        assert resp.status == 200
        harness.validate_chat_completion_json(resp.json())
    _, stderr = srv.stop()
    verify_lines = harness.parse_profile_lines(stderr, "VERIFY_PROFILE")
    assert len(verify_lines) >= 1
    assert verify_lines[0]["dispatches"] > 0 and verify_lines[0]["gpu_ms"] > 0.0


def _wait_for_quiet_swapins(max_wait_s: float = 20.0) -> None:
    if sys.platform != "darwin":
        return
    import ctypes
    import ctypes.util

    libc = ctypes.CDLL(ctypes.util.find_library("System"))
    ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
    objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
    ptr = ctypes.c_void_p
    objc.objc_getClass.restype = objc.sel_registerName.restype = ptr
    send = ctypes.CFUNCTYPE(ptr, ptr, ptr)(("objc_msgSend", objc))
    thermal_fn = ctypes.CFUNCTYPE(ctypes.c_long, ptr, ptr)(("objc_msgSend", objc))
    info = send(
        objc.objc_getClass(b"NSProcessInfo"), objc.sel_registerName(b"processInfo")
    )
    thermal_sel = objc.sel_registerName(b"thermalState")

    def _is_nominal() -> bool:
        pressure = ctypes.c_int()
        size = ctypes.c_size_t(ctypes.sizeof(pressure))
        p_ok = (
            libc.sysctlbyname(
                b"kern.memorystatus_vm_pressure_level",
                ctypes.byref(pressure),
                ctypes.byref(size),
                None,
                0,
            )
            == 0
            and pressure.value == 1
        )
        t_ok = bool(info) and thermal_fn(info, thermal_sel) == 0
        return p_ok and t_ok

    def _read_swapins() -> int | None:
        vm = subprocess.check_output(["/usr/bin/vm_stat"], text=True, timeout=5)
        match = re.search(r"^Swapins:\s+(\d+)\.", vm, re.M)
        return int(match.group(1)) if match else None

    prev = _read_swapins()
    if prev is None:
        return
    quiet_ticks = 0
    deadline = time.monotonic() + max_wait_s
    while time.monotonic() < deadline:
        time.sleep(1.0)
        cur = _read_swapins()
        if cur is None:
            return
        if cur == prev and _is_nominal():
            quiet_ticks += 1
            if quiet_ticks >= 2:
                return
        else:
            quiet_ticks = 0
            prev = cur


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f1_03_benchmark_parked_chats_oracle(
    isolated_state_root: Path, tmp_path: Path
) -> None:
    for attempt in range(2):
        run_root = isolated_state_root / f"attempt_{attempt}"
        out_path = tmp_path / f"oracle_{attempt}.jsonl"
        with harness.ServerProcess(
            run_root,
            serve_args=["--speed-preset", "--drafter", "mtp", "--mtp-depth", "3"],
        ) as srv:
            key_file = srv.state_dir / "api-key"
            assert srv.client().get_health().status == 200
            assert key_file.exists()
            _wait_for_quiet_swapins()
            res = harness.run_script(
                "benchmark_parked_chats.py",
                [
                    "oracle",
                    "--endpoint",
                    f"http://127.0.0.1:{srv.port}/v1",
                    "--api-key-file",
                    str(key_file),
                    "--log",
                    str(srv.stdout_log_path),
                    "--out",
                    str(out_path),
                    "--gap",
                    "0",
                ],
            )
            if res.returncode != 0:
                srv_out, srv_err = srv.stop()
                raise AssertionError(
                    f"{res.stderr}{res.stdout}\n"
                    f"server stderr:\n{srv_err}\nserver stdout:\n{srv_out}"
                )
            assert out_path.exists()
            lines = [
                json.loads(line)
                for line in out_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            header, *rows, result = lines
            assert header["schema"] == 1 and header["mode"] == "oracle"
            assert header["gap_s"] == 0
            assert [row["turn"] for row in rows] == ["J1", "J2", "J3", "R1", "R2", "R3"]
            for row in rows:
                assert row["before"] is not None and row["after"] is not None
                assert row["swap_delta"] is not None
                assert row["restores"] == 0 and row["restored_tokens"] == 0
                assert row["http_s"] > 0.0 and row["prefill_s"] > 0.0
                cached = row["usage"]["prompt_tokens_details"]["cached_tokens"]
                if row["turn"] in ("J1", "R1"):
                    assert cached == 0
                else:
                    assert cached > 0
            assert result["resource_eligible"]["full"] is not None
            assert result["resource_eligible"]["returns"] is not None
            assert (
                result["http_seconds"]["full"]["all"]
                > result["http_seconds"]["returns"]["all"]
                > 0.0
            )
            if (
                result["resource_eligible"]["full"] is True
                and result["resource_eligible"]["returns"] is True
            ):
                return
            cold_turns_only_bounded_swapins = (
                result["resource_eligible"]["returns"] is True
                and all(
                    r["swap_delta"]["swapins"] == 0
                    for r in rows
                    if r["turn"] not in ("J1", "R1")
                )
                and 0
                < sum(
                    r["swap_delta"]["swapins"]
                    for r in rows
                    if r["turn"] in ("J1", "R1")
                )
                <= 256
            )
            if attempt == 1 and cold_turns_only_bounded_swapins:
                return
            cold_swapins_only = (
                attempt == 0
                and all(
                    r["before"]["thermal"] == 0
                    and r["after"]["thermal"] == 0
                    and r["before"]["pressure"] == 1
                    and r["after"]["pressure"] == 1
                    and r["swap_delta"]["swapouts"] == 0
                    and r["swap_delta"]["swap_used"] <= 0
                    and r["before"]["swapouts"] == rows[0]["before"]["swapouts"]
                    and r["after"]["swapouts"] == rows[0]["before"]["swapouts"]
                    and r["after"]["swap_used"] <= r["before"]["swap_used"]
                    and r["after"]["swap_used"] <= rows[0]["before"]["swap_used"]
                    for r in rows
                )
                and any(r["swap_delta"]["swapins"] > 0 for r in rows)
            )
            assert cold_swapins_only, (
                f"oracle attempt {attempt} ineligible "
                f"(resource_eligible={result['resource_eligible']}): {rows}"
            )
    raise AssertionError("oracle retry failed resource_eligible['returns']")


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f1_04_stream_collector_sse_capture(
    isolated_state_root: Path, tmp_path: Path
) -> None:
    body_path = tmp_path / "req.json"
    out_path = tmp_path / "stream.jsonl"
    body_path.write_text(
        json.dumps(
            harness.make_chat_payload(
                "Explain Rust ownership briefly.",
                stream=True,
                include_usage=True,
                max_tokens=20,
            )
        ),
        encoding="utf-8",
    )
    with harness.ServerProcess(isolated_state_root) as srv:
        res = harness.run_script(
            "stream_collector.py",
            [
                "--endpoint",
                f"http://127.0.0.1:{srv.port}/v1",
                "--body-file",
                str(body_path),
                "--api-key-file",
                str(srv.state_dir / "api-key"),
                "--repeat",
                "1",
                "--packet-tokens",
                "8",
                "--out",
                str(out_path),
            ],
        )
        assert res.returncode == 0, res.stderr
        rec = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
        assert rec["outcome"] == "complete"
        assert rec["first_payload_ms"] > 0.0
        assert len(rec["digest"]) == 64


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f1_05_greedy_final_logit_sha_parity(
    isolated_state_root: Path,
) -> None:
    payload = harness.make_chat_payload(
        "Compute 17 + 25 in one short sentence.", seed=9983, temperature=0.0
    )
    with harness.ServerProcess(isolated_state_root) as srv:
        client = srv.client()
        r1 = client.post_chat(payload)
        r2 = client.post_chat(payload)
        assert r1.status == 200 and r2.status == 200
        assert r1.json()["choices"][0]["message"] == r2.json()["choices"][0]["message"]
    out, _ = srv.stop()
    done = harness.extract_done_events(harness.parse_jsonl_records(out))
    assert len(done) >= 2
    assert (
        done[0]["hc_exact_final_logit_sha256"] == done[1]["hc_exact_final_logit_sha256"]
    )


# ---------------------------------------------------------------------------
# F2: Holdout Manifest (holdout.rs)
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.e2e_offline
def test_t1_f2_01_valid_manifest_single_use(
    tmp_path: Path, synthetic_r5_factory
) -> None:
    suite_res = harness.run_r5_holdout_suite()
    assert suite_res.returncode == 0
    bundle = synthetic_r5_factory("f2_valid")
    res = harness.run_r5_holdout_eval(
        tmp_path,
        bundle["manifest_path"].parent,
        "holdout_0",
        bundle["ledger_path"],
    )
    assert res.returncode == 0, res.stderr + res.stdout
    outcome = json.loads(res.stdout)
    assert outcome["holdout_id"] == "holdout_0"
    assert len(str(outcome["consumption_token"])) >= 16


@pytest.mark.tier1
@pytest.mark.e2e_offline
def test_t1_f2_02_hash_chained_ledger_genesis(
    tmp_path: Path, synthetic_r5_factory
) -> None:
    bundle = synthetic_r5_factory("f2_genesis", num_holdouts=2)
    r0 = harness.run_r5_holdout_eval(
        tmp_path / "run0",
        bundle["manifest_path"].parent,
        "holdout_0",
        bundle["ledger_path"],
    )
    r1 = harness.run_r5_holdout_eval(
        tmp_path / "run1",
        bundle["manifest_path"].parent,
        "holdout_1",
        bundle["ledger_path"],
    )
    assert r0.returncode == 0 and r1.returncode == 0
    entries = json.loads(bundle["ledger_path"].read_text(encoding="utf-8"))["entries"]
    assert len(entries) == 2
    assert entries[0]["prev_chain_hash"] == "GENESIS"
    assert entries[1]["prev_chain_hash"] == entries[0]["record_chain_hash"]


@pytest.mark.tier1
@pytest.mark.e2e_offline
def test_t1_f2_03_zero_calibration_leakage(
    tmp_path: Path, synthetic_r5_factory
) -> None:
    bundle = synthetic_r5_factory("f2_noleak", leak_calibration=False)
    cal_shas = {s["expected_sha256"] for s in bundle["manifest"]["calibration_splits"]}
    hold_shas = {s["expected_sha256"] for s in bundle["manifest"]["holdout_splits"]}
    assert cal_shas.isdisjoint(hold_shas)
    res = harness.run_r5_holdout_eval(
        tmp_path,
        bundle["manifest_path"].parent,
        "holdout_0",
        bundle["ledger_path"],
    )
    assert res.returncode == 0, res.stderr + res.stdout
    outcome = json.loads(res.stdout)
    assert outcome["holdout_id"] == "holdout_0"
    assert outcome["zero_calibration_leakage"] is True


@pytest.mark.tier1
@pytest.mark.e2e_offline
def test_t1_f2_04_accurate_eval_records_pass(
    tmp_path: Path, synthetic_r5_factory
) -> None:
    bundle = synthetic_r5_factory(
        "f2_pass", max_relative_l2_error=0.03, reference_error=0.025
    )
    res = harness.run_r5_holdout_eval(
        tmp_path,
        bundle["manifest_path"].parent,
        "holdout_0",
        bundle["ledger_path"],
    )
    assert res.returncode == 0, res.stderr + res.stdout
    ledger = json.loads(bundle["ledger_path"].read_text(encoding="utf-8"))
    assert ledger["entries"][0]["passed"] is True
    assert ledger["entries"][0]["relative_l2_error"] <= 0.03


@pytest.mark.tier1
@pytest.mark.e2e_offline
def test_t1_f2_05_isolated_ledger_0600_write(
    tmp_path: Path, synthetic_r5_factory
) -> None:
    bundle = synthetic_r5_factory("f2_isolated")
    manifest_before = bundle["manifest_path"].read_bytes()
    res = harness.run_r5_holdout_eval(
        tmp_path,
        bundle["manifest_path"].parent,
        "holdout_0",
        bundle["ledger_path"],
    )
    assert res.returncode == 0, res.stderr + res.stdout
    assert bundle["ledger_path"].exists()
    assert (bundle["ledger_path"].stat().st_mode & 0o777) == 0o600
    assert bundle["manifest_path"].read_bytes() == manifest_before


# ---------------------------------------------------------------------------
# F3: Sequential Futility Stopping (transfer.rs, ci_tests.rs, futility.rs)
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f3_01_loser_2block_futility_stop(tmp_path: Path) -> None:
    suite_res = harness.run_r5_futility_suite()
    assert suite_res.returncode == 0
    blocks = harness.build_paired_block_series("loser_2block")
    res = harness.run_r5_futility_eval(
        tmp_path, {"mode": "futility", "alpha": 0.05, "blocks": blocks}
    )
    assert res.returncode == 0, res.stderr + res.stdout
    out = json.loads(res.stdout)
    assert out["stop_for_futility"] is True and out["blocks_evaluated"] == 2


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f3_02_winner_2block_continues(tmp_path: Path) -> None:
    blocks = harness.build_paired_block_series("winner_2block")
    res = harness.run_r5_futility_eval(
        tmp_path, {"mode": "futility", "alpha": 0.05, "blocks": blocks}
    )
    assert res.returncode == 0, res.stderr + res.stdout
    out = json.loads(res.stdout)
    assert out["stop_for_futility"] is False and out["p_value"] < 0.01


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f3_03_wilcoxon_signed_rank_ties(tmp_path: Path) -> None:
    diffs = [1.0, 1.0, -1.0, 2.0, 3.0, 4.0]
    res = harness.run_r5_futility_eval(tmp_path, {"mode": "wilcoxon", "diffs": diffs})
    assert res.returncode == 0, res.stderr + res.stdout
    out = json.loads(res.stdout)
    assert out["n_eff"] == 6 and math.isclose(out["w_pos"] + out["w_neg"], 21.0)
    assert 0.0 < out["p_value"] <= 1.0


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f3_04_bh_fdr_step_up_monotone(tmp_path: Path) -> None:
    p_values = [0.002, 0.03, 0.025, 0.80]
    res = harness.run_r5_futility_eval(
        tmp_path, {"mode": "bh_fdr", "alpha": 0.05, "p_values": p_values}
    )
    assert res.returncode == 0, res.stderr + res.stdout
    out = json.loads(res.stdout)
    assert out["rejected"][0] is True and out["rejected"][3] is False
    assert out["q_values"][2] <= out["q_values"][1] <= out["q_values"][3]


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f3_05_zero_diff_pairs_excluded(tmp_path: Path) -> None:
    diffs = [0.0, -0.0, 0.0, 2.0, 3.0, 4.0, 5.0]
    res = harness.run_r5_futility_eval(tmp_path, {"mode": "wilcoxon", "diffs": diffs})
    assert res.returncode == 0, res.stderr + res.stdout
    out = json.loads(res.stdout)
    assert out["n_eff"] == 4 and math.isclose(out["w_pos"], 10.0)
    assert math.isclose(out["w_neg"], 0.0) and math.isclose(out["p_value"], 0.125)


# ---------------------------------------------------------------------------
# F4: native/rust.mk Receipt Cache
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f4_01_cold_rust_check_populates_cache(tmp_path: Path) -> None:
    shim, log_path = harness.create_cargo_logging_shim(tmp_path)
    cache_dir = tmp_path / "cold_cache"
    res = harness.run_make_native(
        ["rust-check", f"RUST_RECEIPT_CACHE_DIR={cache_dir}", f"CARGO={shim}"],
        makefile="rust.mk",
        env={"TOKSPUR_E2E_CARGO_LOG": str(log_path)},
    )
    assert res.returncode == 0, res.stdout + "\n" + res.stderr
    assert log_path.exists() and len(harness.read_cached_cargo_calls(log_path)) >= 1
    assert len(list(cache_dir.rglob("*.json"))) >= 1


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f4_02_warm_rust_check_skips_rebuild(tmp_path: Path) -> None:
    shim, cold_log = harness.create_cargo_logging_shim(tmp_path)
    cache_dir = tmp_path / "warm_cache"
    first = harness.run_make_native(
        ["rust-check", f"RUST_RECEIPT_CACHE_DIR={cache_dir}", f"CARGO={shim}"],
        makefile="rust.mk",
        env={"TOKSPUR_E2E_CARGO_LOG": str(cold_log)},
    )
    assert first.returncode == 0, first.stdout + "\n" + first.stderr
    mtimes_before = {p: p.stat().st_mtime_ns for p in sorted(cache_dir.rglob("*.json"))}
    assert len(mtimes_before) >= 1
    warm_log = tmp_path / "warm_cargo.log"
    second = harness.run_make_native(
        ["rust-check", f"RUST_RECEIPT_CACHE_DIR={cache_dir}", f"CARGO={shim}"],
        makefile="rust.mk",
        env={"TOKSPUR_E2E_CARGO_LOG": str(warm_log)},
    )
    assert second.returncode == 0, second.stdout + "\n" + second.stderr
    assert harness.read_cached_cargo_calls(warm_log) == []
    mtimes_after = {p: p.stat().st_mtime_ns for p in sorted(cache_dir.rglob("*.json"))}
    assert mtimes_after == mtimes_before


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f4_03_receipt_records_input_digests(tmp_path: Path) -> None:
    cache_dir = tmp_path / "digest_cache"
    res = harness.run_make_native(
        ["rust-check", f"RUST_RECEIPT_CACHE_DIR={cache_dir}"], makefile="rust.mk"
    )
    assert res.returncode == 0, res.stdout + "\n" + res.stderr
    receipts = sorted(cache_dir.rglob("*.json"))
    assert len(receipts) >= 1
    sample = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert isinstance(sample, dict)
    digests = [
        val
        for val in json.dumps(sample).split('"')
        if re.fullmatch(r"[0-9a-f]{64}", val)
    ]
    assert len(digests) >= 1


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f4_04_native_build_inputs_clean() -> None:
    res = harness.run_make_native(["rust-check"], makefile="rust.mk")
    assert res.returncode == 0, res.stdout + "\n" + res.stderr
    assert list(harness.NATIVE_DIR.rglob("*receipt*")) == []
    py_res = harness.run_cmd(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_native_build_inputs.py",
            "-q",
        ]
    )
    assert py_res.returncode == 0, py_res.stdout + py_res.stderr


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f4_05_rust_release_builds_cleanly(tmp_path: Path) -> None:
    out_dir = tmp_path / "rust-release"
    res = harness.run_make_native(
        ["rust-release", f"RUST_SERVER_DIR={out_dir}"], makefile="rust.mk"
    )
    assert res.returncode == 0, res.stdout + "\n" + res.stderr
    cli = out_dir / "release/tokspur"
    srv = out_dir / "release/tokspur-server"
    assert cli.exists() and srv.exists()
    cli_res = harness.run_cmd([str(cli), "--help"])
    assert cli_res.returncode == 0, cli_res.stderr
    srv_res = harness.run_cmd([str(srv), "--help"])
    assert srv_res.returncode == 1, srv_res.stderr + srv_res.stdout
    assert "duplicate server option or missing value" in srv_res.stderr


# ---------------------------------------------------------------------------
# F5: Parked-Lane (committed, sampling, seed) Recording
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f5_01_thinking_on_parks_and_restores(
    isolated_state_root: Path,
) -> None:
    p_a = harness.make_long_repeated_prompt("convA")
    p_b = harness.make_long_repeated_prompt("convB")
    with harness.ServerProcess(
        isolated_state_root, serve_args=["--speed-preset"]
    ) as srv:
        c = srv.client()
        r1 = c.post_chat(harness.make_chat_payload(p_a, enable_thinking=True, seed=11))
        assert r1.status == 200
        a_msg = r1.json()["choices"][0]["message"]
        r2 = c.post_chat(harness.make_chat_payload(p_b, enable_thinking=True, seed=22))
        assert r2.status == 200
        r3 = c.post_chat(
            harness.make_chat_payload(
                "Follow-up question on convA.",
                history=[
                    {"role": "user", "content": p_a},
                    {"role": "assistant", "content": a_msg["content"]},
                ],
                enable_thinking=True,
                seed=11,
            )
        )
        assert r3.status == 200
        cached = harness.validate_chat_completion_json(r3.json(), min_cached_tokens=1)
        assert cached > 0
    out, _ = srv.stop()
    done = harness.extract_done_events(harness.parse_jsonl_records(out))
    assert done[-1]["parked_lanes"]["parks"] >= 1
    assert done[-1]["parked_lanes"]["restores"] >= 1


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f5_02_thinking_off_parks_and_restores(
    isolated_state_root: Path,
) -> None:
    p_a = harness.make_long_repeated_prompt("noThinkA")
    p_b = harness.make_long_repeated_prompt("noThinkB")
    with harness.ServerProcess(
        isolated_state_root, serve_args=["--speed-preset"]
    ) as srv:
        c = srv.client()
        r1 = c.post_chat(harness.make_chat_payload(p_a, enable_thinking=False, seed=31))
        assert r1.status == 200
        a_content = r1.json()["choices"][0]["message"]["content"]
        assert (
            c.post_chat(
                harness.make_chat_payload(p_b, enable_thinking=False, seed=32)
            ).status
            == 200
        )
        r3 = c.post_chat(
            harness.make_chat_payload(
                "Next turn without thinking.",
                history=[
                    {"role": "user", "content": p_a},
                    {"role": "assistant", "content": a_content},
                ],
                enable_thinking=False,
                seed=31,
            )
        )
        assert r3.status == 200
        assert harness.validate_chat_completion_json(r3.json(), min_cached_tokens=1) > 0
    out, _ = srv.stop()
    done = harness.extract_done_events(harness.parse_jsonl_records(out))
    assert done[-1]["parked_lanes"]["restores"] >= 1


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f5_03_explicit_seed_match_restores(
    isolated_state_root: Path,
) -> None:
    p_a = harness.make_long_repeated_prompt("seedMatchA")
    p_b = harness.make_long_repeated_prompt("seedMatchB")
    with harness.ServerProcess(
        isolated_state_root, serve_args=["--speed-preset"]
    ) as srv:
        c = srv.client()
        r1 = c.post_chat(
            harness.make_chat_payload(
                p_a, enable_thinking=False, seed=777, temperature=0.6, top_k=20
            )
        )
        assert r1.status == 200
        a_txt = r1.json()["choices"][0]["message"]["content"]
        assert (
            c.post_chat(
                harness.make_chat_payload(p_b, enable_thinking=False, seed=888)
            ).status
            == 200
        )
        r3 = c.post_chat(
            harness.make_chat_payload(
                "Continue with same seed.",
                history=[
                    {"role": "user", "content": p_a},
                    {"role": "assistant", "content": a_txt},
                ],
                enable_thinking=False,
                seed=777,
                temperature=0.6,
                top_k=20,
            )
        )
        assert r3.status == 200
        usage_cached = harness.validate_chat_completion_json(
            r3.json(), min_cached_tokens=1
        )
        assert usage_cached > 0
    out, _ = srv.stop()
    done = harness.extract_done_events(harness.parse_jsonl_records(out))
    assert done[-1]["parked_lanes"]["restores"] >= 1
    assert done[-1]["parked_lanes"]["restored_tokens"] > 0
    assert done[-1]["sampling"]["seed"] == 777
    comp_events = [
        t for t in srv.prefix_trace_records() if t.get("event") == "completion"
    ]
    assert comp_events[-1]["cached_tokens"] == usage_cached


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f5_04_omitted_seed_compatible_match(
    isolated_state_root: Path,
) -> None:
    p_a = harness.make_long_repeated_prompt("omitSeedA")
    p_b = harness.make_long_repeated_prompt("omitSeedB")
    with harness.ServerProcess(
        isolated_state_root, serve_args=["--speed-preset"]
    ) as srv:
        c = srv.client()
        r1 = c.post_chat(
            harness.make_chat_payload(p_a, enable_thinking=False, seed=555)
        )
        assert r1.status == 200
        a_txt = r1.json()["choices"][0]["message"]["content"]
        assert (
            c.post_chat(
                harness.make_chat_payload(p_b, enable_thinking=False, seed=556)
            ).status
            == 200
        )
        r3 = c.post_chat(
            harness.make_chat_payload(
                "Return with omitted seed.",
                history=[
                    {"role": "user", "content": p_a},
                    {"role": "assistant", "content": a_txt},
                ],
                enable_thinking=False,
                seed=None,
            )
        )
        assert r3.status == 200
        assert harness.validate_chat_completion_json(r3.json(), min_cached_tokens=1) > 0


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f5_05_multi_conversation_independent(
    isolated_state_root: Path,
) -> None:
    p_a = harness.make_long_repeated_prompt("slotA")
    p_b = harness.make_long_repeated_prompt("slotB")
    with harness.ServerProcess(
        isolated_state_root, serve_args=["--speed-preset"]
    ) as srv:
        c = srv.client()
        ra1 = c.post_chat(harness.make_chat_payload(p_a, enable_thinking=False, seed=1))
        rb1 = c.post_chat(harness.make_chat_payload(p_b, enable_thinking=False, seed=2))
        assert ra1.status == 200 and rb1.status == 200
        ra2 = c.post_chat(
            harness.make_chat_payload(
                "Turn 2 A",
                history=[
                    {"role": "user", "content": p_a},
                    {
                        "role": "assistant",
                        "content": ra1.json()["choices"][0]["message"]["content"],
                    },
                ],
                enable_thinking=False,
                seed=1,
            )
        )
        rb2 = c.post_chat(
            harness.make_chat_payload(
                "Turn 2 B",
                history=[
                    {"role": "user", "content": p_b},
                    {
                        "role": "assistant",
                        "content": rb1.json()["choices"][0]["message"]["content"],
                    },
                ],
                enable_thinking=False,
                seed=2,
            )
        )
        assert ra2.status == 200 and rb2.status == 200
        assert (
            harness.validate_chat_completion_json(ra2.json(), min_cached_tokens=1) > 0
        )
        assert (
            harness.validate_chat_completion_json(rb2.json(), min_cached_tokens=1) > 0
        )


# ---------------------------------------------------------------------------
# F6: Hoisted Render/Encode & Echoed Reasoning
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f6_01_echoed_reasoning_content_hit(
    isolated_state_root: Path,
) -> None:
    p_a = harness.make_long_repeated_prompt("echoA")
    p_b = harness.make_long_repeated_prompt("echoB")
    with harness.ServerProcess(
        isolated_state_root, serve_args=["--speed-preset"]
    ) as srv:
        c = srv.client()
        r1 = c.post_chat(
            harness.make_chat_payload(
                p_a,
                enable_thinking=True,
                seed=24319,
                max_tokens=512,
                extra={"thinking_budget": 32},
            )
        )
        assert r1.status == 200
        assert r1.json()["choices"][0]["finish_reason"] == "stop"
        msg1 = r1.json()["choices"][0]["message"]
        assert (
            c.post_chat(harness.make_chat_payload(p_b, enable_thinking=False)).status
            == 200
        )
        echoed_assistant: dict[str, object] = {
            "role": "assistant",
            "content": msg1["content"],
        }
        if "reasoning_content" in msg1:
            echoed_assistant["reasoning_content"] = msg1["reasoning_content"]
        r3 = c.post_chat(
            harness.make_chat_payload(
                "Continue after echoed reasoning.",
                history=[{"role": "user", "content": p_a}, echoed_assistant],
                enable_thinking=True,
                seed=24319,
                extra={"thinking_budget": 32},
            )
        )
        assert r3.status == 200
        assert harness.validate_chat_completion_json(r3.json(), min_cached_tokens=1) > 0


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f6_02_shared_turns_hoisted_encode(
    isolated_state_root: Path,
) -> None:
    p_shared = harness.make_long_repeated_prompt("sharedPrefix")
    with harness.ServerProcess(
        isolated_state_root, serve_args=["--speed-preset"]
    ) as srv:
        c = srv.client()
        r1 = c.post_chat(
            harness.make_chat_payload(
                p_shared + "\nBranch 1.", enable_thinking=False, seed=101
            )
        )
        r2 = c.post_chat(
            harness.make_chat_payload(
                p_shared + "\nBranch 2.", enable_thinking=False, seed=101
            )
        )
        r3 = c.post_chat(
            harness.make_chat_payload(
                p_shared + "\nBranch 3.", enable_thinking=False, seed=101
            )
        )
        assert r1.status == 200 and r2.status == 200 and r3.status == 200
        cached = harness.validate_chat_completion_json(r3.json(), min_cached_tokens=1)
        assert cached > 0


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f6_03_empty_turns_base_prompt_hit(
    isolated_state_root: Path,
) -> None:
    p_a = harness.make_long_repeated_prompt("emptyTurnsA")
    p_b = harness.make_long_repeated_prompt("emptyTurnsB")
    with harness.ServerProcess(
        isolated_state_root, serve_args=["--speed-preset"]
    ) as srv:
        c = srv.client()
        r1 = c.post_chat(harness.make_chat_payload(p_a, enable_thinking=False, seed=7))
        assert r1.status == 200
        assert (
            c.post_chat(
                harness.make_chat_payload(p_b, enable_thinking=False, seed=8)
            ).status
            == 200
        )
        r3 = c.post_chat(
            harness.make_chat_payload(
                "Next empty-turns turn.",
                history=[
                    {"role": "user", "content": p_a},
                    {
                        "role": "assistant",
                        "content": r1.json()["choices"][0]["message"]["content"],
                    },
                ],
                enable_thinking=False,
                seed=7,
            )
        )
        assert r3.status == 200
        assert harness.validate_chat_completion_json(r3.json(), min_cached_tokens=1) > 0


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f6_04_interleaved_matches_oracle(
    isolated_state_root: Path,
) -> None:
    p_a = harness.make_long_repeated_prompt("oracleA")
    p_b = harness.make_long_repeated_prompt("oracleB")
    with harness.ServerProcess(
        isolated_state_root / "oracle", serve_args=["--speed-preset"]
    ) as srv_o:
        co = srv_o.client()
        oa1 = co.post_chat(
            harness.make_chat_payload(p_a, enable_thinking=False, seed=9)
        )
        oa2 = co.post_chat(
            harness.make_chat_payload(
                "Follow up A",
                history=[
                    {"role": "user", "content": p_a},
                    {
                        "role": "assistant",
                        "content": oa1.json()["choices"][0]["message"]["content"],
                    },
                ],
                enable_thinking=False,
                seed=9,
            )
        )
        oracle_msg = oa2.json()["choices"][0]["message"]
    with harness.ServerProcess(
        isolated_state_root / "interleaved", serve_args=["--speed-preset"]
    ) as srv_i:
        ci = srv_i.client()
        ia1 = ci.post_chat(
            harness.make_chat_payload(p_a, enable_thinking=False, seed=9)
        )
        assert (
            ci.post_chat(
                harness.make_chat_payload(p_b, enable_thinking=False, seed=10)
            ).status
            == 200
        )
        ia2 = ci.post_chat(
            harness.make_chat_payload(
                "Follow up A",
                history=[
                    {"role": "user", "content": p_a},
                    {
                        "role": "assistant",
                        "content": ia1.json()["choices"][0]["message"]["content"],
                    },
                ],
                enable_thinking=False,
                seed=9,
            )
        )
        assert ia2.json()["choices"][0]["message"] == oracle_msg


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f6_05_prefix_trace_matches_usage(
    isolated_state_root: Path,
) -> None:
    p_a = harness.make_long_repeated_prompt("traceUsageA")
    with harness.ServerProcess(
        isolated_state_root, serve_args=["--speed-preset"]
    ) as srv:
        c = srv.client()
        r1 = c.post_chat(harness.make_chat_payload(p_a, enable_thinking=False, seed=12))
        r2 = c.post_chat(
            harness.make_chat_payload(
                "Next question.",
                history=[
                    {"role": "user", "content": p_a},
                    {
                        "role": "assistant",
                        "content": r1.json()["choices"][0]["message"]["content"],
                    },
                ],
                enable_thinking=False,
                seed=12,
            )
        )
        usage_cached = harness.validate_chat_completion_json(
            r2.json(), min_cached_tokens=1
        )
        traces = [
            t for t in srv.prefix_trace_records() if t.get("event") == "prompt_prefix"
        ]
        assert len(traces) >= 2
        assert traces[-1]["cached_tokens"] == usage_cached


# ---------------------------------------------------------------------------
# F7: Parked-Lane respelled_on_committed
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f7_01_noncanonical_bpe_parked_hit(
    isolated_state_root: Path,
) -> None:
    p_a = harness.make_long_repeated_prompt("bpeA")
    p_b = harness.make_long_repeated_prompt("bpeB")
    with harness.ServerProcess(
        isolated_state_root, serve_args=["--speed-preset"]
    ) as srv:
        c = srv.client()
        r1 = c.post_chat(
            harness.make_chat_payload(
                p_a,
                enable_thinking=False,
                seed=24319,
                temperature=0.7,
                top_k=20,
                top_p=0.95,
                max_tokens=96,
            )
        )
        assert r1.status == 200
        a_txt = r1.json()["choices"][0]["message"]["content"]
        assert (
            c.post_chat(
                harness.make_chat_payload(p_b, enable_thinking=False, seed=99)
            ).status
            == 200
        )
        r3 = c.post_chat(
            harness.make_chat_payload(
                "Elaborate on step 0010.",
                history=[
                    {"role": "user", "content": p_a},
                    {"role": "assistant", "content": a_txt},
                ],
                enable_thinking=False,
                seed=24319,
                temperature=0.7,
                top_k=20,
                top_p=0.95,
                max_tokens=48,
            )
        )
        assert r3.status == 200
        assert harness.validate_chat_completion_json(r3.json(), min_cached_tokens=1) > 0
    out, _ = srv.stop()
    done = harness.extract_done_events(harness.parse_jsonl_records(out))
    assert done[-1]["parked_lanes"]["restores"] >= 1


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f7_02_live_committed_respell_append(
    isolated_state_root: Path,
) -> None:
    p_a = harness.make_long_repeated_prompt("liveRespell")
    with harness.ServerProcess(
        isolated_state_root, serve_args=["--speed-preset"]
    ) as srv:
        c = srv.client()
        r1 = c.post_chat(
            harness.make_chat_payload(
                p_a,
                enable_thinking=False,
                seed=404,
                temperature=0.8,
                top_k=20,
                max_tokens=64,
            )
        )
        r2 = c.post_chat(
            harness.make_chat_payload(
                "Continue live.",
                history=[
                    {"role": "user", "content": p_a},
                    {
                        "role": "assistant",
                        "content": r1.json()["choices"][0]["message"]["content"],
                    },
                ],
                enable_thinking=False,
                seed=404,
                temperature=0.8,
                top_k=20,
                max_tokens=32,
            )
        )
        assert r2.status == 200
        traces = [
            t for t in srv.prefix_trace_records() if t.get("event") == "prompt_prefix"
        ]
        assert traces[-1]["append"] is True


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f7_03_whitespace_punct_bpe_respell(
    isolated_state_root: Path,
) -> None:
    p_code = harness.make_long_repeated_prompt("rustCode") + "\n```rust\nfn f() {}\n```"
    p_other = harness.make_long_repeated_prompt("other")
    with harness.ServerProcess(
        isolated_state_root, serve_args=["--speed-preset"]
    ) as srv:
        c = srv.client()
        r1 = c.post_chat(
            harness.make_chat_payload(
                p_code + "\nReturn indented Rust code only.",
                enable_thinking=False,
                seed=808,
                temperature=0.65,
                top_k=20,
                max_tokens=80,
            )
        )
        assert c.post_chat(harness.make_chat_payload(p_other, seed=809)).status == 200
        r3 = c.post_chat(
            harness.make_chat_payload(
                "Add unit test for it.",
                history=[
                    {
                        "role": "user",
                        "content": p_code + "\nReturn indented Rust code only.",
                    },
                    {
                        "role": "assistant",
                        "content": r1.json()["choices"][0]["message"]["content"],
                    },
                ],
                enable_thinking=False,
                seed=808,
                temperature=0.65,
                top_k=20,
            )
        )
        assert r3.status == 200
        assert harness.validate_chat_completion_json(r3.json(), min_cached_tokens=1) > 0


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f7_04_thinking_off_sampled_respell(
    isolated_state_root: Path,
) -> None:
    p_a = harness.make_long_repeated_prompt("sampledNoThink")
    p_b = harness.make_long_repeated_prompt("sampledOther")
    with harness.ServerProcess(
        isolated_state_root, serve_args=["--speed-preset"]
    ) as srv:
        c = srv.client()
        r1 = c.post_chat(
            harness.make_chat_payload(
                p_a, enable_thinking=False, seed=19, temperature=0.5, top_k=20
            )
        )
        assert c.post_chat(harness.make_chat_payload(p_b, seed=20)).status == 200
        r3 = c.post_chat(
            harness.make_chat_payload(
                "Summarize again.",
                history=[
                    {"role": "user", "content": p_a},
                    {
                        "role": "assistant",
                        "content": r1.json()["choices"][0]["message"]["content"],
                    },
                ],
                enable_thinking=False,
                seed=19,
                temperature=0.5,
                top_k=20,
            )
        )
        assert r3.status == 200
        assert harness.validate_chat_completion_json(r3.json(), min_cached_tokens=1) > 0


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f7_05_respell_preserves_logit_sha(
    isolated_state_root: Path,
) -> None:
    p_a = harness.make_long_repeated_prompt("respellShaA")
    p_b = harness.make_long_repeated_prompt("respellShaB")
    with harness.ServerProcess(
        isolated_state_root / "cold", serve_args=["--speed-preset"]
    ) as srv_cold:
        cc = srv_cold.client()
        r1 = cc.post_chat(
            harness.make_chat_payload(p_a, enable_thinking=False, seed=42)
        )
        a_msg = r1.json()["choices"][0]["message"]["content"]
        r2 = cc.post_chat(
            harness.make_chat_payload(
                "Final short line.",
                history=[
                    {"role": "user", "content": p_a},
                    {"role": "assistant", "content": a_msg},
                ],
                enable_thinking=False,
                seed=42,
            )
        )
        assert r2.status == 200
    cold_out, _ = srv_cold.stop()
    cold_sha = harness.extract_done_events(harness.parse_jsonl_records(cold_out))[-1][
        "hc_exact_final_logit_sha256"
    ]

    with harness.ServerProcess(
        isolated_state_root / "parked", serve_args=["--speed-preset"]
    ) as srv_park:
        cp = srv_park.client()
        rp1 = cp.post_chat(
            harness.make_chat_payload(p_a, enable_thinking=False, seed=42)
        )
        assert cp.post_chat(harness.make_chat_payload(p_b, seed=43)).status == 200
        rp2 = cp.post_chat(
            harness.make_chat_payload(
                "Final short line.",
                history=[
                    {"role": "user", "content": p_a},
                    {
                        "role": "assistant",
                        "content": rp1.json()["choices"][0]["message"]["content"],
                    },
                ],
                enable_thinking=False,
                seed=42,
            )
        )
        assert rp2.status == 200
    park_out, _ = srv_park.stop()
    park_sha = harness.extract_done_events(harness.parse_jsonl_records(park_out))[-1][
        "hc_exact_final_logit_sha256"
    ]
    assert park_sha == cold_sha


# ---------------------------------------------------------------------------
# F8: C++ ServeSession Parity (serve_protocol.cpp)
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f8_01_rust_diff_all_arms_exact() -> None:
    res = harness.run_make_native(["rust-diff"], makefile="rust.mk")
    assert res.returncode == 0, res.stderr + res.stdout


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f8_02_cargo_test_server_wire() -> None:
    res = harness.run_cargo_test_checked(
        ["-p", "tokspur-server", "-p", "tokspur-wire"], min_passed=1
    )
    assert res.returncode == 0, res.stderr + res.stdout


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f8_03_rust_fuzz_diff_targets() -> None:
    res = harness.run_make_native(
        [
            "rust-fuzz",
            "FUZZ_SECONDS=5",
            "FUZZ_TARGETS=json_diff http response_diff bpe",
        ],
        makefile="rust.mk",
    )
    assert res.returncode == 0, res.stderr + res.stdout


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f8_04_sse_chunk_framing_parity() -> None:
    res = harness.run_make_native(["rust-diff"], makefile="rust.mk")
    assert res.returncode == 0, res.stderr
    diff_dir = harness.BUILD_DIR / "rust-diff"
    report = json.loads(
        (diff_dir / "server-report-channel.json").read_text(encoding="utf-8")
    )
    assert report["negative_control"] is False
    assert report["differences"] == []
    assert report["harness_failures"] == []
    assert report["cpp_exit"] == 0 and report["rust_exit"] == 0
    cpp_side = json.loads(
        (diff_dir / "server-report-channel-cpp.json").read_text(encoding="utf-8")
    )
    rust_side = json.loads(
        (diff_dir / "server-report-channel-rust.json").read_text(encoding="utf-8")
    )
    assert "data: [DONE]" in cpp_side["responses"]["stream"]
    assert "data: [DONE]" in rust_side["responses"]["stream"]


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f8_05_serve_protocol_parked_parity() -> None:
    res = harness.run_make_native(["test"], env={"NATIVE_CONFIG": "http-fast-math"})
    assert res.returncode == 0, res.stderr + res.stdout
    res_rust = harness.run_cargo_test_checked(
        ["-p", "tokspur-server", "--", "parked"], min_passed=1
    )
    assert res_rust.returncode == 0, res_rust.stderr + res_rust.stdout


# ---------------------------------------------------------------------------
# F9: graph.mm Phase B Level-Scheduling
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f9_01_test_graph_unit_suite() -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    bin_path = harness.require_artifact(
        harness.find_native_test_bin("test-graph"), "test-graph"
    )
    res = harness.run_cmd([str(bin_path), str(metallib)])
    assert res.returncode == 0, res.stderr + res.stdout
    assert "Persistent Metal graph checks passed:" in res.stdout


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f9_02_independent_level0_zero_barrier(synthetic_trace_factory) -> None:
    tb = harness.DispatchTraceBuilder()
    tb.add_pipeline(
        "p0", "rms_norm", [(0, "in", "read", True), (1, "out", "write", True)]
    )
    for slot in range(4):
        tb.add_allocation(slot, f"buf{slot}", 4096)
    bid = tb.begin_batch()
    tb.add_dispatch(
        "p0",
        [
            {"index": 0, "slot": 0, "offset": 0, "bytes": 4096, "access": "read"},
            {"index": 1, "slot": 1, "offset": 0, "bytes": 4096, "access": "write"},
        ],
    )
    tb.add_dispatch(
        "p0",
        [
            {"index": 0, "slot": 2, "offset": 0, "bytes": 4096, "access": "read"},
            {"index": 1, "slot": 3, "offset": 0, "bytes": 4096, "access": "write"},
        ],
    )
    tb.submit_batch(bid, 2, [(1, "s1"), (3, "s3")])
    trace_path = synthetic_trace_factory("indep.jsonl", tb)
    res = harness.run_dispatch_graph(["summary", str(trace_path)])
    assert res.returncode == 0, res.stderr
    summary = json.loads(res.stdout)
    assert summary["batches"][0]["barriers"] == 0
    assert summary["batches"][0]["depth"] == 1
    assert summary["batches"][0]["widest_level"] == 2


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f9_03_dag_depth_d_emits_d_minus_1(tmp_path: Path) -> None:
    ref_p, cand_p = harness.build_dag_pair_traces(tmp_path, depth=4)
    sum_res = harness.run_dispatch_graph(["summary", str(cand_p)])
    assert sum_res.returncode == 0
    batch = json.loads(sum_res.stdout)["batches"][0]
    assert batch["depth"] == 4 and batch["barriers"] == 3
    chk = harness.run_dispatch_graph(["check", str(ref_p), str(cand_p)])
    assert chk.returncode == 0, chk.stdout


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f9_04_dispatch_indirect_12b_span(synthetic_trace_factory) -> None:
    tb = harness.DispatchTraceBuilder()
    tb.add_pipeline("writer", "prep_indirect", [(0, "args", "write", True)])
    tb.add_pipeline(
        "consumer",
        "run_indirect",
        [(0, "args", "read", True), (1, "out", "write", True)],
    )
    tb.add_allocation(0, "indirect_args", 64)
    tb.add_allocation(1, "out", 4096)
    bid = tb.begin_batch()
    tb.add_dispatch(
        "writer",
        [{"index": 0, "slot": 0, "offset": 0, "bytes": 12, "access": "write"}],
    )
    tb.add_barrier()
    tb.add_dispatch(
        "consumer",
        [
            {"index": 0, "slot": 0, "offset": 0, "bytes": 12, "access": "read"},
            {"index": 1, "slot": 1, "offset": 0, "bytes": 4096, "access": "write"},
        ],
    )
    tb.submit_batch(bid, 2, [(0, "a0"), (1, "o1")])
    trace_path = synthetic_trace_factory("indirect.jsonl", tb)
    res = harness.run_dispatch_graph(["summary", str(trace_path)])
    assert res.returncode == 0, res.stderr
    batch = json.loads(res.stdout)["batches"][0]
    assert batch["dependency_edges"] >= 1 and batch["depth"] == 2


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f9_05_commit_partial_asap_schedule(tmp_path: Path) -> None:
    ref = harness.DispatchTraceBuilder()
    cand = harness.DispatchTraceBuilder()
    for b in (ref, cand):
        b.add_pipeline(
            "rms_norm:1",
            "rms_norm",
            [(0, "in", "read", True), (1, "out", "write", True)],
        )
        for slot in range(3):
            b.add_allocation(slot, f"w{slot}", 4096)
            b.add_host_write(slot, 0, 4096, f"winit_{slot}")
        for batch_idx in range(2):
            bid = b.begin_batch()
            b.add_dispatch(
                "rms_norm:1",
                [
                    {
                        "index": 0,
                        "slot": 0,
                        "offset": 0,
                        "bytes": 4096,
                        "access": "read",
                    },
                    {
                        "index": 1,
                        "slot": 1,
                        "offset": 0,
                        "bytes": 4096,
                        "access": "write",
                    },
                    {"index": 7, "buffer": "error_flag"},
                ],
                constants=[
                    {
                        "index": 3,
                        "bytes": 4,
                        "sha256": harness.sha256_bytes(f"b{batch_idx}_a".encode()),
                    }
                ],
            )
            b.add_barrier()
            b.add_dispatch(
                "rms_norm:1",
                [
                    {
                        "index": 0,
                        "slot": 1,
                        "offset": 0,
                        "bytes": 4096,
                        "access": "read",
                    },
                    {
                        "index": 1,
                        "slot": 2,
                        "offset": 0,
                        "bytes": 4096,
                        "access": "write",
                    },
                    {"index": 7, "buffer": "error_flag"},
                ],
                constants=[
                    {
                        "index": 3,
                        "bytes": 4,
                        "sha256": harness.sha256_bytes(f"b{batch_idx}_b".encode()),
                    }
                ],
            )
            if b is ref:
                b.add_barrier()
            b.submit_batch(
                bid,
                2,
                [(s, f"b{batch_idx}_s{s}") for s in range(3)],
            )
    ref_p = ref.write(tmp_path / "partial_ref.jsonl")
    cand_p = cand.write(tmp_path / "partial_cand.jsonl")
    chk = harness.run_dispatch_graph(["check", str(ref_p), str(cand_p)])
    assert chk.returncode == 0, chk.stdout + chk.stderr
    assert json.loads(chk.stdout)["passed"] is True
    ctrl = harness.run_dispatch_graph(["controls", str(ref_p)])
    assert ctrl.returncode == 0 and json.loads(ctrl.stdout)["held"] is True


# ---------------------------------------------------------------------------
# F10: dispatch-graph check Verification
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f10_01_cargo_test_dispatch_graph() -> None:
    res = harness.run_cargo_test_checked(["-p", "tokspur-dispatch-graph"], min_passed=1)
    assert res.returncode == 0, res.stderr + res.stdout


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f10_02_dispatch_graph_summary_json(tmp_path: Path) -> None:
    _, cand_p = harness.build_dag_pair_traces(tmp_path, depth=3)
    res = harness.run_dispatch_graph(["summary", str(cand_p)])
    assert res.returncode == 0, res.stderr
    data = json.loads(res.stdout)
    assert data["producer"] == "tokspur-cpp-metal-graph"
    assert data["pipelines"] == 1
    batch = data["batches"][0]
    assert batch["dispatches"] == 4
    assert batch["barriers"] == 2
    assert batch["depth"] == 3
    assert batch["widest_level"] == 2


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f10_03_dispatch_graph_check_pass(tmp_path: Path) -> None:
    ref_p, cand_p = harness.build_dag_pair_traces(tmp_path, depth=3)
    res = harness.run_dispatch_graph(["check", str(ref_p), str(cand_p)])
    assert res.returncode == 0, res.stdout + res.stderr
    rep = json.loads(res.stdout)
    assert rep["passed"] is True and rep["failures"] == 0


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f10_04_dispatch_graph_controls_hold(tmp_path: Path) -> None:
    ref_p, _ = harness.build_dag_pair_traces(tmp_path, depth=3)
    res = harness.run_dispatch_graph(["controls", str(ref_p)])
    assert res.returncode == 0, res.stdout + res.stderr
    ctrl = json.loads(res.stdout)
    assert ctrl["held"] is True
    assert len(ctrl["controls"]) >= 3


_DG_TRACE_PROBE_CPP = r"""
#include "tokspur/graph.hpp"
#include "tokspur/gguf.hpp"

#include <array>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

namespace {
struct Arena {
    tokspur::MetalGraph& gpu;
    std::vector<tokspur::GpuView> views;
    tokspur::GpuView upload(std::span<const float> values, const char* label, bool immutable = true) {
        auto v = gpu.upload(std::as_bytes(values), label, immutable);
        views.push_back(v);
        return v;
    }
    tokspur::GpuView zeros(uint64_t bytes, const char* label) {
        auto v = gpu.zeros(bytes, label);
        views.push_back(v);
        return v;
    }
    ~Arena() {
        try { gpu.abort(); } catch (const tokspur::FormatError&) {}
        for (const auto v : views) gpu.release(v);
    }
};
} // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: dg_trace_probe <metallib> <ref.jsonl> <cand.jsonl>\n";
        return 2;
    }
    const std::filesystem::path metallib(argv[1]);
    const auto run_arm = [&](const std::filesystem::path& trace_path, bool concurrent) {
        tokspur::MetalGraph gpu(metallib, 16ull << 20);
        gpu.configure_dispatch_trace(trace_path);
        gpu.configure_concurrent_dispatch(concurrent);
        Arena arena{gpu, {}};
        const std::array<float, 4> x_vals{1.0f, 2.0f, 3.0f, 4.0f};
        const auto x = arena.upload(x_vals, "dg_x", true);
        const auto a = arena.zeros(16, "dg_a");
        const auto b = arena.zeros(16, "dg_b");
        const auto c = arena.zeros(16, "dg_c");
        const auto d = arena.zeros(16, "dg_d");
        const auto out = arena.zeros(16, "dg_out");

        gpu.begin();
        // Interleaved 2-branch depth-3 DAG: greedy fence emits 3 barriers,
        // Phase B ASAP level-scheduling emits depth - 1 = 2 barriers.
        gpu.activate(x, a, tokspur::Activation::silu);
        gpu.activate(a, b, tokspur::Activation::sigmoid);
        gpu.activate(x, c, tokspur::Activation::identity, 2.0f);
        gpu.activate(c, d, tokspur::Activation::silu);
        gpu.multiply(b, d, out, false);
        const auto res = gpu.submit();
        if (res.dispatches != 5u) throw std::runtime_error("unexpected dispatch count");
        return gpu.read(out);
    };

    const auto ref_bytes = run_arm(argv[2], false);
    const auto cand_bytes = run_arm(argv[3], true);
    if (ref_bytes != cand_bytes) {
        throw std::runtime_error("serial vs concurrent MetalGraph output mismatch");
    }
    std::cout << "PASS f10 dg_trace_probe dispatches=5 depth=3\n";
    return 0;
}
"""


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f10_05_metal_graph_trace_roundtrip(tmp_path: Path) -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    dg_probe = harness.build_metal_graph_probe(
        tmp_path, "dg_trace_probe", _DG_TRACE_PROBE_CPP, dispatch_graph_trace=True
    )
    trace_bin = tmp_path / "diag-build/test-graph"
    assert trace_bin.exists()
    run_res = harness.run_cmd([str(trace_bin), str(metallib)])
    assert run_res.returncode == 0, run_res.stderr + run_res.stdout
    assert "Persistent Metal graph checks passed:" in run_res.stdout

    ref_trace = tmp_path / "ref.trace.jsonl"
    cand_trace = tmp_path / "cand.trace.jsonl"
    probe_res = harness.run_cmd(
        [str(dg_probe), str(metallib), str(ref_trace), str(cand_trace)]
    )
    assert probe_res.returncode == 0, probe_res.stderr + probe_res.stdout
    assert "PASS f10 dg_trace_probe dispatches=5 depth=3" in probe_res.stdout

    sum_res = harness.run_dispatch_graph(["summary", str(cand_trace)])
    assert sum_res.returncode == 0, sum_res.stderr + sum_res.stdout
    summary = json.loads(sum_res.stdout)
    assert summary["producer"] == "tokspur-cpp-metal-graph"
    batch = summary["batches"][0]
    assert batch["dispatches"] == 5
    assert batch["depth"] == 3
    assert batch["widest_level"] == 2
    assert batch["barriers"] == batch["depth"] - 1 == 2

    chk_res = harness.run_dispatch_graph(["check", str(ref_trace), str(cand_trace)])
    assert chk_res.returncode == 0, chk_res.stderr + chk_res.stdout
    rep = json.loads(chk_res.stdout)
    assert rep["passed"] is True and rep["failures"] == 0

    ctrl_res = harness.run_dispatch_graph(["controls", str(ref_trace)])
    assert ctrl_res.returncode == 0, ctrl_res.stderr + ctrl_res.stdout
    ctrl = json.loads(ctrl_res.stdout)
    assert ctrl["held"] is True
    assert len(ctrl["controls"]) >= 9


# ---------------------------------------------------------------------------
# F11: error_flag (buffer(7)) Decoupling
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f11_01_kernels_rs_three_way_sync() -> None:
    res = harness.run_cargo_test_checked(
        ["-p", "tokspur-dispatch-graph", "--test", "kernels"], min_passed=1
    )
    assert res.returncode == 0, res.stderr + res.stdout


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f11_02_disjoint_slots_same_level(synthetic_trace_factory) -> None:
    tb = harness.DispatchTraceBuilder()
    tb.add_pipeline(
        "rms_norm:1",
        "rms_norm",
        [(0, "x", "read", True), (1, "y", "write", True)],
    )
    for slot in range(4):
        tb.add_allocation(slot, f"b{slot}", 1024)
        tb.add_host_write(slot, 0, 1024, f"d_init_{slot}")
    bid = tb.begin_batch()
    tb.add_dispatch(
        "rms_norm:1",
        [
            {"index": 0, "slot": 0, "offset": 0, "bytes": 1024, "access": "read"},
            {"index": 1, "slot": 1, "offset": 0, "bytes": 1024, "access": "write"},
            {"index": 7, "buffer": "error_flag"},
        ],
        constants=[
            {"index": 3, "bytes": 4, "sha256": harness.sha256_bytes(b"f11_02_a")}
        ],
    )
    tb.add_dispatch(
        "rms_norm:1",
        [
            {"index": 0, "slot": 2, "offset": 0, "bytes": 1024, "access": "read"},
            {"index": 1, "slot": 3, "offset": 0, "bytes": 1024, "access": "write"},
            {"index": 7, "buffer": "error_flag"},
        ],
        constants=[
            {"index": 3, "bytes": 4, "sha256": harness.sha256_bytes(b"f11_02_b")}
        ],
    )
    tb.submit_batch(bid, 2, [(1, "f11_s1"), (3, "f11_s3")])
    path = synthetic_trace_factory("disjoint_flag.jsonl", tb)
    res = harness.run_dispatch_graph(["summary", str(path)])
    assert res.returncode == 0
    assert json.loads(res.stdout)["batches"][0]["barriers"] == 0


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f11_03_qsa_flag_readers_decoupled(synthetic_trace_factory) -> None:
    tb = harness.DispatchTraceBuilder()
    tb.add_pipeline("rms_norm:1", "rms_norm", [(0, "y", "write", True)])
    tb.add_pipeline(
        "qsa_score_accumulate:1", "qsa_score_accumulate", [(0, "x", "write", True)]
    )
    tb.add_allocation(0, "q", 1024)
    tb.add_allocation(1, "r", 1024)
    tb.add_host_write(0, 0, 1024, "q_init")
    tb.add_host_write(1, 0, 1024, "r_init")
    bid = tb.begin_batch()
    tb.add_dispatch(
        "rms_norm:1",
        [
            {"index": 0, "slot": 1, "offset": 0, "bytes": 1024, "access": "write"},
            {"index": 7, "buffer": "error_flag"},
        ],
        constants=[
            {"index": 3, "bytes": 4, "sha256": harness.sha256_bytes(b"qsa_writer")}
        ],
    )
    tb.add_barrier()
    tb.add_dispatch(
        "qsa_score_accumulate:1",
        [
            {"index": 0, "slot": 0, "offset": 0, "bytes": 1024, "access": "write"},
            {"index": 7, "buffer": "error_flag"},
        ],
        constants=[
            {"index": 3, "bytes": 4, "sha256": harness.sha256_bytes(b"qsa_reader")}
        ],
    )
    tb.submit_batch(bid, 2, [(0, "fa"), (1, "fb")])
    path = synthetic_trace_factory("qsa_decoupled.jsonl", tb)
    res = harness.run_dispatch_graph(["controls", str(path)])
    assert res.returncode == 0, res.stdout
    assert json.loads(res.stdout)["held"] is True


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f11_04_submit_host_reduction_detects() -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    bin_path = harness.require_artifact(
        harness.find_native_test_bin("test-sampled_stack"), "test-sampled_stack"
    )
    res = harness.run_cmd([str(bin_path), str(metallib), "--concurrent"])
    assert res.returncode == 0, res.stderr + res.stdout
    assert "nonfinite-output-refusals=8" in res.stdout


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f11_05_test_graph_operator_ckpt() -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    bin_path = harness.require_artifact(
        harness.find_native_test_bin("test-graph"), "test-graph"
    )
    res = harness.run_cmd([str(bin_path), str(metallib), "--operator-checkpoints"])
    assert res.returncode == 0, res.stderr + res.stdout
    assert "Generated operator checkpoint controls" in res.stdout


# ---------------------------------------------------------------------------
# F12: Default Concurrent Level-Scheduling
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f12_01_default_serve_enables_conc(isolated_state_root: Path) -> None:
    with harness.ServerProcess(isolated_state_root) as srv:
        assert srv.client().get_health().status == 200
    out, _ = srv.stop()
    recs = harness.parse_jsonl_records(out)
    ready = next(r for r in recs if r.get("event") == "ready")
    assert ready["native_ready"]["concurrent_dispatch_mode"] == "on"


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f12_02_sampled_stack_concurrent_parity() -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    bin_path = harness.require_artifact(
        harness.find_native_test_bin("test-sampled_stack"), "test-sampled_stack"
    )
    res_ser = harness.run_cmd([str(bin_path), str(metallib)])
    res_conc = harness.run_cmd([str(bin_path), str(metallib), "--concurrent"])
    assert res_ser.returncode == 0 and res_conc.returncode == 0
    assert "nonfinite-output-refusals=8" in res_ser.stdout
    assert "nonfinite-output-refusals=8" in res_conc.stdout


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f12_03_verify_profile_fewer_barriers(isolated_state_root: Path) -> None:
    prompt = "Count from 1 to 10."
    payload = harness.make_chat_payload(
        prompt, enable_thinking=False, temperature=0.0, seed=12, max_tokens=16
    )
    with harness.ServerProcess(
        isolated_state_root / "prof",
        serve_args=["--no-concurrent-dispatch"],
        extra_env={
            "TOKSPUR_DUMP_VERIFY_PROFILE": "1",
            "TOKSPUR_ALLOW_KNOWN_PROFILE": "1",
        },
    ) as srv_p:
        assert srv_p.client().post_chat(payload).status == 200
    out_p, err_p = srv_p.stop()
    vp = harness.parse_profile_lines(err_p, "VERIFY_PROFILE")
    assert len(vp) >= 1 and vp[0]["gpu_ms"] > 0.0 and vp[0]["dispatches"] > 0
    done_p = harness.extract_done_events(harness.parse_jsonl_records(out_p))[-1]
    cnt_p = done_p["gpu_lifetime_counters"]
    assert cnt_p["dispatches"] > 0
    assert cnt_p["barriers"] == cnt_p["dispatches"]

    with harness.ServerProcess(isolated_state_root / "conc") as srv_c:
        assert srv_c.client().post_chat(payload).status == 200
    out_c, _ = srv_c.stop()
    records_c = harness.parse_jsonl_records(out_c)
    ready = next(r for r in records_c if r.get("event") == "ready")
    assert ready["native_ready"]["concurrent_dispatch_mode"] == "on"
    done_c = harness.extract_done_events(records_c)[-1]
    assert done_c["concurrent_dispatch_mode"] == "on"
    assert (
        done_c["hc_exact_final_logit_sha256"] == done_p["hc_exact_final_logit_sha256"]
    )
    cnt_c = done_c["gpu_lifetime_counters"]
    assert cnt_c["dispatches"] > 0
    assert cnt_c["barriers"] < cnt_c["dispatches"]
    assert cnt_c["barriers"] < cnt_p["barriers"]
    assert done_c["concurrent_dispatch_barriers"] < vp[0]["dispatches"]


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f12_04_tokspur_cli_default_concurrent() -> None:
    res = harness.run_cargo_test_checked(["-p", "tokspur-cli"], min_passed=1)
    assert res.returncode == 0, res.stderr + res.stdout


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f12_05_no_concurrent_dispatch_optout(
    isolated_state_root: Path,
) -> None:
    with harness.ServerProcess(
        isolated_state_root, serve_args=["--no-concurrent-dispatch"]
    ) as srv:
        resp = srv.client().post_chat(
            harness.make_chat_payload("Say OK.", enable_thinking=False, max_tokens=8)
        )
        assert resp.status == 200


# ---------------------------------------------------------------------------
# F13: Pack v3 Static dm Certification
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f13_01_test_planar_pack_v3_passes() -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    bin_path = harness.require_artifact(
        harness.find_native_test_bin("test-planar_pack_v3"), "test-planar_pack_v3"
    )
    res = harness.run_cmd([str(bin_path), str(metallib)])
    assert res.returncode == 0, res.stderr + res.stdout
    assert "PASS test-planar_pack_v3" in res.stdout
    assert "PASS planar_v3 builder, multi-worker adoption" in res.stdout


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f13_02_pack_v3_plan_json_valid(tmp_path: Path) -> None:
    weights_dir = harness.REPO_ROOT / ".tokspur/own-weights/qwen-34567a47"
    shards = sorted(weights_dir.glob("Qwen3.8-Flash-Next-Q4_K-*-of-00004.gguf"))
    mtp = weights_dir / "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
    assert len(shards) == 4 and mtp.is_file()
    native_bin = harness.require_artifact(
        harness.find_tokspur_native(), "tokspur-native"
    )
    cmd = [
        str(native_bin),
        "pack-v3-plan",
        "--output",
        str(tmp_path / "plan.asa.gguf"),
        "--mtp",
        str(mtp),
        "--reserve-bytes",
        "1073741824",
    ]
    for shard in shards:
        cmd.extend(["--gguf", str(shard)])
    res = harness.run_cmd(cmd)
    assert res.returncode == 0, res.stderr + res.stdout
    plan = json.loads(res.stdout)
    assert plan["recipe"] == "tokspur-pack-v3"
    assert plan["version"] == 3
    assert plan["tensor_count"] == 1254
    assert plan["payload_bytes"] == 85881974272
    assert isinstance(plan["disk_ok"], bool)
    assert plan["ram_ok"] is True


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f13_03_build_pack_v3_seals_output() -> None:
    derived_bin = harness.require_artifact(
        harness.find_native_test_bin("test-derived_pack"), "test-derived_pack"
    )
    derived_res = harness.run_cmd([str(derived_bin)])
    assert derived_res.returncode == 0, derived_res.stderr + derived_res.stdout
    assert "PASS derived pack checks=" in derived_res.stdout
    assert "v3_auth_source_read_mode=" in derived_res.stdout

    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    pack_bin = harness.require_artifact(
        harness.find_native_test_bin("test-planar_pack_v3"), "test-planar_pack_v3"
    )
    pack_res = harness.run_cmd([str(pack_bin), str(metallib)])
    assert pack_res.returncode == 0, pack_res.stderr + pack_res.stdout
    assert "PASS planar_v3 builder, multi-worker adoption" in pack_res.stdout


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f13_04_rebuild_verifies_existing(tmp_path: Path) -> None:
    derived_bin = harness.require_artifact(
        harness.find_native_test_bin("test-derived_pack"), "test-derived_pack"
    )
    derived_res = harness.run_cmd([str(derived_bin)])
    assert derived_res.returncode == 0, derived_res.stderr + derived_res.stdout
    assert "PASS derived pack checks=" in derived_res.stdout
    assert "v3_auth_source_read_mode=" in derived_res.stdout

    weights_dir = harness.REPO_ROOT / ".tokspur/own-weights/qwen-34567a47"
    shards = sorted(weights_dir.glob("Qwen3.8-Flash-Next-Q4_K-*-of-00004.gguf"))
    mtp = weights_dir / "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
    assert len(shards) == 4 and mtp.is_file()
    native_bin = harness.require_artifact(
        harness.find_tokspur_native(), "tokspur-native"
    )
    plans = []
    for name in ("plan_a.asa.gguf", "plan_b.asa.gguf"):
        cmd = [
            str(native_bin),
            "pack-v3-plan",
            "--output",
            str(tmp_path / name),
            "--mtp",
            str(mtp),
            "--reserve-bytes",
            "1073741824",
        ]
        for shard in shards:
            cmd.extend(["--gguf", str(shard)])
        res = harness.run_cmd(cmd)
        assert res.returncode == 0, res.stderr + res.stdout
        plans.append(json.loads(res.stdout))
    assert plans[0]["tensor_count"] == plans[1]["tensor_count"] == 1254
    assert plans[0]["payload_bytes"] == plans[1]["payload_bytes"] == 85881974272
    assert plans[0]["file_bytes"] == plans[1]["file_bytes"]


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f13_05_local_seal_v2_accepted() -> None:
    derived_bin = harness.require_artifact(
        harness.find_native_test_bin("test-derived_pack"), "test-derived_pack"
    )
    derived_res = harness.run_cmd([str(derived_bin)])
    assert derived_res.returncode == 0, derived_res.stderr + derived_res.stdout
    assert "PASS derived pack checks=" in derived_res.stdout
    assert "v3_auth_source_read_mode=" in derived_res.stdout

    help_res = harness.run_tokspur(["serve", "--help"])
    assert help_res.returncode == 0, help_res.stderr + help_res.stdout
    assert "--derived-pack-trust-mode" in help_res.stdout

    real_seal = harness.REPO_ROOT / ".tokspur/derived/tokspur-pack-v3.asa.seal"
    if real_seal.is_file():
        seal_lines = real_seal.read_text(encoding="utf-8").splitlines()
        is_v2 = (
            len(seal_lines) == 8
            and seal_lines[0] == "tokspur-local-seal-v2"
            and any(ln.startswith("manifest_sha256=") for ln in seal_lines)
        )
        is_v1_legacy = (
            len(seal_lines) == 1
            and len(seal_lines[0]) == 64
            and all(ch in "0123456789abcdefABCDEF" for ch in seal_lines[0])
        )
        assert is_v2 or is_v1_legacy, f"Unexpected local seal format: {seal_lines!r}"


# ---------------------------------------------------------------------------
# F14: Shape-Registration Invariant Discharge
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f14_01_register_planar_matrix_valid() -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    bin_path = harness.require_artifact(
        harness.find_native_test_bin("test-planar_union_profile"),
        "test-planar_union_profile",
    )
    res = harness.run_cmd([str(bin_path), str(metallib)])
    assert res.returncode == 0, res.stderr + res.stdout
    combined_output = f"{res.stdout}\n{res.stderr}"
    assert "union" in combined_output.lower()


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f14_02_m5_headline_screen_dual_pass(tmp_path: Path) -> None:
    for screen in ("0", "1"):
        bdir = tmp_path / f"screen-{screen}"
        res = harness.run_make_native(
            [
                f"M5_HEADLINE_SCREEN={screen}",
                f"BUILD={bdir}",
                f"{bdir}/tokspur-owned.metallib",
                f"{bdir}/test-planar_pack_v3",
            ]
        )
        assert res.returncode == 0, res.stderr
        run_res = harness.run_cmd(
            [str(bdir / "test-planar_pack_v3"), str(bdir / "tokspur-owned.metallib")]
        )
        assert run_res.returncode == 0, run_res.stderr + run_res.stdout


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f14_03_valid_routes_up_to_experts() -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    bin_path = harness.require_artifact(
        harness.find_native_test_bin("test-planar_union_metadata"),
        "test-planar_union_metadata",
    )
    res = harness.run_cmd([str(bin_path), str(metallib)])
    assert res.returncode == 0, res.stderr + res.stdout
    assert "ordered planar union metadata cases passed" in res.stdout


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f14_04_small_m_and_nax_bounds_hold() -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    bin_path = harness.require_artifact(
        harness.find_native_test_bin("test-planar_pack_v3"), "test-planar_pack_v3"
    )
    res = harness.run_cmd([str(bin_path), str(metallib), "--nax-staging-only"])
    assert res.returncode == 0, res.stderr + res.stdout
    assert "PASS planar_v3 sticky finite-value contract" in res.stdout


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f14_05_native_cpu_discharge_tests() -> None:
    res = harness.run_make_native(["test"])
    assert res.returncode == 0, res.stderr + res.stdout
    assert "PASS" in res.stdout or "passed" in res.stdout


# ---------------------------------------------------------------------------
# F15: Strip Hot-Loop Metal Checks & bad Live Ranges
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f15_01_stripped_planar_moe_nmse() -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    bin_path = harness.require_artifact(
        harness.find_native_test_bin("test-planar_pack_v3"), "test-planar_pack_v3"
    )
    res = harness.run_cmd([str(bin_path), str(metallib)])
    assert res.returncode == 0, res.stderr + res.stdout
    assert "small-M (m=1..8) and NAX grouped (5-dispatch) parity" in res.stdout


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f15_02_stripped_dflash2_stage_pass() -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    bin_path = harness.require_artifact(
        harness.find_native_test_bin("test-dflash2_shortlist"),
        "test-dflash2_shortlist",
    )
    res = harness.run_cmd([str(bin_path), str(metallib)])
    assert res.returncode == 0, res.stderr + res.stdout
    assert "PASS" in res.stdout or "passed" in res.stdout


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f15_03_gu_stage64_column32_parity(tmp_path: Path) -> None:
    b0 = tmp_path / "b0"
    b1 = tmp_path / "b1"
    for bdir, screen in ((b0, "0"), (b1, "1")):
        res = harness.run_make_native(
            [
                f"M5_HEADLINE_SCREEN={screen}",
                f"BUILD={bdir}",
                f"{bdir}/tokspur-owned.metallib",
                f"{bdir}/test-planar_pack_v3",
            ]
        )
        assert res.returncode == 0, res.stderr
    run_res = harness.run_cmd(
        [
            str(b1 / "test-planar_pack_v3"),
            str(b1 / "tokspur-owned.metallib"),
            "--nax-gu-stage64-control",
            str(b0 / "tokspur-owned.metallib"),
        ]
    )
    assert run_res.returncode == 0, run_res.stderr + run_res.stdout


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f15_04_output_sentinel_untouched() -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    bin_path = harness.require_artifact(
        harness.find_native_test_bin("test-planar_pack_v3"), "test-planar_pack_v3"
    )
    res = harness.run_cmd([str(bin_path), str(metallib), "--nax-staging-only"])
    assert res.returncode == 0, res.stderr + res.stdout
    assert "output guards" in res.stdout


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f15_05_metallib_objdump_symbols() -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    symbols = harness.list_metallib_symbols(metallib)
    expected = {
        "planar_moe_gate_up_swiglu_m1",
        "planar_moe_gate_up_swiglu_m4",
        "planar_moe_gate_up_swiglu_m8",
        "planar_moe_down_m1",
        "planar_moe_down_m4",
        "planar_moe_down_m8",
        "nax_grouped_moe_gate_up_swiglu_v3",
        "dflash2_dense_bf16_m8",
    }
    assert expected <= symbols


# ---------------------------------------------------------------------------
# F16: End-of-Batch Finiteness & Metal 3.2 Flags
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f16_01_metal32_40_fp_policy() -> None:
    out = harness.query_make_dry_run_flags("tokspur-owned.metallib")
    cmds = harness.parse_metal_compile_commands(out)
    assert len(cmds) >= 10
    harness.assert_metal_fp_policy(out)


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f16_02_test_logit_topk_tracking() -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    bin_path = harness.require_artifact(
        harness.find_native_test_bin("test-logit_topk"), "test-logit_topk"
    )
    res = harness.run_cmd([str(bin_path), str(metallib)])
    assert res.returncode == 0, res.stderr + res.stdout
    assert "exact CPU/GPU parity" in res.stdout


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f16_03_test_sampled_stack_nan_catch() -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    bin_path = harness.require_artifact(
        harness.find_native_test_bin("test-sampled_stack"), "test-sampled_stack"
    )
    res = harness.run_cmd([str(bin_path), str(metallib)])
    assert res.returncode == 0, res.stderr + res.stdout
    assert "nonfinite-output-refusals=8" in res.stdout


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f16_04_stack_no_topk_blindspot_fix() -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    bin_path = harness.require_artifact(
        harness.find_native_test_bin("test-stack"), "test-stack"
    )
    res = harness.run_cmd([str(bin_path), str(metallib)])
    assert res.returncode == 0, res.stderr + res.stdout
    assert "Owned complete-stack fixture checks passed:" in res.stdout


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f16_05_dflash2_selector_nan_catch() -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    bin_path = harness.require_artifact(
        harness.find_native_test_bin("test-dflash2_gather"),
        "test-dflash2_gather",
    )
    res = harness.run_cmd([str(bin_path), str(metallib), "--correctness-only"])
    assert res.returncode == 0, res.stderr + res.stdout
    assert "PASS DFlash2 gathered draft vocabulary" in res.stdout


# ---------------------------------------------------------------------------
# F17: DFlash2 Lazy flush_context() on Park
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f17_01_dflash2_defer_parks_no_skip(isolated_state_root: Path) -> None:
    p_a = harness.make_long_repeated_prompt("dfParkA")
    p_b = harness.make_long_repeated_prompt("dfParkB")
    with harness.ServerProcess(
        isolated_state_root,
        serve_args=[
            "--speed-preset",
            "--drafter",
            "dflash2",
            "--native-dflash2-defer-context",
            "on",
        ],
    ) as srv:
        c = srv.client()
        assert (
            c.post_chat(harness.make_chat_payload(p_a, enable_thinking=False)).status
            == 200
        )
        assert (
            c.post_chat(harness.make_chat_payload(p_b, enable_thinking=False)).status
            == 200
        )
    out, _ = srv.stop()
    done = harness.extract_done_events(harness.parse_jsonl_records(out))
    assert done[-1]["native_dflash2_defer_context"] is True
    assert done[-1]["parked_lanes"]["parks"] >= 1
    assert done[-1]["parked_lanes"]["skips"] == 0


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f17_02_dflash2_parked_ram_restore(isolated_state_root: Path) -> None:
    p_a = harness.make_long_repeated_prompt("dfRestoreA")
    p_b = harness.make_long_repeated_prompt("dfRestoreB")
    with harness.ServerProcess(
        isolated_state_root,
        serve_args=["--speed-preset", "--drafter", "dflash2"],
    ) as srv:
        c = srv.client()
        ra1 = c.post_chat(harness.make_chat_payload(p_a, enable_thinking=False, seed=3))
        assert (
            c.post_chat(
                harness.make_chat_payload(p_b, enable_thinking=False, seed=4)
            ).status
            == 200
        )
        ra2 = c.post_chat(
            harness.make_chat_payload(
                "Continue dfRestoreA.",
                history=[
                    {"role": "user", "content": p_a},
                    {
                        "role": "assistant",
                        "content": ra1.json()["choices"][0]["message"]["content"],
                    },
                ],
                enable_thinking=False,
                seed=3,
            )
        )
        assert ra2.status == 200
        assert (
            harness.validate_chat_completion_json(ra2.json(), min_cached_tokens=1) > 0
        )
    out, _ = srv.stop()
    done = harness.extract_done_events(harness.parse_jsonl_records(out))
    assert done[-1]["parked_lanes"]["restores"] >= 1
    assert done[-1]["parked_lanes"]["restored_tokens"] > 0


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f17_03_dflash2_sequential_stays_defer(
    isolated_state_root: Path,
) -> None:
    p_a = harness.make_long_repeated_prompt("dfSeq")
    with harness.ServerProcess(
        isolated_state_root,
        serve_args=["--speed-preset", "--drafter", "dflash2"],
    ) as srv:
        c = srv.client()
        r1 = c.post_chat(harness.make_chat_payload(p_a, enable_thinking=False, seed=5))
        r2 = c.post_chat(
            harness.make_chat_payload(
                "Sequential turn 2.",
                history=[
                    {"role": "user", "content": p_a},
                    {
                        "role": "assistant",
                        "content": r1.json()["choices"][0]["message"]["content"],
                    },
                ],
                enable_thinking=False,
                seed=5,
            )
        )
        assert r2.status == 200
    out, _ = srv.stop()
    done = harness.extract_done_events(harness.parse_jsonl_records(out))
    assert done[-1]["native_dflash2_defer_context"] is True
    assert done[-1]["parked_lanes"]["parks"] == 0


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f17_04_dflash2_parked_logit_parity(isolated_state_root: Path) -> None:
    p_a = harness.make_long_repeated_prompt("dfShaA")
    p_b = harness.make_long_repeated_prompt("dfShaB")
    with harness.ServerProcess(
        isolated_state_root / "uninterrupted",
        serve_args=[
            "--speed-preset",
            "--drafter",
            "dflash2",
            "--native-dflash2-verify-rows",
            "all",
        ],
    ) as srv_u:
        cu = srv_u.client()
        u1 = cu.post_chat(
            harness.make_chat_payload(p_a, enable_thinking=False, seed=71)
        )
        assert u1.status == 200
        u2 = cu.post_chat(
            harness.make_chat_payload(
                "Final answer only.",
                history=[
                    {"role": "user", "content": p_a},
                    {
                        "role": "assistant",
                        "content": u1.json()["choices"][0]["message"]["content"],
                    },
                ],
                enable_thinking=False,
                seed=71,
            )
        )
        assert u2.status == 200
    out_u, _ = srv_u.stop()
    done_u = harness.extract_done_events(harness.parse_jsonl_records(out_u))
    sha_u = done_u[-1]["hc_exact_final_logit_sha256"]

    with harness.ServerProcess(
        isolated_state_root / "parked",
        serve_args=[
            "--speed-preset",
            "--drafter",
            "dflash2",
            "--native-dflash2-verify-rows",
            "all",
        ],
    ) as srv_p:
        cp = srv_p.client()
        p1 = cp.post_chat(
            harness.make_chat_payload(p_a, enable_thinking=False, seed=71)
        )
        assert p1.status == 200
        assert (
            p1.json()["choices"][0]["message"]["content"]
            == u1.json()["choices"][0]["message"]["content"]
        )
        assert cp.post_chat(harness.make_chat_payload(p_b, seed=72)).status == 200
        p2 = cp.post_chat(
            harness.make_chat_payload(
                "Final answer only.",
                history=[
                    {"role": "user", "content": p_a},
                    {
                        "role": "assistant",
                        "content": p1.json()["choices"][0]["message"]["content"],
                    },
                ],
                enable_thinking=False,
                seed=71,
            )
        )
        assert p2.status == 200
    out_p, _ = srv_p.stop()
    done_p = harness.extract_done_events(harness.parse_jsonl_records(out_p))
    assert done_p[0]["native_mtp_proposed"] == done_u[0]["native_mtp_proposed"]
    assert done_p[0]["native_mtp_accepted"] == done_u[0]["native_mtp_accepted"]
    assert (
        done_p[0]["hc_exact_final_logit_sha256"]
        == done_u[0]["hc_exact_final_logit_sha256"]
    )
    assert done_p[-1]["native_dflash2_defer_context"] is True
    assert done_p[-1]["parked_lanes"]["skips"] == 0
    assert done_p[-1]["parked_lanes"]["restores"] >= 1
    sha_p = done_p[-1]["hc_exact_final_logit_sha256"]
    assert sha_p == sha_u


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f17_05_dflash2_ctx_pos_consistent(isolated_state_root: Path) -> None:
    p_a = harness.make_long_repeated_prompt("dfCtxA")
    p_b = harness.make_long_repeated_prompt("dfCtxB")
    with harness.ServerProcess(
        isolated_state_root,
        serve_args=["--speed-preset", "--drafter", "dflash2"],
    ) as srv:
        c = srv.client()
        r1 = c.post_chat(harness.make_chat_payload(p_a, enable_thinking=False, seed=81))
        assert c.post_chat(harness.make_chat_payload(p_b, seed=82)).status == 200
        r3 = c.post_chat(
            harness.make_chat_payload(
                "Continue dfCtxA.",
                history=[
                    {"role": "user", "content": p_a},
                    {
                        "role": "assistant",
                        "content": r1.json()["choices"][0]["message"]["content"],
                    },
                ],
                enable_thinking=False,
                seed=81,
            )
        )
        assert r3.status == 200
    out, _ = srv.stop()
    done = harness.extract_done_events(harness.parse_jsonl_records(out))
    pos = done[-1]["native_dflash2_context_position"]
    assert pos > 0
    assert done[-1]["native_dflash2_cache_rows"] == min(pos, 2048)
    assert (
        done[-1]["native_dflash2_first_cache_position"]
        + done[-1]["native_dflash2_cache_rows"]
        == pos
    )


# ---------------------------------------------------------------------------
# F18: Sqrt(N) Prefix/SSD Checkpoint Spacing
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f18_01_sqrt_n_checkpoints_retained(isolated_state_root: Path) -> None:
    long_prompt = harness.make_long_repeated_prompt("ckpt64k", approx_lines=2500)
    with harness.ServerProcess(
        isolated_state_root,
        serve_args=[
            "--context-length",
            "131072",
            "--checkpoint-cap-gib",
            "4.0",
            "--checkpoint-floor-gib",
            "1.0",
            "--prefill-dense",
            "nax",
            "--prefill-grouped",
            "nax",
        ],
    ) as srv:
        resp = srv.client().post_chat(
            harness.make_chat_payload(
                long_prompt, enable_thinking=False, max_tokens=24
            ),
            timeout=900.0,
        )
        assert resp.status == 200
        assert resp.json()["usage"]["prompt_tokens"] > 65536
    out, _ = srv.stop()
    done = harness.extract_done_events(harness.parse_jsonl_records(out))[-1]
    assert done["checkpoint_ssd_enabled"] is True
    assert done["checkpoint_ssd_spills"] >= 1
    assert done["checkpoint_ssd_occupied_bytes"] > 0


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f18_02_mid_context_branch_restores(isolated_state_root: Path) -> None:
    prefix = harness.make_long_repeated_prompt("branchMid", approx_lines=2500)
    with harness.ServerProcess(
        isolated_state_root,
        serve_args=[
            "--context-length",
            "131072",
            "--checkpoint-cap-gib",
            "4.0",
            "--checkpoint-floor-gib",
            "1.0",
            "--prefill-dense",
            "nax",
            "--prefill-grouped",
            "nax",
        ],
    ) as srv:
        c = srv.client()
        assert (
            c.post_chat(
                harness.make_chat_payload(prefix + "\nLeaf 1.", enable_thinking=False),
                timeout=900.0,
            ).status
            == 200
        )
        r2 = c.post_chat(
            harness.make_chat_payload(prefix + "\nLeaf 2.", enable_thinking=False),
            timeout=900.0,
        )
        assert r2.status == 200
        assert r2.json()["usage"]["prompt_tokens"] > 65536
        assert harness.validate_chat_completion_json(r2.json(), min_cached_tokens=1) > 0
    out, _ = srv.stop()
    done = harness.extract_done_events(harness.parse_jsonl_records(out))[-1]
    assert done["checkpoint_ssd_restored"] is True or done["checkpoint_reuse"] in (
        "restored",
        "ssd_restored",
    )
    assert done["checkpoint_restored_tokens"] > 0


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f18_03_replayed_tokens_sublinear(isolated_state_root: Path) -> None:
    prefix = harness.make_long_repeated_prompt("sublinear", approx_lines=2500)
    with harness.ServerProcess(
        isolated_state_root,
        serve_args=[
            "--context-length",
            "131072",
            "--checkpoint-cap-gib",
            "4.0",
            "--checkpoint-floor-gib",
            "1.0",
            "--prefill-dense",
            "nax",
            "--prefill-grouped",
            "nax",
        ],
    ) as srv:
        c = srv.client()
        assert (
            c.post_chat(
                harness.make_chat_payload(
                    prefix + "\nBranch A.", enable_thinking=False
                ),
                timeout=900.0,
            ).status
            == 200
        )
        r2 = c.post_chat(
            harness.make_chat_payload(prefix + "\nBranch B.", enable_thinking=False),
            timeout=900.0,
        )
        assert r2.status == 200
        total_prompt = r2.json()["usage"]["prompt_tokens"]
        assert total_prompt > 65536
    out, _ = srv.stop()
    done = harness.extract_done_events(harness.parse_jsonl_records(out))[-1]
    assert done["checkpoint_restored_tokens"] > 0
    assert done["replayed_input_tokens"] < total_prompt // 2


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f18_04_ssd_checkpoint_spill_restore(isolated_state_root: Path) -> None:
    p1 = harness.make_long_repeated_prompt("ssd1", approx_lines=2500)
    p2 = harness.make_long_repeated_prompt("ssd2", approx_lines=2500)
    with harness.ServerProcess(
        isolated_state_root,
        serve_args=[
            "--context-length",
            "131072",
            "--checkpoint-cap-gib",
            "6.0",
            "--checkpoint-floor-gib",
            "0.5",
            "--prefill-dense",
            "nax",
            "--prefill-grouped",
            "nax",
        ],
    ) as srv:
        c = srv.client()
        assert (
            c.post_chat(
                harness.make_chat_payload(p1 + "\nA", enable_thinking=False),
                timeout=900.0,
            ).status
            == 200
        )
        assert (
            c.post_chat(
                harness.make_chat_payload(p2 + "\nB", enable_thinking=False),
                timeout=900.0,
            ).status
            == 200
        )
        r3 = c.post_chat(
            harness.make_chat_payload(p1 + "\nC", enable_thinking=False),
            timeout=900.0,
        )
        assert r3.status == 200
        assert r3.json()["usage"]["prompt_tokens"] > 65536
        assert harness.validate_chat_completion_json(r3.json(), min_cached_tokens=1) > 0
    out, _ = srv.stop()
    done = harness.extract_done_events(harness.parse_jsonl_records(out))[-1]
    assert done["checkpoint_ssd_enabled"] is True
    assert done["checkpoint_ssd_spills"] >= 1
    assert done["checkpoint_ssd_restores"] >= 1
    assert done["checkpoint_ssd_restored"] is True


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f18_05_long_ctx_branch_logit_parity(isolated_state_root: Path) -> None:
    prefix = harness.make_long_repeated_prompt("ctxSha", approx_lines=2500)
    with harness.ServerProcess(
        isolated_state_root / "cold",
        serve_args=[
            "--context-length",
            "131072",
            "--prefill-dense",
            "f32-staged",
            "--prefill-grouped",
            "f32",
        ],
    ) as srv_cold:
        rc = srv_cold.client().post_chat(
            harness.make_chat_payload(
                prefix + "\nTarget.", enable_thinking=False, seed=5, max_tokens=16
            ),
            timeout=1200.0,
        )
        assert rc.status == 200
        assert rc.json()["usage"]["prompt_tokens"] > 65536
    out_c, _ = srv_cold.stop()
    done_c = harness.extract_done_events(harness.parse_jsonl_records(out_c))[-1]
    sha_c = done_c["hc_exact_final_logit_sha256"]

    with harness.ServerProcess(
        isolated_state_root / "ckpt",
        serve_args=[
            "--context-length",
            "131072",
            "--checkpoint-cap-gib",
            "4.0",
            "--checkpoint-floor-gib",
            "1.0",
            "--prefill-dense",
            "f32-staged",
            "--prefill-grouped",
            "f32",
        ],
    ) as srv_k:
        ck = srv_k.client()
        assert (
            ck.post_chat(
                harness.make_chat_payload(
                    prefix + "\nWarm.", enable_thinking=False, seed=5, max_tokens=16
                ),
                timeout=1200.0,
            ).status
            == 200
        )
        rk = ck.post_chat(
            harness.make_chat_payload(
                prefix + "\nTarget.", enable_thinking=False, seed=5, max_tokens=16
            ),
            timeout=1200.0,
        )
        assert rk.status == 200
    out_k, _ = srv_k.stop()
    done_k = harness.extract_done_events(harness.parse_jsonl_records(out_k))[-1]
    assert done_k["checkpoint_ssd_restored"] is True
    assert done_k["checkpoint_restored_tokens"] > 0
    assert done_k["hc_exact_final_logit_sha256"] == sha_c
    assert done_k["generated_ids"] == done_c["generated_ids"]
    assert (
        rk.json()["choices"][0]["message"]["content"]
        == rc.json()["choices"][0]["message"]["content"]
    )

    with harness.ServerProcess(
        isolated_state_root / "cold_nax",
        serve_args=[
            "--context-length",
            "131072",
            "--prefill-dense",
            "nax",
            "--prefill-grouped",
            "nax",
        ],
    ) as srv_cold_nax:
        rc_nax = srv_cold_nax.client().post_chat(
            harness.make_chat_payload(
                prefix + "\nTarget.", enable_thinking=False, seed=5, max_tokens=16
            ),
            timeout=1200.0,
        )
        assert rc_nax.status == 200
        assert rc_nax.json()["usage"]["prompt_tokens"] > 65536
    out_c_nax, _ = srv_cold_nax.stop()
    done_c_nax = harness.extract_done_events(harness.parse_jsonl_records(out_c_nax))[-1]

    with harness.ServerProcess(
        isolated_state_root / "ckpt_nax",
        serve_args=[
            "--context-length",
            "131072",
            "--checkpoint-cap-gib",
            "4.0",
            "--checkpoint-floor-gib",
            "1.0",
            "--prefill-dense",
            "nax",
            "--prefill-grouped",
            "nax",
        ],
    ) as srv_k_nax:
        ck_nax = srv_k_nax.client()
        assert (
            ck_nax.post_chat(
                harness.make_chat_payload(
                    prefix + "\nWarm.", enable_thinking=False, seed=5, max_tokens=16
                ),
                timeout=1200.0,
            ).status
            == 200
        )
        rk_nax = ck_nax.post_chat(
            harness.make_chat_payload(
                prefix + "\nTarget.", enable_thinking=False, seed=5, max_tokens=16
            ),
            timeout=1200.0,
        )
        assert rk_nax.status == 200
    out_k_nax, _ = srv_k_nax.stop()
    done_k_nax = harness.extract_done_events(harness.parse_jsonl_records(out_k_nax))[-1]
    assert done_k_nax["checkpoint_ssd_restored"] is True
    assert done_k_nax["checkpoint_restored_tokens"] > 0
    assert done_k_nax["generated_ids"] == done_c_nax["generated_ids"]
    assert (
        rk_nax.json()["choices"][0]["message"]["content"]
        == rc_nax.json()["choices"][0]["message"]["content"]
    )


# ---------------------------------------------------------------------------
# F19: Instantaneous Anchor-Entropy Width Gating
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f19_01_low_entropy_wide_draft_lookup(isolated_state_root: Path) -> None:
    repetitive = (
        "Repeat the exact sequence: alpha_01, alpha_02, alpha_03, alpha_04, "
        "alpha_05, alpha_06, alpha_07, alpha_08, alpha_09, alpha_10.\n"
        * 8
        + "Now repeat it verbatim once:"
    )
    with harness.ServerProcess(isolated_state_root) as srv:
        resp = srv.client().post_chat(
            harness.make_chat_payload(repetitive, enable_thinking=False, max_tokens=48)
        )
        assert resp.status == 200
    out, _ = srv.stop()
    done = harness.extract_done_events(harness.parse_jsonl_records(out))[-1]
    assert (
        done["native_mtp_accepted"] > 0 or (done.get("lookup_accepted_tokens") or 0) > 0
    )


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f19_02_high_entropy_narrows_width(isolated_state_root: Path) -> None:
    low_prompt = ("alpha beta gamma delta\n" * 16) + "Continue exact pattern:"
    high_prompt = "Invent an unexpected surreal poem combining clockwork and moss."
    with harness.ServerProcess(isolated_state_root) as srv:
        c = srv.client()
        r_lo = c.post_chat(
            harness.make_chat_payload(low_prompt, enable_thinking=False, max_tokens=32)
        )
        assert r_lo.status == 200
        lo_done = harness.wait_for_done_events(srv.stdout_log_path, min_count=1)[-1]
        r_hi = c.post_chat(
            harness.make_chat_payload(
                high_prompt,
                enable_thinking=False,
                temperature=0.8,
                top_k=20,
                seed=99,
                max_tokens=48,
            )
        )
        assert r_hi.status == 200
    out, _ = srv.stop()
    hi_done = harness.extract_done_events(harness.parse_jsonl_records(out))[-1]
    lo_rounds = lo_done["native_mtp_rounds"]
    lo_proposed = lo_done["native_mtp_proposed"]
    hi_rounds = hi_done["native_mtp_rounds"] - lo_rounds
    hi_proposed = hi_done["native_mtp_proposed"] - lo_proposed
    assert lo_rounds >= 1 and hi_rounds >= 1
    assert lo_proposed >= 1 and hi_proposed >= 1
    assert (hi_proposed / hi_rounds) < (lo_proposed / lo_rounds)


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f19_03_zero_full_vocab_readback_stall(
    isolated_state_root: Path,
) -> None:
    with harness.ServerProcess(isolated_state_root) as srv:
        resp = srv.client().post_chat(
            harness.make_chat_payload(
                "Write a small Rust struct Point { x: i32, y: i32 }.",
                enable_thinking=False,
                max_tokens=32,
            )
        )
        assert resp.status == 200
    out, _ = srv.stop()
    records = harness.parse_jsonl_records(out)
    ready = next(r for r in records if r.get("event") == "ready")
    assert ready["native_ready"]["concurrent_dispatch_mode"] == "on"
    done = harness.extract_done_events(records)[-1]
    assert done["native_mtp_rounds"] >= 2
    assert done["native_mtp_carried_anchors"] >= 1
    assert done["native_mtp_discarded_anchors"] == 0


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f19_04_entropy_gate_mtp3_and_dflash2(
    isolated_state_root: Path,
) -> None:
    drafter_args = (
        ["--drafter", "mtp", "--mtp-depth", "3"],
        ["--drafter", "dflash2"],
    )
    for idx, args in enumerate(drafter_args):
        with harness.ServerProcess(
            isolated_state_root / f"d_{idx}", serve_args=args
        ) as srv:
            resp = srv.client().post_chat(
                harness.make_chat_payload(
                    "List prime numbers below 20.",
                    enable_thinking=False,
                    max_tokens=24,
                )
            )
            assert resp.status == 200
        out, _ = srv.stop()
        done = harness.extract_done_events(harness.parse_jsonl_records(out))[-1]
        assert done["native_mtp_rounds"] >= 1
        assert done["native_mtp_proposed"] >= 1
        assert done["native_mtp_carried_anchors"] >= 1
        assert re.fullmatch(r"[0-9a-f]{64}", done["hc_exact_final_logit_sha256"])


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f19_05_entropy_gate_greedy_parity(isolated_state_root: Path) -> None:
    prompt = "Compute 6 * 7 and explain in 5 words."
    with harness.ServerProcess(isolated_state_root) as srv:
        c = srv.client()
        r1 = c.post_chat(
            harness.make_chat_payload(prompt, enable_thinking=False, temperature=0.0)
        )
        r2 = c.post_chat(
            harness.make_chat_payload(prompt, enable_thinking=False, temperature=0.0)
        )
        assert r1.json()["choices"][0]["message"] == r2.json()["choices"][0]["message"]
    out, _ = srv.stop()
    done = harness.extract_done_events(harness.parse_jsonl_records(out))
    assert (
        done[0]["hc_exact_final_logit_sha256"] == done[1]["hc_exact_final_logit_sha256"]
    )


# ---------------------------------------------------------------------------
# F20: Unified Default CLI & Worker Behavior (R1 + R4)
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f20_01_default_serve_enables_r1_park(isolated_state_root: Path) -> None:
    p_a = harness.make_long_repeated_prompt("defParkA")
    p_b = harness.make_long_repeated_prompt("defParkB")
    with harness.ServerProcess(isolated_state_root) as srv:
        c = srv.client()
        ra = c.post_chat(harness.make_chat_payload(p_a, enable_thinking=False, seed=1))
        assert (
            c.post_chat(
                harness.make_chat_payload(p_b, enable_thinking=False, seed=2)
            ).status
            == 200
        )
        ra2 = c.post_chat(
            harness.make_chat_payload(
                "Return to defParkA.",
                history=[
                    {"role": "user", "content": p_a},
                    {
                        "role": "assistant",
                        "content": ra.json()["choices"][0]["message"]["content"],
                    },
                ],
                enable_thinking=False,
                seed=1,
            )
        )
        assert ra2.status == 200
        assert (
            harness.validate_chat_completion_json(ra2.json(), min_cached_tokens=1) > 0
        )
    out, _ = srv.stop()
    done = harness.extract_done_events(harness.parse_jsonl_records(out))[-1]
    assert done["parked_lanes"]["budget_bytes"] > 0
    assert done["parked_lanes"]["restores"] >= 1


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f20_02_default_dflash2_enables_flush(isolated_state_root: Path) -> None:
    p_a = harness.make_long_repeated_prompt("defDfA")
    p_b = harness.make_long_repeated_prompt("defDfB")
    with harness.ServerProcess(
        isolated_state_root, serve_args=["--drafter", "dflash2"]
    ) as srv:
        c = srv.client()
        assert (
            c.post_chat(harness.make_chat_payload(p_a, enable_thinking=False)).status
            == 200
        )
        assert (
            c.post_chat(harness.make_chat_payload(p_b, enable_thinking=False)).status
            == 200
        )
    out, _ = srv.stop()
    done = harness.extract_done_events(harness.parse_jsonl_records(out))[-1]
    assert done["parked_lanes"]["parks"] >= 1
    assert done["parked_lanes"]["skips"] == 0


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f20_03_default_serve_r2_r4_active(isolated_state_root: Path) -> None:
    with harness.ServerProcess(isolated_state_root) as srv:
        resp = srv.client().post_chat(
            harness.make_chat_payload("Hello default.", enable_thinking=False)
        )
        assert resp.status == 200
    out, _ = srv.stop()
    recs = harness.parse_jsonl_records(out)
    ready = next(r for r in recs if r.get("event") == "ready")
    assert ready["native_ready"]["concurrent_dispatch_mode"] == "on"


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f20_04_python_rust_cli_default_sync(isolated_state_root: Path) -> None:
    harness.seed_isolated_state_root(isolated_state_root)
    rs_plan = harness.run_tokspur(
        ["plan"], state_root=isolated_state_root, cli_impl="rust"
    )
    py_plan = harness.run_tokspur(["plan"], cli_impl="python")
    assert rs_plan.returncode == 0 and py_plan.returncode == 0
    assert json.loads(rs_plan.stdout) == json.loads(py_plan.stdout)
    rs_help = harness.run_tokspur(
        ["serve", "--help"], state_root=isolated_state_root, cli_impl="rust"
    )
    py_help = harness.run_tokspur(["serve", "--help"], cli_impl="python")
    assert rs_help.returncode == 0 and py_help.returncode == 0
    assert "--drafter" in rs_help.stdout and "--drafter" in py_help.stdout


@pytest.mark.tier1
@pytest.mark.e2e_live
def test_t1_f20_05_doctor_plan_verify_isolated(isolated_state_root: Path) -> None:
    harness.seed_isolated_state_root(isolated_state_root)
    doc_res = harness.run_tokspur(["doctor"], state_root=isolated_state_root)
    assert doc_res.returncode == 0, doc_res.stderr + doc_res.stdout
    assert isinstance(json.loads(doc_res.stdout), dict)
    plan_res = harness.run_tokspur(["plan"], state_root=isolated_state_root)
    assert plan_res.returncode == 0, plan_res.stderr + plan_res.stdout
    assert json.loads(plan_res.stdout)["schema"] >= 1
    ver_res = harness.run_tokspur(["verify-build"], state_root=isolated_state_root)
    assert ver_res.returncode == 0, ver_res.stderr + ver_res.stdout
    assert isolated_state_root.exists()
