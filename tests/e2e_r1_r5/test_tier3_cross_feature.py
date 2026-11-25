# SPDX-License-Identifier: Apache-2.0
"""Tier 3 E2E tests: 20 pairwise cross-feature combination tests across R1-R5."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from . import harness


@pytest.mark.tier3
@pytest.mark.e2e_live
def test_t3_01_r1_r4_dflash2_parked_respelling(isolated_state_root: Path) -> None:
    p_a = harness.make_long_repeated_prompt("t3_df_bpe_a")
    p_b = harness.make_long_repeated_prompt("t3_df_bpe_b")
    with harness.ServerProcess(
        isolated_state_root, serve_args=["--speed-preset", "--drafter", "dflash2"]
    ) as srv:
        c = srv.client()
        ra1 = c.post_chat(
            harness.make_chat_payload(
                p_a, enable_thinking=False, seed=401, temperature=0.65, top_k=20
            )
        )
        assert c.post_chat(harness.make_chat_payload(p_b, seed=402)).status == 200
        ra2 = c.post_chat(
            harness.make_chat_payload(
                "Continue with BPE respell.",
                history=[
                    {"role": "user", "content": p_a},
                    {
                        "role": "assistant",
                        "content": ra1.json()["choices"][0]["message"]["content"],
                    },
                ],
                enable_thinking=False,
                seed=401,
                temperature=0.65,
                top_k=20,
            )
        )
        assert ra2.status == 200
        assert (
            harness.validate_chat_completion_json(ra2.json(), min_cached_tokens=1) > 0
        )
    out, _ = srv.stop()
    done = harness.extract_done_events(harness.parse_jsonl_records(out))[-1]
    assert done["parked_lanes"]["skips"] == 0
    assert done["parked_lanes"]["restores"] >= 1


@pytest.mark.tier3
@pytest.mark.e2e_live
def test_t3_02_r1_r4_entropy_gated_parked_turns(isolated_state_root: Path) -> None:
    p_a = harness.make_long_repeated_prompt("t3_ent_a")
    p_b = harness.make_long_repeated_prompt("t3_ent_b")
    with harness.ServerProcess(isolated_state_root) as srv:
        c = srv.client()
        ra1 = c.post_chat(
            harness.make_chat_payload(
                p_a,
                enable_thinking=True,
                seed=501,
                max_tokens=512,
                extra={"thinking_budget": 24},
            )
        )
        assert ra1.status == 200
        assert ra1.json()["choices"][0]["finish_reason"] == "stop"
        msg1 = ra1.json()["choices"][0]["message"]
        assert c.post_chat(harness.make_chat_payload(p_b, seed=502)).status == 200
        hist_msg = {"role": "assistant", "content": msg1["content"]}
        if "reasoning_content" in msg1:
            hist_msg["reasoning_content"] = msg1["reasoning_content"]
        ra2 = c.post_chat(
            harness.make_chat_payload(
                "Repeat exact header word.",
                history=[{"role": "user", "content": p_a}, hist_msg],
                enable_thinking=True,
                seed=501,
            )
        )
        assert ra2.status == 200
        assert (
            harness.validate_chat_completion_json(ra2.json(), min_cached_tokens=1) > 0
        )
    out, _ = srv.stop()
    records = harness.parse_jsonl_records(out)
    ready = next(r for r in records if r.get("event") == "ready")
    assert ready["native_ready"]["concurrent_dispatch_mode"] == "on"
    done = harness.extract_done_events(records)[-1]
    assert done["parked_lanes"]["restores"] >= 1


@pytest.mark.tier3
@pytest.mark.e2e_live
def test_t3_03_r2_r3_level_scheduled_planar_moe() -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    pack_bin = harness.require_artifact(
        harness.find_native_test_bin("test-planar_pack_v3"), "test-planar_pack_v3"
    )
    graph_bin = harness.require_artifact(
        harness.find_native_test_bin("test-graph"), "test-graph"
    )
    assert harness.run_cmd([str(graph_bin), str(metallib)]).returncode == 0
    assert harness.run_cmd([str(pack_bin), str(metallib)]).returncode == 0


@pytest.mark.tier3
@pytest.mark.e2e_live
def test_t3_04_r2_r3_error_flag_end_of_batch_nan() -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    bin_path = harness.require_artifact(
        harness.find_native_test_bin("test-sampled_stack"), "test-sampled_stack"
    )
    res = harness.run_cmd([str(bin_path), str(metallib), "--concurrent"])
    assert res.returncode == 0, res.stderr + res.stdout
    assert "nonfinite-output-refusals=8" in res.stdout
    sync_res = harness.run_cargo_test_checked(
        ["-p", "tokspur-dispatch-graph", "--test", "kernels"], min_passed=1
    )
    assert sync_res.returncode == 0


@pytest.mark.tier3
@pytest.mark.e2e_live
def test_t3_05_r3_r4_dflash2_stripped_sqrt_ckpt(isolated_state_root: Path) -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    stage_bin = harness.require_artifact(
        harness.find_native_test_bin("test-dflash2_context_stage"),
        "test-dflash2_context_stage",
    )
    assert harness.run_cmd([str(stage_bin), str(metallib)]).returncode == 0
    prefix = harness.make_long_repeated_prompt("t3_df_sqrt", approx_lines=2500)
    with harness.ServerProcess(
        isolated_state_root,
        serve_args=[
            "--drafter",
            "dflash2",
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
                harness.make_chat_payload(prefix + "\nA", enable_thinking=False),
                timeout=900.0,
            ).status
            == 200
        )
        r2 = c.post_chat(
            harness.make_chat_payload(prefix + "\nB", enable_thinking=False),
            timeout=900.0,
        )
        assert r2.status == 200
        assert r2.json()["usage"]["prompt_tokens"] > 65536
        assert harness.validate_chat_completion_json(r2.json(), min_cached_tokens=1) > 0
    out, _ = srv.stop()
    done = harness.extract_done_events(harness.parse_jsonl_records(out))[-1]
    assert done["checkpoint_restored_tokens"] > 0


@pytest.mark.tier3
@pytest.mark.e2e_live
def test_t3_06_r3_r4_anchor_entropy_topk_finite(isolated_state_root: Path) -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    topk_bin = harness.require_artifact(
        harness.find_native_test_bin("test-logit_topk"), "test-logit_topk"
    )
    assert harness.run_cmd([str(topk_bin), str(metallib)]).returncode == 0
    with harness.ServerProcess(isolated_state_root) as srv:
        r = srv.client().post_chat(
            harness.make_chat_payload("List 5 integers.", enable_thinking=False)
        )
        assert r.status == 200
        harness.validate_chat_completion_json(r.json())


@pytest.mark.tier3
@pytest.mark.e2e_live
def test_t3_07_r1_r5_rust_diff_under_cache(tmp_path: Path) -> None:
    shim, log1 = harness.create_cargo_logging_shim(tmp_path)
    cache_dir = tmp_path / "rcache"
    r_chk = harness.run_make_native(
        ["rust-check", f"RUST_RECEIPT_CACHE_DIR={cache_dir}", f"CARGO={shim}"],
        makefile="rust.mk",
        env={"TOKSPUR_E2E_CARGO_LOG": str(log1)},
    )
    assert r_chk.returncode == 0, r_chk.stdout + "\n" + r_chk.stderr
    log2 = tmp_path / "warm.log"
    r_warm = harness.run_make_native(
        ["rust-check", f"RUST_RECEIPT_CACHE_DIR={cache_dir}", f"CARGO={shim}"],
        makefile="rust.mk",
        env={"TOKSPUR_E2E_CARGO_LOG": str(log2)},
    )
    assert r_warm.returncode == 0, r_warm.stdout + "\n" + r_warm.stderr
    assert harness.read_cached_cargo_calls(log2) == []
    r_diff = harness.run_make_native(["rust-diff"], makefile="rust.mk")
    assert r_diff.returncode == 0, r_diff.stdout + "\n" + r_diff.stderr


@pytest.mark.tier3
@pytest.mark.e2e_live
def test_t3_08_r1_r5_parked_ab_holdout_futility(
    isolated_state_root: Path, synthetic_r5_factory, tmp_path: Path
) -> None:
    bundle = synthetic_r5_factory("t3_ab_holdout")
    h_res = harness.run_r5_holdout_eval(
        tmp_path / "holdout",
        bundle["manifest_path"].parent,
        "holdout_0",
        bundle["ledger_path"],
    )
    assert h_res.returncode == 0
    blocks = harness.build_paired_block_series("winner_2block")
    f_res = harness.run_r5_futility_eval(
        tmp_path / "futility",
        {"mode": "futility", "alpha": 0.05, "blocks": blocks},
    )
    assert (
        f_res.returncode == 0 and json.loads(f_res.stdout)["stop_for_futility"] is False
    )
    p_a = harness.make_long_repeated_prompt("t3_ab_a")
    with harness.ServerProcess(isolated_state_root) as srv:
        r = srv.client().post_chat(
            harness.make_chat_payload(p_a, enable_thinking=False)
        )
        assert r.status == 200


@pytest.mark.tier3
@pytest.mark.e2e_live
def test_t3_09_r2_r4_concurrent_sqrt_replay(
    isolated_state_root: Path, tmp_path: Path
) -> None:
    ref_p, cand_p = harness.build_dag_pair_traces(tmp_path, depth=3)
    assert (
        harness.run_dispatch_graph(["check", str(ref_p), str(cand_p)]).returncode == 0
    )
    prefix = harness.make_long_repeated_prompt("t3_conc_sqrt", approx_lines=2500)
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
                harness.make_chat_payload(prefix + "\nPath1", enable_thinking=False),
                timeout=900.0,
            ).status
            == 200
        )
        r2 = c.post_chat(
            harness.make_chat_payload(prefix + "\nPath2", enable_thinking=False),
            timeout=900.0,
        )
        assert r2.status == 200
        assert r2.json()["usage"]["prompt_tokens"] > 65536
        assert harness.validate_chat_completion_json(r2.json(), min_cached_tokens=1) > 0
    out, _ = srv.stop()
    done = harness.extract_done_events(harness.parse_jsonl_records(out))[-1]
    assert done["checkpoint_restored_tokens"] > 0


@pytest.mark.tier3
@pytest.mark.e2e_live
def test_t3_10_r2_r4_decoupled_flag_entropy_draft(
    isolated_state_root: Path,
) -> None:
    assert (
        harness.run_cargo_test_checked(
            ["-p", "tokspur-dispatch-graph", "--test", "kernels"], min_passed=1
        ).returncode
        == 0
    )
    with harness.ServerProcess(
        isolated_state_root, serve_args=["--drafter", "dflash2"]
    ) as srv:
        r = srv.client().post_chat(
            harness.make_chat_payload("Summarize binary search.", enable_thinking=False)
        )
        assert r.status == 200


@pytest.mark.tier3
@pytest.mark.e2e_live
def test_t3_11_r4_r5_dflash2_sweep_futility_stop(
    isolated_state_root: Path, tmp_path: Path
) -> None:
    blocks = harness.build_paired_block_series("loser_2block")
    f_res = harness.run_r5_futility_eval(
        tmp_path, {"mode": "futility", "alpha": 0.05, "blocks": blocks}
    )
    assert f_res.returncode == 0
    assert json.loads(f_res.stdout)["stop_for_futility"] is True
    with harness.ServerProcess(
        isolated_state_root, serve_args=["--drafter", "dflash2"]
    ) as srv:
        assert srv.client().get_health().status == 200


@pytest.mark.tier3
@pytest.mark.e2e_live
def test_t3_12_r4_r5_long_ctx_holdout_manifest(
    isolated_state_root: Path, synthetic_r5_factory, tmp_path: Path
) -> None:
    bundle = synthetic_r5_factory("t3_long_ctx")
    h_res = harness.run_r5_holdout_eval(
        tmp_path,
        bundle["manifest_path"].parent,
        "holdout_0",
        bundle["ledger_path"],
    )
    assert h_res.returncode == 0
    with harness.ServerProcess(
        isolated_state_root, serve_args=["--context-length", "131072"]
    ) as srv:
        r = srv.client().post_chat(
            harness.make_chat_payload("Long ctx ping.", enable_thinking=False)
        )
        assert r.status == 200


@pytest.mark.tier3
@pytest.mark.e2e_live
def test_t3_13_r1_r2_concurrent_parked_mtp3(isolated_state_root: Path) -> None:
    p_a = harness.make_long_repeated_prompt("t3_conc_mtp_a")
    p_b = harness.make_long_repeated_prompt("t3_conc_mtp_b")
    with harness.ServerProcess(isolated_state_root) as srv:
        c = srv.client()
        r1 = c.post_chat(harness.make_chat_payload(p_a, enable_thinking=False, seed=11))
        assert c.post_chat(harness.make_chat_payload(p_b, seed=12)).status == 200
        r3 = c.post_chat(
            harness.make_chat_payload(
                "Return conc MTP3.",
                history=[
                    {"role": "user", "content": p_a},
                    {
                        "role": "assistant",
                        "content": r1.json()["choices"][0]["message"]["content"],
                    },
                ],
                enable_thinking=False,
                seed=11,
            )
        )
        assert r3.status == 200
        assert harness.validate_chat_completion_json(r3.json(), min_cached_tokens=1) > 0
    out, _ = srv.stop()
    ready = next(
        r for r in harness.parse_jsonl_records(out) if r.get("event") == "ready"
    )
    assert ready["native_ready"]["concurrent_dispatch_mode"] == "on"


@pytest.mark.tier3
@pytest.mark.e2e_live
def test_t3_14_r1_r2_cpp_parity_dispatch_graph(tmp_path: Path) -> None:
    assert harness.run_make_native(["rust-diff"], makefile="rust.mk").returncode == 0
    ref_p, _ = harness.build_dag_pair_traces(tmp_path, depth=3)
    ctrl = harness.run_dispatch_graph(["controls", str(ref_p)])
    assert ctrl.returncode == 0 and json.loads(ctrl.stdout)["held"] is True
    assert (
        harness.run_cargo_test_checked(
            ["-p", "tokspur-dispatch-graph", "--test", "kernels"], min_passed=1
        ).returncode
        == 0
    )


@pytest.mark.tier3
@pytest.mark.e2e_live
def test_t3_15_r1_r3_certified_pack_echoed_reason(
    isolated_state_root: Path,
) -> None:
    metallib = harness.require_artifact(harness.find_metallib(), "metallib")
    pack_bin = harness.require_artifact(
        harness.find_native_test_bin("test-planar_pack_v3"), "test-planar_pack_v3"
    )
    assert harness.run_cmd([str(pack_bin), str(metallib)]).returncode == 0
    p_a = harness.make_long_repeated_prompt("t3_pack_echo_a")
    p_b = harness.make_long_repeated_prompt("t3_pack_echo_b")
    with harness.ServerProcess(isolated_state_root) as srv:
        c = srv.client()
        r1 = c.post_chat(
            harness.make_chat_payload(
                p_a,
                enable_thinking=True,
                seed=7,
                max_tokens=512,
                extra={"thinking_budget": 16},
            )
        )
        assert r1.status == 200
        assert r1.json()["choices"][0]["finish_reason"] == "stop"
        msg1 = r1.json()["choices"][0]["message"]
        assert c.post_chat(harness.make_chat_payload(p_b)).status == 200
        hist = {"role": "assistant", "content": msg1["content"]}
        if "reasoning_content" in msg1:
            hist["reasoning_content"] = msg1["reasoning_content"]
        r3 = c.post_chat(
            harness.make_chat_payload(
                "Echo check.",
                history=[{"role": "user", "content": p_a}, hist],
                enable_thinking=True,
                seed=7,
            )
        )
        assert r3.status == 200
        assert harness.validate_chat_completion_json(r3.json(), min_cached_tokens=1) > 0


@pytest.mark.tier3
@pytest.mark.e2e_live
def test_t3_16_r1_r3_respelling_shape_invariants(
    isolated_state_root: Path,
) -> None:
    assert harness.run_make_native(["test"]).returncode == 0
    p_a = harness.make_long_repeated_prompt("t3_respell_shape_a")
    p_b = harness.make_long_repeated_prompt("t3_respell_shape_b")
    with harness.ServerProcess(isolated_state_root) as srv:
        c = srv.client()
        r1 = c.post_chat(
            harness.make_chat_payload(
                p_a, enable_thinking=False, seed=62, temperature=0.6, top_k=20
            )
        )
        assert c.post_chat(harness.make_chat_payload(p_b, seed=63)).status == 200
        r3 = c.post_chat(
            harness.make_chat_payload(
                "Respell shape.",
                history=[
                    {"role": "user", "content": p_a},
                    {
                        "role": "assistant",
                        "content": r1.json()["choices"][0]["message"]["content"],
                    },
                ],
                enable_thinking=False,
                seed=62,
                temperature=0.6,
                top_k=20,
            )
        )
        assert r3.status == 200
        assert harness.validate_chat_completion_json(r3.json(), min_cached_tokens=1) > 0


@pytest.mark.tier3
@pytest.mark.e2e_live
def test_t3_17_r2_r5_cached_dispatch_graph_ci(tmp_path: Path) -> None:
    assert harness.run_make_native(["rust-check"], makefile="rust.mk").returncode == 0
    ref_p, cand_p = harness.build_dag_pair_traces(tmp_path, depth=3)
    assert (
        harness.run_dispatch_graph(["check", str(ref_p), str(cand_p)]).returncode == 0
    )
    assert harness.run_dispatch_graph(["controls", str(ref_p)]).returncode == 0


@pytest.mark.tier3
@pytest.mark.e2e_live
def test_t3_18_r2_r5_concurrent_barrier_futility(
    isolated_state_root: Path, tmp_path: Path
) -> None:
    payload = harness.make_chat_payload(
        "Say ready.", enable_thinking=False, temperature=0.0, seed=12, max_tokens=16
    )
    with harness.ServerProcess(
        isolated_state_root / "serial",
        serve_args=["--no-concurrent-dispatch"],
        extra_env={
            "TOKSPUR_DUMP_VERIFY_PROFILE": "1",
            "TOKSPUR_ALLOW_KNOWN_PROFILE": "1",
        },
    ) as srv_s:
        assert srv_s.client().post_chat(payload).status == 200
    out_s, err_s = srv_s.stop()
    assert len(harness.parse_profile_lines(err_s, "VERIFY_PROFILE")) >= 1
    done_s = harness.extract_done_events(harness.parse_jsonl_records(out_s))[-1]
    assert (
        done_s["gpu_lifetime_counters"]["barriers"]
        == done_s["gpu_lifetime_counters"]["dispatches"]
    )

    with harness.ServerProcess(isolated_state_root / "conc") as srv_c:
        assert srv_c.client().post_chat(payload).status == 200
    out_c, _ = srv_c.stop()
    records_c = harness.parse_jsonl_records(out_c)
    ready_c = next(r for r in records_c if r.get("event") == "ready")
    assert ready_c["native_ready"]["concurrent_dispatch_mode"] == "on"
    done_c = harness.extract_done_events(records_c)[-1]
    assert done_c["concurrent_dispatch_mode"] == "on"
    assert (
        done_c["gpu_lifetime_counters"]["barriers"]
        < done_c["gpu_lifetime_counters"]["dispatches"]
    )
    assert (
        done_c["gpu_lifetime_counters"]["barriers"]
        < done_s["gpu_lifetime_counters"]["barriers"]
    )
    assert (
        done_c["hc_exact_final_logit_sha256"] == done_s["hc_exact_final_logit_sha256"]
    )

    blocks = harness.build_paired_block_series("loser_2block")
    f_res = harness.run_r5_futility_eval(
        tmp_path, {"mode": "futility", "alpha": 0.05, "blocks": blocks}
    )
    assert (
        f_res.returncode == 0 and json.loads(f_res.stdout)["stop_for_futility"] is True
    )


@pytest.mark.tier3
@pytest.mark.e2e_live
def test_t3_19_r3_r5_pack_v3_holdout_verify(
    tmp_path: Path, synthetic_r5_factory
) -> None:
    bundle = synthetic_r5_factory("t3_pack_holdout")
    h_res = harness.run_r5_holdout_eval(
        tmp_path / "holdout",
        bundle["manifest_path"].parent,
        "holdout_0",
        bundle["ledger_path"],
    )
    assert h_res.returncode == 0
    bdir = tmp_path / "screen1"
    mk = harness.run_make_native(
        [
            "M5_HEADLINE_SCREEN=1",
            f"BUILD={bdir}",
            f"{bdir}/tokspur-owned.metallib",
            f"{bdir}/test-planar_pack_v3",
        ]
    )
    assert mk.returncode == 0
    run_res = harness.run_cmd(
        [str(bdir / "test-planar_pack_v3"), str(bdir / "tokspur-owned.metallib")]
    )
    assert run_res.returncode == 0


@pytest.mark.tier3
@pytest.mark.e2e_live
def test_t3_20_r3_r5_cached_metal_flags_check() -> None:
    assert harness.run_make_native(["rust-check"], makefile="rust.mk").returncode == 0
    dry = harness.query_make_dry_run_flags("tokspur-owned.metallib")
    harness.assert_metal_fp_policy(dry)
    assert harness.run_script("check_sources.py", []).returncode == 0
