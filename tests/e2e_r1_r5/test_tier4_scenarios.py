# SPDX-License-Identifier: Apache-2.0
"""Tier 4 E2E tests: 10 realistic end-to-end application scenarios."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from . import harness


@pytest.mark.tier4
@pytest.mark.e2e_live
def test_t4_01_multiturn_parked_chats_mtp3(isolated_state_root: Path) -> None:
    """Scenario 1: Multi-turn interleaved parked chats (MTP3, thinking, BPE, seed)."""
    p_j = harness.make_long_repeated_prompt("J1_scenario")
    p_r = harness.make_long_repeated_prompt("R1_scenario")
    with harness.ServerProcess(
        isolated_state_root,
        serve_args=["--speed-preset", "--drafter", "mtp", "--mtp-depth", "3"],
    ) as srv:
        c = srv.client()
        j1 = c.post_chat(
            harness.make_chat_payload(
                p_j,
                enable_thinking=True,
                seed=24319,
                temperature=0.65,
                top_k=20,
                top_p=0.90,
                max_tokens=512,
                extra={"thinking_budget": 32},
            )
        )
        r1 = c.post_chat(
            harness.make_chat_payload(p_r, enable_thinking=False, seed=9983)
        )
        assert j1.status == 200 and r1.status == 200
        assert j1.json()["choices"][0]["finish_reason"] == "stop"
        j1_msg = j1.json()["choices"][0]["message"]
        j_hist_msg: dict[str, object] = {
            "role": "assistant",
            "content": j1_msg["content"],
        }
        if "reasoning_content" in j1_msg:
            j_hist_msg["reasoning_content"] = j1_msg["reasoning_content"]
        j2 = c.post_chat(
            harness.make_chat_payload(
                "Return to J2 with echoed reasoning.",
                history=[{"role": "user", "content": p_j}, j_hist_msg],
                enable_thinking=True,
                seed=24319,
                temperature=0.65,
                top_k=20,
                top_p=0.90,
                extra={"thinking_budget": 32},
            )
        )
        r2 = c.post_chat(
            harness.make_chat_payload(
                "Return to R2 with enable_thinking=false.",
                history=[
                    {"role": "user", "content": p_r},
                    {
                        "role": "assistant",
                        "content": r1.json()["choices"][0]["message"]["content"],
                    },
                ],
                enable_thinking=False,
                seed=9983,
            )
        )
        assert j2.status == 200 and r2.status == 200
        assert harness.validate_chat_completion_json(j2.json(), min_cached_tokens=1) > 0
        assert harness.validate_chat_completion_json(r2.json(), min_cached_tokens=1) > 0
    out, _ = srv.stop()
    done = harness.extract_done_events(harness.parse_jsonl_records(out))[-1]
    assert done["parked_lanes"]["parks"] >= 2
    assert done["parked_lanes"]["restores"] >= 2
    assert done["parked_lanes"]["restored_tokens"] > 0


@pytest.mark.tier4
@pytest.mark.e2e_live
def test_t4_02_multiturn_parked_chats_dflash2(isolated_state_root: Path) -> None:
    """Scenario 2: DFlash2 deferred_context=on lazy flush & RAM parked restores."""
    p_a = harness.make_long_repeated_prompt("df_s2_A")
    p_b = harness.make_long_repeated_prompt("df_s2_B")
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
        a1 = c.post_chat(harness.make_chat_payload(p_a, enable_thinking=False, seed=11))
        b1 = c.post_chat(harness.make_chat_payload(p_b, enable_thinking=False, seed=22))
        assert a1.status == 200 and b1.status == 200
        a2 = c.post_chat(
            harness.make_chat_payload(
                "Continue A after DFlash2 park.",
                history=[
                    {"role": "user", "content": p_a},
                    {
                        "role": "assistant",
                        "content": a1.json()["choices"][0]["message"]["content"],
                    },
                ],
                enable_thinking=False,
                seed=11,
            )
        )
        b2 = c.post_chat(
            harness.make_chat_payload(
                "Continue B after DFlash2 park.",
                history=[
                    {"role": "user", "content": p_b},
                    {
                        "role": "assistant",
                        "content": b1.json()["choices"][0]["message"]["content"],
                    },
                ],
                enable_thinking=False,
                seed=22,
            )
        )
        assert a2.status == 200 and b2.status == 200
        assert harness.validate_chat_completion_json(a2.json(), min_cached_tokens=1) > 0
        assert harness.validate_chat_completion_json(b2.json(), min_cached_tokens=1) > 0
    out, _ = srv.stop()
    done = harness.extract_done_events(harness.parse_jsonl_records(out))[-1]
    assert done["native_dflash2_defer_context"] is True
    assert done["parked_lanes"]["skips"] == 0
    assert done["parked_lanes"]["restores"] >= 2


@pytest.mark.tier4
@pytest.mark.e2e_live
def test_t4_03_long_context_sqrt_checkpoint_replay(
    isolated_state_root: Path,
) -> None:
    """Scenario 3: 64K-128K long-context branch & SSD sqrt(N) replay parity."""
    prefix = harness.make_long_repeated_prompt("sqrt64k", approx_lines=2500)
    with harness.ServerProcess(
        isolated_state_root / "baseline",
        serve_args=[
            "--context-length",
            "131072",
            "--prefill-dense",
            "f32-staged",
            "--prefill-grouped",
            "f32",
        ],
    ) as srv_base:
        rb = srv_base.client().post_chat(
            harness.make_chat_payload(
                prefix + "\nFinal target branch.",
                enable_thinking=False,
                seed=5,
                max_tokens=16,
            ),
            timeout=1200.0,
        )
        assert rb.status == 200
        assert rb.json()["usage"]["prompt_tokens"] > 65536
    out_b, _ = srv_base.stop()
    done_base = harness.extract_done_events(harness.parse_jsonl_records(out_b))[-1]
    sha_base = done_base["hc_exact_final_logit_sha256"]

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
    ) as srv_ck:
        ck = srv_ck.client()
        assert (
            ck.post_chat(
                harness.make_chat_payload(
                    prefix + "\nFirst explored branch.",
                    enable_thinking=False,
                    seed=5,
                    max_tokens=16,
                ),
                timeout=1200.0,
            ).status
            == 200
        )
        rc = ck.post_chat(
            harness.make_chat_payload(
                prefix + "\nFinal target branch.",
                enable_thinking=False,
                seed=5,
                max_tokens=16,
            ),
            timeout=1200.0,
        )
        assert rc.status == 200
        assert harness.validate_chat_completion_json(rc.json(), min_cached_tokens=1) > 0
    out_ck, _ = srv_ck.stop()
    done_ck = harness.extract_done_events(harness.parse_jsonl_records(out_ck))[-1]
    assert done_ck["checkpoint_ssd_enabled"] is True
    assert done_ck["checkpoint_ssd_spills"] >= 1
    assert done_ck["checkpoint_ssd_occupied_bytes"] > 0
    assert done_ck["checkpoint_ssd_restored"] is True
    assert done_ck["checkpoint_restored_tokens"] > 0
    assert done_ck["hc_exact_final_logit_sha256"] == sha_base
    assert done_ck["generated_ids"] == done_base["generated_ids"]
    assert (
        rc.json()["choices"][0]["message"]["content"]
        == rb.json()["choices"][0]["message"]["content"]
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
    ) as srv_base_nax:
        rb_nax = srv_base_nax.client().post_chat(
            harness.make_chat_payload(
                prefix + "\nFinal target branch.",
                enable_thinking=False,
                seed=5,
                max_tokens=16,
            ),
            timeout=1200.0,
        )
        assert rb_nax.status == 200
        assert rb_nax.json()["usage"]["prompt_tokens"] > 65536
    out_b_nax, _ = srv_base_nax.stop()
    done_base_nax = harness.extract_done_events(harness.parse_jsonl_records(out_b_nax))[
        -1
    ]

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
    ) as srv_ck_nax:
        ck_nax = srv_ck_nax.client()
        assert (
            ck_nax.post_chat(
                harness.make_chat_payload(
                    prefix + "\nFirst explored branch.",
                    enable_thinking=False,
                    seed=5,
                    max_tokens=16,
                ),
                timeout=1200.0,
            ).status
            == 200
        )
        rc_nax = ck_nax.post_chat(
            harness.make_chat_payload(
                prefix + "\nFinal target branch.",
                enable_thinking=False,
                seed=5,
                max_tokens=16,
            ),
            timeout=1200.0,
        )
        assert rc_nax.status == 200
        assert (
            harness.validate_chat_completion_json(rc_nax.json(), min_cached_tokens=1)
            > 0
        )
    out_ck_nax, _ = srv_ck_nax.stop()
    done_ck_nax = harness.extract_done_events(harness.parse_jsonl_records(out_ck_nax))[
        -1
    ]
    assert done_ck_nax["checkpoint_ssd_enabled"] is True
    assert done_ck_nax["checkpoint_ssd_spills"] >= 1
    assert done_ck_nax["checkpoint_ssd_occupied_bytes"] > 0
    assert done_ck_nax["checkpoint_ssd_restored"] is True
    assert done_ck_nax["checkpoint_restored_tokens"] > 0
    assert done_ck_nax["generated_ids"] == done_base_nax["generated_ids"]
    assert (
        rc_nax.json()["choices"][0]["message"]["content"]
        == rb_nax.json()["choices"][0]["message"]["content"]
    )


@pytest.mark.tier4
@pytest.mark.e2e_live
def test_t4_04_m5_decode_profile_dispatch_graph(
    isolated_state_root: Path, tmp_path: Path
) -> None:
    """Scenario 4: Decode profile + dispatch-graph check under Phase B & error_flag."""
    assert (
        harness.run_cargo_test_checked(
            ["-p", "tokspur-dispatch-graph", "--test", "kernels"], min_passed=1
        ).returncode
        == 0
    )
    ref_p, cand_p = harness.build_dag_pair_traces(tmp_path, depth=4)
    assert (
        harness.run_dispatch_graph(["check", str(ref_p), str(cand_p)]).returncode == 0
    )
    assert harness.run_dispatch_graph(["controls", str(ref_p)]).returncode == 0
    payload = harness.make_chat_payload(
        "Implement binary search in Rust.",
        enable_thinking=False,
        temperature=0.0,
        seed=12,
        max_tokens=48,
    )
    with harness.ServerProcess(isolated_state_root / "conc") as srv_c:
        resp_c = srv_c.client().post_chat(payload)
        assert resp_c.status == 200
    out_c, _ = srv_c.stop()
    records_c = harness.parse_jsonl_records(out_c)
    ready_c = next(r for r in records_c if r.get("event") == "ready")
    assert ready_c["native_ready"]["concurrent_dispatch_mode"] == "on"
    done_c = harness.extract_done_events(records_c)[-1]
    assert done_c["concurrent_dispatch_mode"] == "on"
    assert done_c["generation_batch_gpu_ms"] > 0.0
    assert done_c["concurrent_dispatch_barriers"] > 0
    assert (
        done_c["gpu_lifetime_counters"]["barriers"]
        < done_c["gpu_lifetime_counters"]["dispatches"]
    )

    with harness.ServerProcess(
        isolated_state_root / "prof",
        serve_args=["--no-concurrent-dispatch"],
        extra_env={
            "TOKSPUR_DUMP_VERIFY_PROFILE": "1",
            "TOKSPUR_ALLOW_KNOWN_PROFILE": "1",
        },
    ) as srv_p:
        resp_p = srv_p.client().post_chat(payload)
        assert resp_p.status == 200
    out_p, err_p = srv_p.stop()
    assert len(harness.parse_profile_lines(err_p, "VERIFY_PROFILE")) >= 1
    done_p = harness.extract_done_events(harness.parse_jsonl_records(out_p))[-1]
    assert (
        done_p["gpu_lifetime_counters"]["barriers"]
        == done_p["gpu_lifetime_counters"]["dispatches"]
    )
    assert (
        done_c["gpu_lifetime_counters"]["barriers"]
        < done_p["gpu_lifetime_counters"]["barriers"]
    )
    assert (
        done_c["hc_exact_final_logit_sha256"] == done_p["hc_exact_final_logit_sha256"]
    )


@pytest.mark.tier4
@pytest.mark.e2e_live
def test_t4_05_instantaneous_anchor_entropy_gating(
    isolated_state_root: Path,
) -> None:
    """Scenario 5: Anchor-entropy width gating across low- vs high-entropy prompts."""
    low_prompt = ("alpha beta gamma delta epsilon\n" * 20) + "Continue pattern:"
    high_prompt = "Write an original surreal haiku about tungsten and snow."
    drafter_args = (
        ["--drafter", "mtp", "--mtp-depth", "3"],
        ["--drafter", "dflash2"],
    )
    for idx, args in enumerate(drafter_args):
        with harness.ServerProcess(
            isolated_state_root / f"ent_{idx}",
            serve_args=[*args],
        ) as srv:
            c = srv.client()
            r_lo = c.post_chat(
                harness.make_chat_payload(
                    low_prompt, enable_thinking=False, max_tokens=24
                )
            )
            assert r_lo.status == 200
            lo_done = harness.wait_for_done_events(srv.stdout_log_path, min_count=1)[-1]
            r_hi = c.post_chat(
                harness.make_chat_payload(
                    high_prompt,
                    enable_thinking=False,
                    temperature=0.8,
                    top_k=20,
                    seed=17,
                    max_tokens=24,
                )
            )
            assert r_hi.status == 200
        out, _ = srv.stop()
        records = harness.parse_jsonl_records(out)
        ready = next(r for r in records if r.get("event") == "ready")
        assert ready["native_ready"]["concurrent_dispatch_mode"] == "on"
        hi_done = harness.extract_done_events(records)[-1]
        lo_rounds = lo_done["native_mtp_rounds"]
        lo_proposed = lo_done["native_mtp_proposed"]
        hi_rounds = hi_done["native_mtp_rounds"] - lo_rounds
        hi_proposed = hi_done["native_mtp_proposed"] - lo_proposed
        assert lo_done["native_mtp_carried_anchors"] >= 1
        assert lo_rounds >= 1 and hi_rounds >= 1
        assert lo_proposed >= 1 and hi_proposed >= 1
        assert (hi_proposed / hi_rounds) < (lo_proposed / lo_rounds)
        assert hi_done["generation_batch_gpu_ms"] > 0.0


@pytest.mark.tier4
@pytest.mark.e2e_live
def test_t4_06_r5_optimization_turnaround_pipeline(
    synthetic_r5_factory, tmp_path: Path
) -> None:
    """Scenario 6: R5 cached 5-way rebuild -> rust-diff -> futility -> holdout."""
    shim, log1 = harness.create_cargo_logging_shim(tmp_path)
    cache_dir = tmp_path / "pipe_cache"
    assert (
        harness.run_make_native(
            ["rust-check", f"RUST_RECEIPT_CACHE_DIR={cache_dir}", f"CARGO={shim}"],
            makefile="rust.mk",
            env={"TOKSPUR_E2E_CARGO_LOG": str(log1)},
        ).returncode
        == 0
    )
    log2 = tmp_path / "warm.log"
    assert (
        harness.run_make_native(
            ["rust-check", f"RUST_RECEIPT_CACHE_DIR={cache_dir}", f"CARGO={shim}"],
            makefile="rust.mk",
            env={"TOKSPUR_E2E_CARGO_LOG": str(log2)},
        ).returncode
        == 0
    )
    assert harness.read_cached_cargo_calls(log2) == []
    assert harness.run_make_native(["rust-diff"], makefile="rust.mk").returncode == 0
    blocks = harness.build_paired_block_series("loser_2block")
    f_res = harness.run_r5_futility_eval(
        tmp_path / "futility", {"mode": "futility", "alpha": 0.05, "blocks": blocks}
    )
    assert (
        f_res.returncode == 0 and json.loads(f_res.stdout)["stop_for_futility"] is True
    )
    bundle = synthetic_r5_factory("t4_r5_pipe")
    h_res = harness.run_r5_holdout_eval(
        tmp_path / "holdout",
        bundle["manifest_path"].parent,
        "holdout_0",
        bundle["ledger_path"],
    )
    assert h_res.returncode == 0


@pytest.mark.tier4
@pytest.mark.e2e_live
def test_t4_07_pack_v3_dual_screen_and_nan_catch(tmp_path: Path) -> None:
    """Scenario 7: Pack v3 dual M5_HEADLINE_SCREEN + EOB NaN/Inf under fp-contract=off."""
    dry = harness.query_make_dry_run_flags("tokspur-owned.metallib")
    harness.assert_metal_fp_policy(dry)
    for screen in ("0", "1"):
        bdir = tmp_path / f"t4_s_{screen}"
        mk = harness.run_make_native(
            [
                f"M5_HEADLINE_SCREEN={screen}",
                f"BUILD={bdir}",
                f"{bdir}/tokspur-owned.metallib",
                f"{bdir}/test-planar_pack_v3",
                f"{bdir}/test-sampled_stack",
            ]
        )
        assert mk.returncode == 0
        assert (
            harness.run_cmd(
                [
                    str(bdir / "test-planar_pack_v3"),
                    str(bdir / "tokspur-owned.metallib"),
                ]
            ).returncode
            == 0
        )
        assert (
            harness.run_cmd(
                [
                    str(bdir / "test-sampled_stack"),
                    str(bdir / "tokspur-owned.metallib"),
                    "--concurrent",
                ]
            ).returncode
            == 0
        )


@pytest.mark.tier4
@pytest.mark.e2e_live
def test_t4_08_concurrent_http_sse_streaming(
    isolated_state_root: Path,
) -> None:
    """Scenario 8: SSE streaming chunks, include_usage, cached_tokens, 403 Origin."""
    p_shared = harness.make_long_repeated_prompt("t4_sse")
    with harness.ServerProcess(isolated_state_root) as srv:
        c = srv.client()
        for idx in (1, 2):
            assert (
                c.post_chat(
                    harness.make_chat_payload(
                        p_shared + f"\nWarmup {idx}.", enable_thinking=False
                    )
                ).status
                == 200
            )
        sse_resp = c.post_chat(
            harness.make_chat_payload(
                p_shared + "\nStream me.",
                enable_thinking=False,
                stream=True,
                include_usage=True,
                max_tokens=24,
            )
        )
        assert sse_resp.status == 200
        chunks = c.parse_sse_chunks(sse_resp.body)
        assert chunks[-1] == "[DONE]"
        usage_chunks = [
            ch for ch in chunks[:-1] if isinstance(ch, dict) and ch.get("usage")
        ]
        assert len(usage_chunks) == 1
        assert usage_chunks[0]["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
        origin_refused = c.post_chat(
            harness.make_chat_payload("Blocked"),
            headers={"Origin": "https://untrusted.example"},
        )
        assert origin_refused.status == 403


@pytest.mark.tier4
@pytest.mark.e2e_live
def test_t4_09_zero_opt_in_default_cli_launch(isolated_state_root: Path) -> None:
    """Scenario 9: Default tokspur serve activates R1+R2+R3+R4 with zero manual flags."""
    p_a = harness.make_long_repeated_prompt("t4_def_A")
    p_b = harness.make_long_repeated_prompt("t4_def_B")
    for idx, args in enumerate(([], ["--drafter", "dflash2"])):
        with harness.ServerProcess(
            isolated_state_root / f"m_{idx}", serve_args=args
        ) as srv:
            c = srv.client()
            ra = c.post_chat(
                harness.make_chat_payload(p_a, enable_thinking=False, seed=9)
            )
            assert c.post_chat(harness.make_chat_payload(p_b, seed=10)).status == 200
            ra2 = c.post_chat(
                harness.make_chat_payload(
                    "Return default.",
                    history=[
                        {"role": "user", "content": p_a},
                        {
                            "role": "assistant",
                            "content": ra.json()["choices"][0]["message"]["content"],
                        },
                    ],
                    enable_thinking=False,
                    seed=9,
                )
            )
            assert ra2.status == 200
            assert (
                harness.validate_chat_completion_json(ra2.json(), min_cached_tokens=1)
                > 0
            )
        out, _ = srv.stop()
        done = harness.extract_done_events(harness.parse_jsonl_records(out))[-1]
        assert done["parked_lanes"]["skips"] == 0
        assert done["parked_lanes"]["restores"] >= 1


@pytest.mark.tier4
@pytest.mark.e2e_live
def test_t4_10_end_to_end_ab_eval_parity_gate(
    isolated_state_root: Path, tmp_path: Path
) -> None:
    """Scenario 10: A/B evaluation with parity gate, Wilcoxon/FDR, & holdout."""
    prompt = "Write a deterministic Rust fibonacci function."
    with harness.ServerProcess(isolated_state_root / "arm_a") as srv_a:
        ra = srv_a.client().post_chat(
            harness.make_chat_payload(
                prompt, enable_thinking=False, seed=1, temperature=0.0
            )
        )
        assert ra.status == 200
    out_a, _ = srv_a.stop()
    sha_a = harness.extract_done_events(harness.parse_jsonl_records(out_a))[-1][
        "hc_exact_final_logit_sha256"
    ]

    with harness.ServerProcess(isolated_state_root / "arm_b") as srv_b:
        rb = srv_b.client().post_chat(
            harness.make_chat_payload(
                prompt, enable_thinking=False, seed=1, temperature=0.0
            )
        )
        assert rb.status == 200
    out_b, _ = srv_b.stop()
    sha_b = harness.extract_done_events(harness.parse_jsonl_records(out_b))[-1][
        "hc_exact_final_logit_sha256"
    ]
    assert sha_a == sha_b
    blocks = harness.build_paired_block_series("winner_2block")
    f_res = harness.run_r5_futility_eval(
        tmp_path / "futility", {"mode": "futility", "alpha": 0.05, "blocks": blocks}
    )
    assert (
        f_res.returncode == 0 and json.loads(f_res.stdout)["stop_for_futility"] is False
    )
    bundle = harness.build_holdout_manifest(tmp_path / "manifest")
    h_res = harness.run_r5_holdout_eval(
        tmp_path / "holdout",
        bundle["manifest_path"].parent,
        "holdout_0",
        bundle["ledger_path"],
    )
    assert h_res.returncode == 0
