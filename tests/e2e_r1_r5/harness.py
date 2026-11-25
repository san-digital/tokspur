# SPDX-License-Identifier: Apache-2.0
"""Opaque-box E2E test harness, synthetic builders, and F1-F20 case registry."""

from __future__ import annotations

import fcntl
import hashlib
import http.client
import json
import math
import os
import re
import shlex
import shutil
import socket
import struct
import subprocess
import sys
import time
import tomllib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

import psutil

REPO_ROOT = Path(__file__).resolve().parents[2]
NATIVE_DIR = REPO_ROOT / "native"
RUST_DIR = REPO_ROOT / "rust"
BUILD_DIR = REPO_ROOT / "build"
PROFILES_DIR = REPO_ROOT / "profiles"
SCRIPTS_DIR = REPO_ROOT / "scripts"
TEST_PYTHON = sys.executable

COMPETING_PROCESS_NAMES = frozenset(
    {
        "scalar-channel-target",
        "tokspur-server",
        "tokspur-native",
        "splash",
        "llama-server",
        "llama-cli",
        "omlx",
        "omlx-server",
        "ollama",
    }
)

FLAG_READERS_EXPECTED = (
    "qsa_score_accumulate",
    "qsa_chronological_factors",
    "qsa_chronological_channels",
    "qsa_chronological_serial_factors",
    "qsa_verify_rows_factors",
    "qsa_verify_rows_channels",
)

FEATURE_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "F1": ("Acceptance Criteria", "Baseline A/B & Parity Artifacts"),
    "F2": ("R5", "Holdout Manifest (holdout.rs)"),
    "F3": ("R5", "Sequential Futility Stopping (transfer.rs, ci_tests.rs)"),
    "F4": ("R5", "native/rust.mk Receipt Cache"),
    "F5": ("R1", "Parked-Lane (committed, sampling, seed) Recording"),
    "F6": ("R1", "Hoisted Render/Encode & Echoed Reasoning"),
    "F7": ("R1", "Parked-Lane respelled_on_committed"),
    "F8": ("R1", "C++ ServeSession Parity (serve_protocol.cpp)"),
    "F9": ("R2", "graph.mm Phase B Level-Scheduling"),
    "F10": ("R2", "dispatch-graph check Verification"),
    "F11": ("R2", "error_flag (buffer(7)) Decoupling"),
    "F12": ("R2", "Default Concurrent Level-Scheduling"),
    "F13": ("R3", "Pack v3 Static dm Certification"),
    "F14": ("R3", "Shape-Registration Invariant Discharge"),
    "F15": ("R3", "Strip Hot-Loop Metal Checks & bad Live Ranges"),
    "F16": ("R3", "End-of-Batch Finiteness & Metal 3.2 Flags"),
    "F17": ("R4", "DFlash2 Lazy flush_context() on Park"),
    "F18": ("R4", "Sqrt(N) Prefix/SSD Checkpoint Spacing"),
    "F19": ("R4", "Instantaneous Anchor-Entropy Width Gating"),
    "F20": ("R1, R4", "Unified Default CLI & Worker Behavior"),
}


@dataclass(frozen=True)
class CaseSpec:
    """Metadata mapping a single E2E test node to requirements and features."""

    case_id: str
    tier: int
    feature_ids: tuple[str, ...]
    requirement: str
    description: str


@dataclass(frozen=True)
class CmdResult:
    """Captured subprocess result."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float


@dataclass(frozen=True)
class HttpResponse:
    """Captured HTTP response."""

    status: int
    headers: dict[str, str]
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))


def build_feature_case_registry() -> tuple[CaseSpec, ...]:
    """Return the complete 230-case Tier 1-4 requirement coverage registry."""
    cases: list[CaseSpec] = []

    t1_specs: dict[str, list[tuple[str, str]]] = {
        "F1": [
            ("01_health_and_models_baseline", "GET /health and /v1/models baseline"),
            (
                "02_verify_and_draft_profile_capture",
                "VERIFY_PROFILE/DRAFT_PROFILE dump",
            ),
            ("03_benchmark_parked_chats_oracle", "benchmark_parked_chats oracle"),
            ("04_stream_collector_sse_capture", "stream_collector SSE chunk capture"),
            ("05_greedy_final_logit_sha_parity", "Greedy hc_exact_final_logit_sha256"),
        ],
        "F2": [
            ("01_valid_manifest_single_use", "Valid holdout manifest single-use pass"),
            (
                "02_hash_chained_ledger_genesis",
                "Hash-chained ledger GENESIS progression",
            ),
            (
                "03_zero_calibration_leakage",
                "Disjoint calibration/holdout leakage check",
            ),
            (
                "04_accurate_eval_records_pass",
                "Holdout error within bound records pass",
            ),
            (
                "05_isolated_ledger_0600_write",
                "Isolated 0600 atomic ledger persistence",
            ),
        ],
        "F3": [
            (
                "01_loser_2block_futility_stop",
                "Losing sweep stops after 2 paired blocks",
            ),
            ("02_winner_2block_continues", "Winning sweep continues with q <= alpha"),
            ("03_wilcoxon_signed_rank_ties", "Exact Wilcoxon tie-rank averaging"),
            ("04_bh_fdr_step_up_monotone", "BH-FDR step-up q-value monotonicity"),
            ("05_zero_diff_pairs_excluded", "Zero-difference pairs excluded cleanly"),
        ],
        "F4": [
            ("01_cold_rust_check_populates_cache", "Cold rust-check populates cache"),
            ("02_warm_rust_check_skips_rebuild", "Warm back-to-back rust-check hits"),
            (
                "03_receipt_records_input_digests",
                "Receipt records toolchain/lock digests",
            ),
            ("04_native_build_inputs_clean", "test_native_build_inputs stays clean"),
            ("05_rust_release_builds_cleanly", "rust-release builds with cache active"),
        ],
        "F5": [
            (
                "01_thinking_on_parks_and_restores",
                "Thinking-on chat parks and restores",
            ),
            ("02_thinking_off_parks_and_restores", "enable_thinking=false parks lane"),
            ("03_explicit_seed_match_restores", "Matching explicit seed restores lane"),
            (
                "04_omitted_seed_compatible_match",
                "Omitted seed matches compatible lane",
            ),
            ("05_multi_conversation_independent", "Interleaved lanes keep own slots"),
        ],
        "F6": [
            ("01_echoed_reasoning_content_hit", "Echoed reasoning_content hits parked"),
            ("02_shared_turns_hoisted_encode", "Shared turns hoist render/encode"),
            ("03_empty_turns_base_prompt_hit", "Empty turns evaluate base prompt"),
            ("04_interleaved_matches_oracle", "Interleaved matches oracle outputs"),
            ("05_prefix_trace_matches_usage", "prefix-trace cached_tokens == usage"),
        ],
        "F7": [
            ("01_noncanonical_bpe_parked_hit", "Non-canonical BPE respell hits parked"),
            ("02_live_committed_respell_append", "Live committed respell appends"),
            ("03_whitespace_punct_bpe_respell", "Code whitespace/punct BPE respelling"),
            (
                "04_thinking_off_sampled_respell",
                "enable_thinking=false sampled respell",
            ),
            ("05_respell_preserves_logit_sha", "Respelling preserves final logit SHA"),
        ],
        "F8": [
            ("01_rust_diff_all_arms_exact", "rust-diff byte-exact channel/staged/init"),
            ("02_cargo_test_server_wire", "cargo test tokspur-server and wire"),
            ("03_rust_fuzz_diff_targets", "tokspur-fuzz response/http/bpe parity"),
            ("04_sse_chunk_framing_parity", "SSE framing parity C++ vs Rust"),
            ("05_serve_protocol_parked_parity", "serve_protocol.cpp parked parity"),
        ],
        "F9": [
            ("01_test_graph_unit_suite", "test-graph hazard and row_gemv cases"),
            (
                "02_independent_level0_zero_barrier",
                "Independent dispatches emit 0 barriers",
            ),
            ("03_dag_depth_d_emits_d_minus_1", "ASAP DAG depth D emits D-1 barriers"),
            ("04_dispatch_indirect_12b_span", "dispatch_indirect 12B read span edge"),
            ("05_commit_partial_asap_schedule", "commit_partial preserves ASAP levels"),
        ],
        "F10": [
            ("01_cargo_test_dispatch_graph", "cargo test -p tokspur-dispatch-graph"),
            ("02_dispatch_graph_summary_json", "dispatch-graph summary metrics"),
            ("03_dispatch_graph_check_pass", "dispatch-graph check accepts Phase B"),
            ("04_dispatch_graph_controls_hold", "dispatch-graph controls passes all"),
            (
                "05_metal_graph_trace_roundtrip",
                "MetalGraph trace passes check/controls",
            ),
        ],
        "F11": [
            ("01_kernels_rs_three_way_sync", "kernels.rs 3-way FLAG_READERS sync"),
            ("02_disjoint_slots_same_level", "Disjoint error_flag slots concurrent"),
            ("03_qsa_flag_readers_decoupled", "QSA FLAG_READERS decoupled cleanly"),
            (
                "04_submit_host_reduction_detects",
                "Host submit() reduction catches error",
            ),
            ("05_test_graph_operator_ckpt", "test-graph --operator-checkpoints pass"),
        ],
        "F12": [
            ("01_default_serve_enables_conc", "Default serve emits concurrent on"),
            ("02_sampled_stack_concurrent_parity", "test-sampled_stack --concurrent"),
            (
                "03_verify_profile_fewer_barriers",
                "VERIFY_PROFILE shows less serial stall",
            ),
            ("04_tokspur_cli_default_concurrent", "cargo test -p tokspur-cli defaults"),
            ("05_no_concurrent_dispatch_optout", "--no-concurrent-dispatch disables"),
        ],
        "F13": [
            ("01_test_planar_pack_v3_passes", "test-planar_pack_v3 passes all codes"),
            ("02_pack_v3_plan_json_valid", "tokspur pack plan --format v3 JSON"),
            ("03_build_pack_v3_seals_output", "tokspur-build-pack-v3 builds + seals"),
            ("04_rebuild_verifies_existing", "Re-run returns verified-existing"),
            ("05_local_seal_v2_accepted", "TrustMode::local_seal accepts valid pack"),
        ],
        "F14": [
            ("01_register_planar_matrix_valid", "Valid planar matrix shapes register"),
            ("02_m5_headline_screen_dual_pass", "M5_HEADLINE_SCREEN=0 and 1 pass"),
            ("03_valid_routes_up_to_experts", "Valid route_id < experts execute"),
            ("04_small_m_and_nax_bounds_hold", "sb/group32 < blocks_per_row hold"),
            ("05_native_cpu_discharge_tests", "make -C native test discharge checks"),
        ],
        "F15": [
            ("01_stripped_planar_moe_nmse", "Stripped planar_moe_v3 NMSE <= 1e-4"),
            ("02_stripped_dflash2_stage_pass", "Stripped dflash2.metal stage passes"),
            ("03_gu_stage64_column32_parity", "COLUMN32 & gu-stage64 parity pass"),
            ("04_output_sentinel_untouched", "Output stride/row sentinels untouched"),
            ("05_metallib_objdump_symbols", "metal-objdump lists stripped kernels"),
        ],
        "F16": [
            (
                "01_metal32_40_fp_policy",
                "Metal kernels retain their released contraction policy",
            ),
            ("02_test_logit_topk_tracking", "test-logit_topk tracks nonfinite/sub"),
            ("03_test_sampled_stack_nan_catch", "test-sampled_stack catches NaN/Inf"),
            ("04_stack_no_topk_blindspot_fix", "End-of-batch check runs without topk"),
            ("05_dflash2_selector_nan_catch", "dflash2 selector NaN caught cleanly"),
        ],
        "F17": [
            ("01_dflash2_defer_parks_no_skip", "DFlash2 deferred parks with skips=0"),
            ("02_dflash2_parked_ram_restore", "DFlash2 parked return restores RAM"),
            ("03_dflash2_sequential_stays_defer", "Sequential DFlash2 stays deferred"),
            ("04_dflash2_parked_logit_parity", "Parked DFlash2 matches greedy SHA"),
            ("05_dflash2_ctx_pos_consistent", "DFlash2 context_position consistent"),
        ],
        "F18": [
            ("01_sqrt_n_checkpoints_retained", "64K-128K retains sqrt(N) checkpoints"),
            ("02_mid_context_branch_restores", "Mid-context branch restores nearest"),
            ("03_replayed_tokens_sublinear", "replayed_input_tokens bounded O(sqrtN)"),
            ("04_ssd_checkpoint_spill_restore", "SSD checkpoint spill & restore pass"),
            ("05_long_ctx_branch_logit_parity", "64K-128K branch matches greedy SHA"),
        ],
        "F19": [
            ("01_low_entropy_wide_draft_lookup", "Low-entropy prompt uses wide/lookup"),
            ("02_high_entropy_narrows_width", "High-entropy prompt narrows width"),
            (
                "03_zero_full_vocab_readback_stall",
                "MtpAnchorCarry avoids full readback",
            ),
            ("04_entropy_gate_mtp3_and_dflash2", "Entropy gating works MTP3 & DFlash2"),
            ("05_entropy_gate_greedy_parity", "Entropy gating matches greedy SHA"),
        ],
        "F20": [
            (
                "01_default_serve_enables_r1_park",
                "Default serve enables R1 parked reuse",
            ),
            ("02_default_dflash2_enables_flush", "Default dflash2 enables lazy flush"),
            ("03_default_serve_r2_r4_active", "Default serve enables R2/R4 features"),
            ("04_python_rust_cli_default_sync", "Python and Rust CLIs share defaults"),
            ("05_doctor_plan_verify_isolated", "doctor/plan/verify-build isolated OK"),
        ],
    }

    t2_specs: dict[str, list[tuple[str, str]]] = {
        "F1": [
            ("01_non_nominal_guard_ineligible", "Thermal/swap guard fails eligibility"),
            (
                "02_unknown_key_and_bad_auth_400_401",
                "Unknown JSON key 400 / bad auth 401",
            ),
            ("03_origin_403_and_non_loopback", "Origin header 403 & non-loopback ref"),
            ("04_benchmark_refuses_overwrite", "benchmark_parked_chats refuses --out"),
            ("05_profile_unset_no_dump_mismatch", "Unset env omits profile / SHA diff"),
        ],
        "F2": [
            ("01_single_use_rejects_replay", "Replay by id/sha/lock rejected"),
            (
                "02_consume_before_threshold_burns",
                "Failing accuracy still burns holdout",
            ),
            ("03_path_traversal_symlink_reject", "Path traversal and symlink rejected"),
            ("04_calibration_leakage_rejected", "Calibration/holdout overlap rejected"),
            ("05_ledger_chain_tamper_detected", "Tampered ledger chain rejected"),
        ],
        "F3": [
            ("01_single_block_no_premature_stop", "1 block does not futility-stop"),
            ("02_exact_tie_no_false_stop", "Exact tie does not trigger stop"),
            ("03_borderline_winner_not_aborted", "Borderline winner avoids abort"),
            ("04_invalid_alpha_or_len_mismatch", "Bad alpha or length mismatch error"),
            ("05_bh_fdr_extreme_nan_pvalues", "Extreme 0/1/NaN p-values handled"),
        ],
        "F4": [
            ("01_wire_change_invalidates_cache", "Source/flag change invalidates key"),
            ("02_cache_zero_bypasses_receipts", "RUST_RECEIPT_CACHE=0 forces rebuild"),
            ("03_corrupt_receipt_json_rebuilds", "Corrupted receipt JSON fails safe"),
            ("04_cpp_oracle_toggle_distinct_key", "CPP_ORACLE=0 vs 1 distinct keys"),
            ("05_missing_cache_dir_safe_create", "Missing cache dir recreated outside"),
        ],
        "F5": [
            (
                "01_mismatched_seed_forces_miss",
                "Mismatched seed misses without corrupt",
            ),
            ("02_mismatched_sampling_refuses", "Mismatched sampling policy refuses"),
            ("03_short_below_min_prompt_safe", "Below-min prompt safe no-restore"),
            ("04_ambiguous_prefix_deterministic", "Ambiguous equal prefixes resolve"),
            (
                "05_budget_eviction_cold_fallback",
                "Budget eviction increments evictions",
            ),
        ],
        "F6": [
            (
                "01_tampered_reasoning_invalidates",
                "Modified reasoning invalidates tail",
            ),
            (
                "02_conflicting_thinking_kwargs_400",
                "Conflicting thinking kwargs -> 400",
            ),
            ("03_oversized_reasoning_body_limit", "Oversized echoed reasoning bounded"),
            ("04_unicode_cjk_control_escaping", "CJK/emoji/control char roundtrip"),
            ("05_preserve_thinking_toggle_safe", "preserve_thinking false/true safe"),
        ],
        "F7": [
            ("01_divergent_edit_stops_respell", "Divergent text edit stops respell"),
            (
                "02_noncanonical_bpe_bad_seed_miss",
                "Non-canonical BPE + bad seed misses",
            ),
            ("03_utf8_multibyte_boundary_safe", "UTF-8 multibyte split safe"),
            ("04_repeated_token_longest_match", "Repeated token runs pick longest"),
            ("05_empty_committed_returns_miss", "Empty lane.committed misses safely"),
        ],
        "F8": [
            ("01_rust_diff_negative_ctrl_fails", "rust-diff negative control detects"),
            ("02_malformed_body_method_parity", "Malformed JSON / PUT 405 parity"),
            ("03_cancel_before_start_no_leak", "Client cancel leaves clean worker"),
            ("04_max_tokens_bounds_and_conflict", "max_tokens 1/32768/conflict parity"),
            ("05_calibrated_sampling_400_parity", "Calibrated sampling refusal parity"),
        ],
        "F9": [
            ("01_overlap_vs_disjoint_byte_spans", "Overlap barrier vs disjoint free"),
            (
                "02_zero_unaligned_immutable_reject",
                "Zero/unaligned/immutable FormatError",
            ),
            ("03_batch_state_machine_violations", "Double begin / orphan submit fail"),
            ("04_startup_mode_lock_enforced", "Mode locked after profiling/batch"),
            ("05_empty_batch_zero_dispatch_ok", "Empty begin->submit produces 0/0"),
        ],
        "F10": [
            ("01_missing_barrier_rejected", "check rejects missing conflict barrier"),
            ("02_swapped_or_dropped_rejected", "check rejects swapped/dropped ops"),
            ("03_mismatched_state_sha_rejected", "check rejects state SHA mismatch"),
            ("04_malformed_jsonl_rejected", "Malformed JSONL trace exits 1"),
            ("05_controls_violation_exits_one", "controls failure exits 1 held=false"),
        ],
        "F11": [
            ("01_over_16384_dispatches_fallback", ">16384 dispatches safe fallback"),
            ("02_middle_dispatch_error_sticky", "Mid-batch error preserved to submit"),
            ("03_flag_reader_keeps_dep_barrier", "FLAG_READERS keeps writer barrier"),
            ("04_out_of_sync_flag_table_caught", "Out-of-sync FLAG_READERS caught"),
            (
                "05_aborted_batch_clears_error_flag",
                "abort() clears transient error_flag",
            ),
        ],
        "F12": [
            ("01_concurrent_plus_capture_reject", "Concurrent + capture/profile fails"),
            ("02_aborted_first_batch_locks_mode", "Aborted first batch locks mode"),
            ("03_concurrent_vs_serial_sha_match", "Concurrent == serial logit SHA"),
            ("04_explicit_and_default_flags_same", "Default == --concurrent-dispatch"),
            ("05_invalid_concurrent_flag_reject", "Invalid concurrent CLI flag errors"),
        ],
        "F13": [
            ("01_pack_v3_nan_dm_0x7e00_reject", "FP16 dm NaN 0x7e00 rejected"),
            ("02_pack_v3_inf_dm_0x7c00_reject", "FP16 dm Inf 0x7c00 rejected"),
            (
                "03_tampered_payload_or_seal_reject",
                "1-bit payload/seal tamper rejected",
            ),
            ("04_sm16_out_of_range_64_rejected", "unpack_sm16 rejects sm >= 64"),
            ("05_bad_gguf_suffix_or_chunk_mib", "Bad suffix or chunk-mib 0/65 fails"),
        ],
        "F14": [
            ("01_route_id_ge_experts_rejected", "route_id >= experts FormatError"),
            ("02_unaligned_or_zero_dim_rejected", "Zero or unaligned shape rejected"),
            ("03_non_planar_consumer_rejected", "gemv/copy on planar-v3 rejected"),
            ("04_raw_upload_dm_nan_inf_rejected", "Raw upload dm NaN/Inf rejected"),
            ("05_dflash2_oob_stage_dup_prepare", "OOB stage / dup prepare rejected"),
        ],
        "F15": [
            ("01_narrow_rows_1_2_vs_nax_3_32", "Narrow 1-2 vs NAX 3-32 boundaries"),
            ("02_non_pow2_rows_stride_padding", "Non-pow2 rows keep stride intact"),
            ("03_extreme_finite_65504_no_false", "Extreme finite 65504 passes"),
            ("04_fp16_stage_overflow_caught", "FP16 stage overflow caught EOB"),
            ("05_nax_staging_only_poison_catch", "--nax-staging-only catches poison"),
        ],
        "F16": [
            ("01_final_col_nan_0x7e00_caught", "Final vocab col NaN caught"),
            ("02_pos_neg_inf_0x7c00_fc00_caught", "+Inf/-Inf caught across paths"),
            ("03_swiglu_3e38_overflow_caught", "SwiGLU 3e38 overflow caught EOB"),
            ("04_subnormal_sampling_logit_catch", "Subnormal sampling logit caught"),
            ("05_clean_retry_after_nan_recovers", "Clean retry after NaN recovers"),
        ],
        "F17": [
            ("01_short_single_token_dflash2_safe", "Short prefill park safe"),
            ("02_three_way_ping_pong_zero_skips", "3-way A->B->C->A has skips=0"),
            ("03_dflash2_bad_seed_no_stale_ctx", "Mismatched seed resets context"),
            ("04_dflash2_lookup_park_preserves", "DFlash2+lookup keeps anchor safe"),
            ("05_dflash2_budget_evict_recovers", "Budget eviction cold-recovers"),
        ],
        "F18": [
            ("01_64k_vs_128k_context_boundary", "64K/128K lifts 2048 cap cleanly"),
            ("02_over_context_length_400_reject", "> context_length rejected 400"),
            ("03_checkpoint_cap_range_32gib_max", "checkpoint-cap-gib > 32 rejected"),
            ("04_floor_gt_cap_or_symlink_reject", "floor > cap or symlink dir fails"),
            ("05_tight_ssd_cap_keeps_sqrt_miles", "Tight cap retains sqrt(N) miles"),
        ],
        "F19": [
            ("01_no_pending_anchor_step_safe", "has_pending_anchor=false safe"),
            ("02_entropy_zero_and_uniform_ext", "Zero-entropy vs uniform extremes"),
            ("03_lookup_miss_falls_back_clean", "Lookup miss falls back to draft"),
            ("04_nonfinite_anchor_triggers_r3", "Nonfinite anchor caught by R3"),
            ("05_r6_no_history_early_cap_guard", "R6: no history-only DFlash2 cap"),
        ],
        "F20": [
            ("01_explicit_no_shared_or_park_0", "--no-shared-prefix-cache opts out"),
            ("02_explicit_no_concurrent_optout", "--no-concurrent-dispatch opts out"),
            ("03_conflicting_cli_flags_nonzero", "Conflicting CLI flags exit != 0"),
            ("04_serve_lock_exclusivity_enforced", "serve.lock rejects 2nd owner"),
            ("05_r6_no_lean_nanoda_cli_flags", "R6: no Lean4/nanoda flags exist"),
        ],
    }

    for fid in (f"F{i}" for i in range(1, 21)):
        req, _ = FEATURE_DESCRIPTIONS[fid]
        for suffix, desc in t1_specs[fid]:
            cases.append(
                CaseSpec(
                    case_id=f"t1_{fid.lower()}_{suffix}",
                    tier=1,
                    feature_ids=(fid,),
                    requirement=req,
                    description=desc,
                )
            )
        for suffix, desc in t2_specs[fid]:
            cases.append(
                CaseSpec(
                    case_id=f"t2_{fid.lower()}_{suffix}",
                    tier=2,
                    feature_ids=(fid,),
                    requirement=req,
                    description=desc,
                )
            )

    t3_specs: list[tuple[str, tuple[str, ...], str, str]] = [
        (
            "t3_01_r1_r4_dflash2_parked_respelling",
            ("F5", "F7", "F17"),
            "R1+R4",
            "DFlash2 deferred lazy flush + non-canonical BPE respelling",
        ),
        (
            "t3_02_r1_r4_entropy_gated_parked_turns",
            ("F6", "F19", "F20"),
            "R1+R4",
            "Echoed reasoning parked restore + anchor-entropy width gate",
        ),
        (
            "t3_03_r2_r3_level_scheduled_planar_moe",
            ("F9", "F13", "F15"),
            "R2+R3",
            "Phase B level-scheduling + certified Pack v3 stripped planar MoE",
        ),
        (
            "t3_04_r2_r3_error_flag_end_of_batch_nan",
            ("F11", "F16"),
            "R2+R3",
            "Decoupled error_flag slots + end-of-batch NaN/Inf detection",
        ),
        (
            "t3_05_r3_r4_dflash2_stripped_sqrt_ckpt",
            ("F15", "F17", "F18"),
            "R3+R4",
            "Stripped dflash2.metal + lazy flush + sqrt(N) checkpoint replay",
        ),
        (
            "t3_06_r3_r4_anchor_entropy_topk_finite",
            ("F14", "F16", "F19"),
            "R3+R4",
            "Shape-certified top-k logits + finiteness + entropy width gate",
        ),
        (
            "t3_07_r1_r5_rust_diff_under_cache",
            ("F4", "F8"),
            "R1+R5",
            "Cached rust.mk 5-way build feeding rust-diff byte-exact parity",
        ),
        (
            "t3_08_r1_r5_parked_ab_holdout_futility",
            ("F1", "F2", "F3", "F5"),
            "R1+R5",
            "Parked-chat A/B comparison -> Wilcoxon futility & holdout",
        ),
        (
            "t3_09_r2_r4_concurrent_sqrt_replay",
            ("F10", "F12", "F18"),
            "R2+R4",
            "dispatch-graph verified concurrent dispatch on 64K+ sqrt(N) replay",
        ),
        (
            "t3_10_r2_r4_decoupled_flag_entropy_draft",
            ("F11", "F17", "F19"),
            "R2+R4",
            "Decoupled error_flag across DFlash2 flush + entropy-gated draft",
        ),
        (
            "t3_11_r4_r5_dflash2_sweep_futility_stop",
            ("F3", "F17", "F19"),
            "R4+R5",
            "DFlash2 + entropy-gate sweep evaluated by 2-block futility stop",
        ),
        (
            "t3_12_r4_r5_long_ctx_holdout_manifest",
            ("F2", "F18", "F20"),
            "R4+R5",
            "Default 64K-128K checkpoint replay verified via SHA-256 holdout",
        ),
        (
            "t3_13_r1_r2_concurrent_parked_mtp3",
            ("F5", "F9", "F12"),
            "R1+R2",
            "Default Phase B concurrent dispatch serving MTP3 parked lanes",
        ),
        (
            "t3_14_r1_r2_cpp_parity_dispatch_graph",
            ("F8", "F10", "F11"),
            "R1+R2",
            "C++ ServeSession parity + dispatch-graph controls + FLAG_READERS",
        ),
        (
            "t3_15_r1_r3_certified_pack_echoed_reason",
            ("F6", "F13", "F16"),
            "R1+R3",
            "Pack v3 certified model serving echoed reasoning + EOB finiteness",
        ),
        (
            "t3_16_r1_r3_respelling_shape_invariants",
            ("F7", "F14", "F15"),
            "R1+R3",
            "Non-canonical BPE respelling over shape-discharged stripped MoE",
        ),
        (
            "t3_17_r2_r5_cached_dispatch_graph_ci",
            ("F4", "F9", "F10"),
            "R2+R5",
            "rust.mk cached check + tokspur-dispatch-graph check & controls",
        ),
        (
            "t3_18_r2_r5_concurrent_barrier_futility",
            ("F1", "F3", "F12"),
            "R2+R5",
            "Barrier-serialization VERIFY_PROFILE A/B sweep via Wilcoxon/FDR",
        ),
        (
            "t3_19_r3_r5_pack_v3_holdout_verify",
            ("F2", "F13", "F14"),
            "R3+R5",
            "Pack v3 static certification & M5_HEADLINE_SCREEN via holdout",
        ),
        (
            "t3_20_r3_r5_cached_metal_flags_check",
            ("F4", "F15", "F16"),
            "R3+R5",
            "rust.mk receipt cache + Metal 3.2/4.0 flags + check_sources.py",
        ),
    ]
    for cid, fids, req, desc in t3_specs:
        cases.append(
            CaseSpec(
                case_id=cid,
                tier=3,
                feature_ids=fids,
                requirement=req,
                description=desc,
            )
        )

    t4_specs: list[tuple[str, tuple[str, ...], str, str]] = [
        (
            "t4_01_multiturn_parked_chats_mtp3",
            ("F5", "F6", "F7", "F20"),
            "R1",
            "Multi-turn interleaved parked chats (MTP3, echoed reasoning, BPE)",
        ),
        (
            "t4_02_multiturn_parked_chats_dflash2",
            ("F5", "F7", "F17", "F20"),
            "R1+R4",
            "Multi-turn interleaved parked chats (DFlash2 lazy flush, skips=0)",
        ),
        (
            "t4_03_long_context_sqrt_checkpoint_replay",
            ("F1", "F18", "F20"),
            "R4",
            "64K-128K long-context branch & SSD sqrt(N) checkpoint replay",
        ),
        (
            "t4_04_m5_decode_profile_dispatch_graph",
            ("F9", "F10", "F11", "F12"),
            "R2",
            "M5 Max decode + VERIFY/DRAFT_PROFILE + dispatch-graph check",
        ),
        (
            "t4_05_instantaneous_anchor_entropy_gating",
            ("F1", "F19", "F20"),
            "R4",
            "Instantaneous anchor-entropy width gating (lookup/MTP3/DFlash2)",
        ),
        (
            "t4_06_r5_optimization_turnaround_pipeline",
            ("F2", "F3", "F4", "F8"),
            "R5+R1",
            "R5 pipeline: cached 5-way build -> rust-diff -> futility -> holdout",
        ),
        (
            "t4_07_pack_v3_dual_screen_and_nan_catch",
            ("F13", "F14", "F15", "F16"),
            "R3",
            "Pack v3 certification, M5_HEADLINE_SCREEN=0/1 & EOB NaN catch",
        ),
        (
            "t4_08_concurrent_http_sse_streaming",
            ("F1", "F5", "F6", "F8"),
            "R1",
            "Concurrent HTTP/SSE streaming, include_usage, 429, 403 Origin",
        ),
        (
            "t4_09_zero_opt_in_default_cli_launch",
            ("F12", "F17", "F19", "F20"),
            "R1+R2+R3+R4",
            "Zero-opt-in default tokspur serve (mtp & dflash2)",
        ),
        (
            "t4_10_end_to_end_ab_eval_parity_gate",
            ("F1", "F2", "F3", "F20"),
            "Acceptance+R5",
            "End-to-end A/B evaluation & greedy logit parity gate",
        ),
    ]
    for cid, fids, req, desc in t4_specs:
        cases.append(
            CaseSpec(
                case_id=cid,
                tier=4,
                feature_ids=fids,
                requirement=req,
                description=desc,
            )
        )

    return tuple(cases)


FEATURE_CASE_REGISTRY: tuple[CaseSpec, ...] = build_feature_case_registry()


def find_competing_inference_processes() -> list[tuple[int, str]]:
    """Return running inference/GPU processes that must not overlap live runs."""
    found: list[tuple[int, str]] = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (proc.info.get("name") or "").strip()
            cmdline = proc.info.get("cmdline") or []
            exe_name = Path(cmdline[0]).name if cmdline else ""
            if name in COMPETING_PROCESS_NAMES or exe_name in COMPETING_PROCESS_NAMES:
                found.append((int(proc.info["pid"]), exe_name or name))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return found


def assert_no_competing_inference_processes() -> None:
    """Refuse to start GPU/serving tests while another inference process runs."""
    competing = find_competing_inference_processes()
    if competing:
        details = ", ".join(f"{name}(pid={pid})" for pid, name in competing)
        raise RuntimeError(
            f"Refusing live GPU/server test: competing inference process(es): {details}"
        )


@contextmanager
def acquire_state_lock(state_dir: Path) -> Iterator[None]:
    """Acquire non-blocking flock on <state_dir>/serve.lock."""
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / "serve.lock"
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"State lock already held: {lock_path}") from None
        yield
    finally:
        os.close(fd)


def _first_existing(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def find_tokspur_cli() -> Path | None:
    return _first_existing(
        [
            BUILD_DIR / "rust-fast-math/release/tokspur",
            BUILD_DIR / "rust-target/release/tokspur",
            BUILD_DIR / "rust-target/debug/tokspur",
        ]
    )


def seed_isolated_state_root(
    state_root: Path, *, serve_args: list[str] | None = None
) -> list[str]:
    """Seed adoption.json and derived-text assets into an isolated state root."""
    tokspur_dir = state_root / ".tokspur"
    state_dir = tokspur_dir / "state"
    derived_text_dir = state_dir / "gguf" / "derived-text"
    for d in (state_root, tokspur_dir, state_dir, state_dir / "gguf", derived_text_dir):
        d.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(d, 0o700)

    src_adoption = REPO_ROOT / ".tokspur/adoption.json"
    dst_adoption = tokspur_dir / "adoption.json"
    if src_adoption.is_file() and not dst_adoption.exists():
        shutil.copyfile(src_adoption, dst_adoption)
        os.chmod(dst_adoption, 0o600)

    src_derived = REPO_ROOT / ".tokspur/state/gguf/derived-text"
    for asset_name in ("tokenizer.json", "chat_template.jinja"):
        src_asset = src_derived / asset_name
        dst_asset = derived_text_dir / asset_name
        if src_asset.is_file() and not dst_asset.exists():
            shutil.copyfile(src_asset, dst_asset)
            os.chmod(dst_asset, 0o600)

    args = serve_args or []
    extra_args: list[str] = []
    if not dst_adoption.exists():
        model_dir = os.environ.get("TOKSPUR_E2E_MODEL_DIR", "").strip()
        if model_dir and "--model-dir" not in args:
            extra_args.extend(["--model-dir", model_dir])

    wants_dflash2 = False
    for idx, tok in enumerate(args):
        if tok == "--drafter=dflash2" or (
            tok == "--drafter" and idx + 1 < len(args) and args[idx + 1] == "dflash2"
        ):
            wants_dflash2 = True
            break
    if wants_dflash2 and not any(
        a == "--dflash2-dir" or a.startswith("--dflash2-dir=") for a in args
    ):
        dflash2_env = os.environ.get("TOKSPUR_E2E_DFLASH2_DIR", "").strip()
        dflash2_dir = (
            Path(dflash2_env)
            if dflash2_env
            else (REPO_ROOT / ".tokspur/drafter/hf-stage")
        )
        if dflash2_dir.exists():
            extra_args.extend(["--dflash2-dir", str(dflash2_dir)])
    return extra_args


def find_tokspur_server() -> Path | None:
    return _first_existing(
        [
            BUILD_DIR / "rust-fast-math/release/tokspur-server",
            BUILD_DIR / "rust-ordinary/release/tokspur-server",
            BUILD_DIR / "owned-native/tokspur-server",
        ]
    )


def find_tokspur_native() -> Path | None:
    return _first_existing(
        [
            BUILD_DIR / "owned-native/tokspur-native",
            BUILD_DIR / "native-plain-loads-component/tokspur-native",
        ]
    )


def find_metallib() -> Path | None:
    return _first_existing(
        [
            BUILD_DIR / "owned-native/tokspur-owned.metallib",
            BUILD_DIR / "carrier-diagnostic/tokspur-owned.metallib",
        ]
    )


def find_dispatch_graph() -> Path | None:
    return _first_existing(
        [
            BUILD_DIR / "rust-target/release/dispatch-graph",
            BUILD_DIR / "rust-target/debug/dispatch-graph",
            BUILD_DIR / "rust-fast-math/release/dispatch-graph",
        ]
    )


def find_build_pack_v3() -> Path | None:
    return _first_existing(
        [
            BUILD_DIR / "owned-native/tokspur-build-pack-v3",
        ]
    )


def find_native_test_bin(name: str) -> Path | None:
    return _first_existing(
        [
            BUILD_DIR / "owned-native" / name,
            BUILD_DIR / "carrier-diagnostic" / name,
        ]
    )


def find_r5_cli_bin(tool_hint: str) -> Path | None:
    """Locate a fresh R5 CLI binary (holdout or futility) if newer than Rust sources."""
    candidates = [
        BUILD_DIR / "rust-fast-math/release" / tool_hint,
        BUILD_DIR / "rust-target/release" / tool_hint,
        BUILD_DIR / "rust-target/debug" / tool_hint,
    ]
    found = _first_existing(candidates)
    if found is None:
        return None
    eval_dir = RUST_DIR / "crates/tokspur-eval"
    if eval_dir.is_dir():
        bin_mtime = found.stat().st_mtime_ns
        for src in (eval_dir / "Cargo.toml", *eval_dir.rglob("*.rs")):
            if src.is_file() and src.stat().st_mtime_ns > bin_mtime:
                return None
    return found


def require_artifact(path: Path | None, label: str) -> Path:
    """Fail loudly in live mode when a required binary or asset is absent."""
    if path is None or not path.exists():
        raise AssertionError(f"Required live artifact missing: {label}")
    return path


def run_cmd(
    argv: list[str] | tuple[str, ...],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float = 300.0,
) -> CmdResult:
    """Run subprocess synchronously and capture stdout, stderr, returncode."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    start = time.monotonic()
    proc = subprocess.run(
        list(argv),
        cwd=str(cwd or REPO_ROOT),
        env=merged_env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    elapsed = time.monotonic() - start
    return CmdResult(
        argv=tuple(argv),
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        duration_s=elapsed,
    )


def run_tokspur(
    args: list[str],
    *,
    state_root: Path | None = None,
    cli_impl: str = "rust",
    env: dict[str, str] | None = None,
    timeout: float = 120.0,
) -> CmdResult:
    """Invoke tokspur CLI with optional --state-root isolation."""
    extra_env = {"TOKSPUR_CLI_IMPL": cli_impl}
    if env:
        extra_env.update(env)
    if cli_impl == "python":
        search_path = extra_env.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
        extra_env["PYTHONPATH"] = os.pathsep.join(
            filter(None, [str(REPO_ROOT / "src"), search_path])
        )
        cmd = [TEST_PYTHON, "-m", "tokspur.cli"]
    else:
        cli_bin = require_artifact(find_tokspur_cli(), "tokspur CLI")
        cmd = [str(cli_bin)]
    if state_root is not None and cli_impl != "python":
        cmd.extend(["--state-root", str(state_root)])
    cmd.extend(args)
    return run_cmd(cmd, env=extra_env, timeout=timeout)


def run_dispatch_graph(args: list[str], *, timeout: float = 60.0) -> CmdResult:
    """Invoke dispatch-graph binary (or cargo run --frozen fallback)."""
    dg_bin = find_dispatch_graph()
    if dg_bin is not None:
        return run_cmd([str(dg_bin), *args], timeout=timeout)
    return run_cmd(
        [
            "cargo",
            "run",
            "--frozen",
            "-q",
            "-p",
            "tokspur-dispatch-graph",
            "--",
            *args,
        ],
        cwd=RUST_DIR,
        timeout=timeout,
    )


def run_make_native(
    args: list[str],
    *,
    makefile: str | None = None,
    env: dict[str, str] | None = None,
    timeout: float = 600.0,
) -> CmdResult:
    """Invoke make -C native ..."""
    cmd = ["make", "-C", str(NATIVE_DIR)]
    if makefile:
        cmd.extend(["-f", makefile])
    cmd.extend(args)
    return run_cmd(cmd, env=env, timeout=timeout)


_CARGO_RESULT_RE = re.compile(
    r"^test result:\s+(\w+)\.\s+(\d+)\s+passed;\s+(\d+)\s+failed;\s+(\d+)\s+ignored;",
    re.MULTILINE,
)


def parse_cargo_test_counts(stdout: str) -> tuple[int, int, int]:
    """Sum (passed, failed, ignored) across all `test result:` summary lines."""
    passed = 0
    failed = 0
    ignored = 0
    for match in _CARGO_RESULT_RE.finditer(stdout):
        passed += int(match.group(2))
        failed += int(match.group(3))
        ignored += int(match.group(4))
    return passed, failed, ignored


def run_cargo_test(
    args: list[str], *, env: dict[str, str] | None = None, timeout: float = 600.0
) -> CmdResult:
    """Invoke cargo test --frozen inside rust/."""
    return run_cmd(
        ["cargo", "test", "--frozen", *args],
        cwd=RUST_DIR,
        env=env,
        timeout=timeout,
    )


def run_cargo_test_checked(
    args: list[str],
    *,
    min_passed: int = 1,
    env: dict[str, str] | None = None,
    timeout: float = 600.0,
) -> CmdResult:
    """Run cargo test --frozen and fail if 0 tests passed or any failed."""
    res = run_cargo_test(args, env=env, timeout=timeout)
    assert res.returncode == 0, f"cargo test failed: {res.stderr}\n{res.stdout}"
    passed, failed, _ = parse_cargo_test_counts(res.stdout)
    assert failed == 0, f"cargo test reported {failed} failures:\n{res.stdout}"
    assert passed >= min_passed, (
        f"cargo test passed={passed} < min_passed={min_passed} "
        f"(silent filter miss?):\n{res.stdout}"
    )
    return res


def run_cargo_metadata() -> dict[str, Any]:
    """Run `cargo metadata --offline --no-deps --format-version 1` inside rust/."""
    res = run_cmd(
        ["cargo", "metadata", "--offline", "--no-deps", "--format-version", "1"],
        cwd=RUST_DIR,
        timeout=60.0,
    )
    assert res.returncode == 0, f"cargo metadata failed: {res.stderr}"
    return json.loads(res.stdout)


def create_cargo_logging_shim(tmp_path: Path) -> tuple[Path, Path]:
    """Create an executable cargo wrapper that logs non-version invocations."""
    real_cargo = shutil.which("cargo") or "cargo"
    shim_path = tmp_path / "cargo-shim"
    log_path = tmp_path / "cargo_calls.log"
    shim_path.write_text(
        "#!/bin/sh\n"
        'case "${1:-}" in\n'
        "  -V|--version|-vV)\n"
        f'    exec "{real_cargo}" "$@"\n'
        "    ;;\n"
        "esac\n"
        'if [ -n "${TOKSPUR_E2E_CARGO_LOG:-}" ]; then\n'
        '  printf "%s\\n" "$*" >> "$TOKSPUR_E2E_CARGO_LOG"\n'
        "fi\n"
        f'exec "{real_cargo}" "$@"\n',
        encoding="utf-8",
    )
    os.chmod(shim_path, 0o755)
    return shim_path, log_path


def read_cached_cargo_calls(log_path: Path) -> list[str]:
    """Return logged cargo invocations wrapped by receipt cache (clippy/test/build)."""
    if not log_path.exists():
        return []
    return [
        ln.strip()
        for ln in log_path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("fmt ")
    ]


@contextmanager
def temporary_wire_comment_probe() -> Iterator[Path]:
    """Temporarily append a rustfmt-clean comment to tokspur-wire/src/lib.rs."""
    wire_lib = RUST_DIR / "crates/tokspur-wire/src/lib.rs"
    orig_bytes = wire_lib.read_bytes()
    orig_stat = wire_lib.stat()
    probe = b"// e2e-cache-invalidation-probe\n"
    wire_lib.write_bytes(orig_bytes + probe)
    try:
        yield wire_lib
    finally:
        wire_lib.write_bytes(orig_bytes)
        os.utime(wire_lib, ns=(orig_stat.st_atime_ns, orig_stat.st_mtime_ns))


def run_script(
    script_name: str,
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout: float = 300.0,
) -> CmdResult:
    """Invoke a Python script in scripts/ with the test interpreter."""
    script_path = require_artifact(SCRIPTS_DIR / script_name, script_name)
    return run_cmd(
        [TEST_PYTHON, str(script_path), *args],
        env=env,
        timeout=timeout,
    )


def pick_unused_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def read_api_key(state_dir: Path) -> str:
    key_path = state_dir / "api-key"
    key = key_path.read_text(encoding="utf-8").strip()
    if len(key) != 64 or not all(ch in "0123456789abcdefABCDEF" for ch in key):
        raise ValueError(f"Invalid api-key in {key_path}")
    return key


def parse_jsonl_records(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("{"):
            records.append(json.loads(line))
    return records


def extract_done_events(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in records:
        if rec.get("event") == "done":
            out.append(rec.get("native_done", rec))
    return out


def wait_for_done_events(
    stdout_log_path: Path,
    *,
    min_count: int = 1,
    timeout: float = 5.0,
    poll_interval: float = 0.02,
) -> list[dict[str, Any]]:
    """Poll server stdout JSONL until at least `min_count` done events appear."""
    deadline = time.monotonic() + timeout
    last_events: list[dict[str, Any]] = []
    while True:
        if stdout_log_path.exists():
            text = stdout_log_path.read_text(encoding="utf-8")
            if text and not text.endswith("\n"):
                text = text.rsplit("\n", 1)[0] if "\n" in text else ""
            if text:
                events = extract_done_events(parse_jsonl_records(text))
                last_events = events
                if len(events) >= min_count:
                    return events
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out after {timeout:.1f}s waiting for {min_count} done event(s) "
                f"in {stdout_log_path} (saw {len(last_events)})"
            )
        time.sleep(poll_interval)


_PROFILE_RE = re.compile(
    r"^(VERIFY_PROFILE|DRAFT_PROFILE|PREFILL_PROFILE)\s+"
    r"sampled_ms=([0-9.eE+-]+)\s+gpu_ms=([0-9.eE+-]+)\s+dispatches=(\d+)"
)


def parse_profile_lines(stderr: str, kind: str) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for line in stderr.splitlines():
        match = _PROFILE_RE.match(line.strip())
        if match and match.group(1) == kind:
            parsed.append(
                {
                    "kind": kind,
                    "sampled_ms": float(match.group(2)),
                    "gpu_ms": float(match.group(3)),
                    "dispatches": int(match.group(4)),
                    "raw": line.strip(),
                }
            )
    return parsed


class HttpClient:
    """Minimal HTTP/1.1 + SSE client matching tokspur-server wire contract."""

    def __init__(self, port: int, api_key: str, host: str = "127.0.0.1") -> None:
        self.host = host
        self.port = port
        self.api_key = api_key
        self.base_url = f"http://{host}:{port}/v1"

    def request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        include_auth: bool = True,
        timeout: float = 120.0,
    ) -> HttpResponse:
        req_headers = {
            "Host": f"{self.host}:{self.port}",
            "Connection": "close",
        }
        if include_auth:
            req_headers["Authorization"] = f"Bearer {self.api_key}"
        if body is not None:
            req_headers.setdefault("Content-Type", "application/json")
        if headers:
            req_headers.update(headers)
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        try:
            conn.request(method, path, body=body, headers=req_headers)
            resp = conn.getresponse()
            raw = resp.read()
            resp_headers = {k.lower(): v for k, v in resp.getheaders()}
            return HttpResponse(
                status=resp.status,
                headers=resp_headers,
                body=raw,
            )
        finally:
            conn.close()

    def get_health(self, *, include_auth: bool = True) -> HttpResponse:
        return self.request("GET", "/health", include_auth=include_auth)

    def get_models(self) -> HttpResponse:
        return self.request("GET", "/v1/models")

    def post_chat(
        self,
        payload: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 300.0,
    ) -> HttpResponse:
        encoded = json.dumps(payload).encode("utf-8")
        req_headers = {"Accept": "text/event-stream, application/json"}
        if headers:
            req_headers.update(headers)
        return self.request(
            "POST",
            "/v1/chat/completions",
            body=encoded,
            headers=req_headers,
            timeout=timeout,
        )

    @staticmethod
    def parse_sse_chunks(raw_body: bytes) -> list[dict[str, Any] | str]:
        """Parse SSE `data: ...` payloads into dicts or '[DONE]'."""
        events: list[dict[str, Any] | str] = []
        text = raw_body.decode("utf-8")
        for block in text.split("\n\n"):
            for line in block.splitlines():
                if line.startswith("data: "):
                    data = line[len("data: ") :].strip()
                    if data == "[DONE]":
                        events.append("[DONE]")
                    elif data:
                        events.append(json.loads(data))
        return events


def validate_chat_completion_json(
    payload: dict[str, Any], *, min_cached_tokens: int = 0
) -> int:
    """Validate OpenAI chat completion JSON structure and return cached_tokens."""
    assert payload.get("object") == "chat.completion"
    assert isinstance(payload.get("id"), str) and payload["id"].startswith("chatcmpl-")
    choices = payload.get("choices")
    assert isinstance(choices, list) and len(choices) == 1
    assert choices[0].get("finish_reason") in ("stop", "length", "tool_calls")
    usage = payload.get("usage")
    assert isinstance(usage, dict)
    details = usage.get("prompt_tokens_details")
    assert isinstance(details, dict)
    cached = int(details["cached_tokens"])
    assert cached >= min_cached_tokens
    return cached


class ServerProcess:
    """Context-managed tokspur server process with isolated state & telemetry."""

    def __init__(
        self,
        state_root: Path,
        *,
        serve_args: list[str] | None = None,
        extra_env: dict[str, str] | None = None,
        port: int | None = None,
        startup_timeout: float = 180.0,
    ) -> None:
        self.state_root = state_root
        self.serve_args = serve_args or []
        self.extra_env = extra_env or {}
        self.port = port or pick_unused_loopback_port()
        self.startup_timeout = startup_timeout
        self.proc: subprocess.Popen[str] | None = None
        self.stdout_text = ""
        self.stderr_text = ""
        self.state_dir = state_root / ".tokspur/state"
        self.stdout_log_path = state_root / "server.stdout.jsonl"
        self.stderr_log_path = state_root / "server.stderr.log"
        self._stdout_fh: Any = None
        self._stderr_fh: Any = None

    def _close_log_handles(self) -> None:
        if self._stdout_fh is not None:
            self._stdout_fh.close()
            self._stdout_fh = None
        if self._stderr_fh is not None:
            self._stderr_fh.close()
            self._stderr_fh = None
        if self.stdout_log_path.exists():
            self.stdout_text = self.stdout_log_path.read_text(encoding="utf-8")
        if self.stderr_log_path.exists():
            self.stderr_text = self.stderr_log_path.read_text(encoding="utf-8")

    def __enter__(self) -> ServerProcess:
        assert_no_competing_inference_processes()
        cli_bin = require_artifact(find_tokspur_cli(), "tokspur CLI")
        extra_serve_args = seed_isolated_state_root(
            self.state_root, serve_args=self.serve_args
        )
        cmd = [
            str(cli_bin),
            "--state-root",
            str(self.state_root),
            "serve",
            "--port",
            str(self.port),
            "--telemetry-stdout",
            "--private-prefix-trace",
            *self.serve_args,
            *extra_serve_args,
        ]
        env = os.environ.copy()
        env.update(self.extra_env)
        self._stdout_fh = self.stdout_log_path.open("w", buffering=1, encoding="utf-8")
        self._stderr_fh = self.stderr_log_path.open("w", buffering=1, encoding="utf-8")
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=self._stdout_fh,
            stderr=self._stderr_fh,
            text=True,
        )
        deadline = time.monotonic() + self.startup_timeout
        key_path = self.state_dir / "api-key"
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                rc = self.proc.returncode
                self.proc.wait(timeout=5)
                self._close_log_handles()
                self.proc = None
                raise RuntimeError(
                    f"tokspur serve exited early ({rc}): {self.stderr_text}"
                )
            if key_path.exists():
                try:
                    api_key = read_api_key(self.state_dir)
                    resp = HttpClient(self.port, api_key).get_health()
                    if resp.status == 200:
                        ready_deadline = min(deadline, time.monotonic() + 5.0)
                        while time.monotonic() < ready_deadline:
                            txt = (
                                self.stdout_log_path.read_text(encoding="utf-8")
                                if self.stdout_log_path.exists()
                                else ""
                            )
                            if '"event":"ready"' in txt or '"event": "ready"' in txt:
                                return self
                            time.sleep(0.05)
                        return self
                except (OSError, ValueError):
                    pass
            time.sleep(0.2)
        self.stop()
        raise TimeoutError(
            f"tokspur serve did not become ready within {self.startup_timeout}s: "
            f"{self.stderr_text}"
        )

    def client(self) -> HttpClient:
        return HttpClient(self.port, read_api_key(self.state_dir))

    def prefix_trace_records(self) -> list[dict[str, Any]]:
        trace_path = self.state_dir / "prefix-trace.jsonl"
        if not trace_path.exists():
            return []
        return parse_jsonl_records(trace_path.read_text(encoding="utf-8"))

    def telemetry_records(self) -> list[dict[str, Any]]:
        telem_path = self.state_dir / "telemetry.jsonl"
        if not telem_path.exists():
            return []
        return parse_jsonl_records(telem_path.read_text(encoding="utf-8"))

    def stop(self) -> tuple[str, str]:
        if self.proc is None:
            return self.stdout_text, self.stderr_text
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        else:
            self.proc.wait(timeout=5)
        self._close_log_handles()
        self.proc = None
        return self.stdout_text, self.stderr_text

    def __exit__(self, *_exc: object) -> None:
        self.stop()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _gguf_str(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _gguf_kv(key: str, val_type: int, payload: bytes) -> bytes:
    return _gguf_str(key) + struct.pack("<I", val_type) + payload


def build_synthetic_gguf_v3(
    path: Path,
    *,
    alignment: int = 16384,
    metadata: list[tuple[str, int, bytes]] | None = None,
    tensors: list[tuple[str, tuple[int, ...], int, bytes]] | None = None,
    trailing_bytes: int = 0,
) -> Path:
    """Build a minimal valid GGUF v3 binary file inside tmp_path."""
    kvs = metadata or [
        ("general.architecture", 8, _gguf_str("qwen3next")),
        ("general.alignment", 4, struct.pack("<I", alignment)),
    ]
    tensor_list = tensors or [
        ("blk.0.attn_norm.weight", (4,), 0, b"\x00\x00\x80\x3f" * 4),
    ]
    buf = bytearray()
    buf.extend(struct.pack("<IIQQ", 0x46554747, 3, len(tensor_list), len(kvs)))
    for key, vtype, payload in kvs:
        buf.extend(_gguf_kv(key, vtype, payload))

    data_offset = 0
    tensor_payloads = bytearray()
    for name, dims, ggml_type, data in tensor_list:
        buf.extend(_gguf_str(name))
        buf.extend(struct.pack("<I", len(dims)))
        for d in dims:
            buf.extend(struct.pack("<Q", d))
        buf.extend(struct.pack("<IQ", ggml_type, data_offset))
        tensor_payloads.extend(data)
        pad = (alignment - (len(data) % alignment)) % alignment
        tensor_payloads.extend(b"\x00" * pad)
        data_offset += len(data) + pad

    header_pad = (alignment - (len(buf) % alignment)) % alignment
    buf.extend(b"\x00" * header_pad)
    buf.extend(tensor_payloads)
    if trailing_bytes:
        buf.extend(b"\xab" * trailing_bytes)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(buf))
    return path


_F11_PROBE_CPP = r"""
#include "tokspur/graph.hpp"
#include "tokspur/gguf.hpp"

#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
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

template <typename Fn>
void expect_format_error(Fn&& fn, const char* msg) {
    bool threw = false;
    try {
        fn();
    } catch (const tokspur::FormatError&) {
        threw = true;
    }
    if (!threw) throw std::runtime_error(msg);
}

std::array<float, 4> read4(tokspur::MetalGraph& gpu, tokspur::GpuView v) {
    const auto raw = gpu.read(v);
    if (raw.size() != 16) throw std::runtime_error("unexpected readback size");
    std::array<float, 4> out{};
    std::memcpy(out.data(), raw.data(), 16);
    return out;
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: f11_error_flag_probe <metallib> <mode>\n";
        return 2;
    }
    const std::string_view mode(argv[2]);
    tokspur::MetalGraph gpu(argv[1], 64ull << 20);
    gpu.configure_concurrent_dispatch(true);
    Arena arena{gpu, {}};

    const std::array<float, 4> ok_vals{1.0f, 2.0f, 3.0f, 4.0f};
    const std::array<float, 4> nan_vals{std::numeric_limits<float>::quiet_NaN(), 2.0f, 3.0f, 4.0f};
    const std::array<float, 4> inf_vals{std::numeric_limits<float>::infinity(), 2.0f, 3.0f, 4.0f};
    const auto ok_in = arena.upload(ok_vals, "ok_in", true);
    const auto nan_in = arena.upload(nan_vals, "nan_in", true);
    const auto inf_in = arena.upload(inf_vals, "inf_in", true);

    if (mode == "over_16384") {
        const auto out_ok = arena.zeros(16, "out_ok");
        const auto out_bad = arena.zeros(16, "out_bad");
        gpu.begin();
        for (uint32_t i = 0; i < 16385u; ++i) {
            gpu.activate(ok_in, out_ok, tokspur::Activation::identity);
        }
        const auto r1 = gpu.submit();
        if (r1.dispatches != 16385u) throw std::runtime_error("expected 16385 dispatches");
        if (read4(gpu, out_ok) != ok_vals) throw std::runtime_error("over_16384 output mismatch");

        gpu.begin();
        for (uint32_t i = 0; i < 16384u; ++i) {
            gpu.activate(ok_in, out_ok, tokspur::Activation::identity);
        }
        gpu.activate(nan_in, out_bad, tokspur::Activation::identity);
        expect_format_error([&] { static_cast<void>(gpu.submit()); },
                            "over_16384 batch must reject 16385th nonfinite dispatch");

        gpu.begin();
        gpu.activate(ok_in, out_ok, tokspur::Activation::identity);
        const auto r3 = gpu.submit();
        if (r3.dispatches != 1u) throw std::runtime_error("expected clean recovery");
        std::cout << "PASS f11 over_16384 dispatches=16385\n";
        return 0;
    }

    if (mode == "middle_error_sticky") {
        std::vector<tokspur::GpuView> outs;
        outs.reserve(64);
        for (uint32_t i = 0; i < 64u; ++i) {
            outs.push_back(arena.zeros(16, "slot_out"));
        }
        gpu.begin();
        gpu.activate(ok_in, outs[0], tokspur::Activation::identity);
        gpu.activate(ok_in, outs[1], tokspur::Activation::identity);
        gpu.activate(nan_in, outs[2], tokspur::Activation::identity);
        gpu.activate(ok_in, outs[3], tokspur::Activation::identity);
        gpu.activate(ok_in, outs[4], tokspur::Activation::identity);
        expect_format_error([&] { static_cast<void>(gpu.submit()); },
                            "middle nonfinite dispatch must be preserved to submit()");

        gpu.begin();
        for (uint32_t i = 0; i < 64u; ++i) {
            gpu.activate(i == 37u ? inf_in : ok_in, outs[i], tokspur::Activation::identity);
        }
        expect_format_error([&] { static_cast<void>(gpu.submit()); },
                            "disjoint slot 37 inf error must be preserved to submit()");

        gpu.begin();
        for (uint32_t i = 0; i < 64u; ++i) {
            gpu.activate(ok_in, outs[i], tokspur::Activation::identity);
        }
        const auto r3 = gpu.submit();
        if (r3.dispatches != 64u) throw std::runtime_error("expected 64 clean dispatches");
        std::cout << "PASS f11 middle_error_sticky slots=64\n";
        return 0;
    }

    if (mode == "abort_clears_error") {
        const auto out0 = arena.zeros(16, "out0");
        const auto out1 = arena.zeros(16, "out1");

        gpu.begin();
        gpu.activate(nan_in, out0, tokspur::Activation::identity);
        gpu.abort();
        gpu.begin();
        gpu.activate(ok_in, out1, tokspur::Activation::identity);
        const auto r1 = gpu.submit();
        if (r1.dispatches != 1u || read4(gpu, out1) != ok_vals) {
            throw std::runtime_error("abort before commit failed to clear error_flag");
        }

        gpu.begin();
        gpu.activate(nan_in, out0, tokspur::Activation::identity);
        if (!gpu.commit_partial()) throw std::runtime_error("commit_partial returned false");
        gpu.activate(ok_in, out1, tokspur::Activation::identity);
        gpu.abort();
        gpu.begin();
        gpu.activate(ok_in, out1, tokspur::Activation::identity);
        const auto r2 = gpu.submit();
        if (r2.dispatches != 1u || read4(gpu, out1) != ok_vals) {
            throw std::runtime_error("abort after commit_partial failed to clear GPU error_flag");
        }

        gpu.begin();
        gpu.activate(nan_in, out0, tokspur::Activation::identity);
        expect_format_error([&] { static_cast<void>(gpu.submit()); }, "expected FormatError before abort test");
        gpu.begin();
        gpu.activate(nan_in, out0, tokspur::Activation::identity);
        gpu.abort();
        gpu.begin();
        gpu.activate(ok_in, out1, tokspur::Activation::identity);
        const auto r3 = gpu.submit();
        if (r3.dispatches != 1u) throw std::runtime_error("expected recovery after submit+abort");

        std::cout << "PASS f11 abort_clears_error\n";
        return 0;
    }

    std::cerr << "unknown mode: " << mode << "\n";
    return 2;
}
"""

_PLANAR_MATRIX_PROBE_CPP = r"""
#include "tokspur/graph.hpp"
#include "tokspur/planar_pack_v3.hpp"
#include "tokspur/gguf.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
struct Arena {
    tokspur::MetalGraph& gpu;
    std::vector<tokspur::GpuView> views;
    tokspur::GpuView upload(std::span<const std::byte> bytes, const char* label, bool immutable = true) {
        auto v = gpu.upload(bytes, label, immutable);
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
    if (argc != 3) {
        std::cerr << "usage: planar_matrix_probe <metallib> <mode>\n";
        return 2;
    }
    const std::string_view mode(argv[2]);
    using tokspur::planar_v3::PlanarKind;
    using tokspur::planar_v3::make_expert_spec;

    if (mode == "invalid_shapes") {
        tokspur::MetalGraph gpu(argv[1], 256ull << 20);
        Arena arena{gpu, {}};
        const auto spec = make_expert_spec(12u, 256u, 64u);
        const std::vector<std::byte> buf2(spec.planar_expert_bytes * 2u, std::byte{0});
        const std::vector<std::byte> buf3(spec.planar_expert_bytes * 3u, std::byte{0});
        const std::vector<std::byte> buf_pad(spec.planar_expert_bytes * 2u + 16u, std::byte{0});
        const auto v2 = arena.upload(buf2, "valid_2exp");
        const auto v3 = arena.upload(buf3, "valid_3exp");
        const auto v_pad = arena.upload(buf_pad, "padded_2exp");
        const uint32_t kind = uint32_t(PlanarKind::q4k_coloc_gate_v3);

        uint32_t refusals = 0;
        const auto expect_reject = [&](tokspur::GpuView view, tokspur::PlanarMatrixShape shape, const char* msg) {
            bool caught = false;
            try {
                gpu.register_planar_matrix(view, shape);
            } catch (const tokspur::FormatError&) {
                caught = true;
            }
            if (!caught) throw std::runtime_error(msg);
            ++refusals;
        };

        expect_reject(v2, {12u, kind, 0u, 64u, 2u}, "zero cols must be rejected");
        expect_reject(v2, {12u, kind, 256u, 0u, 2u}, "zero rows must be rejected");
        expect_reject(v2, {12u, kind, 256u, 6u, 2u}, "rows % 4 != 0 must be rejected");
        expect_reject(v2, {12u, kind, 128u, 64u, 2u}, "Q4_K cols % 256 != 0 must be rejected");
        expect_reject(v3, {12u, kind, 256u, 64u, 3u}, "odd colocated experts must be rejected");
        expect_reject(gpu.slice(v2, 0u, spec.planar_expert_bytes * 2u - 16u),
                      {12u, kind, 256u, 64u, 2u}, "mismatched byte span must be rejected");
        expect_reject(gpu.slice(v_pad, 4u, spec.planar_expert_bytes * 2u),
                      {12u, kind, 256u, 64u, 2u}, "non-16B offset must be rejected");
        expect_reject(v2, {12u, kind, 256u, 64u, 0u}, "zero experts must be rejected");
        std::cout << "PASS f14 invalid_shapes refusals=" << refusals << "\n";
        return 0;
    }

    if (mode == "raw_dm") {
        constexpr uint32_t channels = 256u, hidden = 64u, experts = 2u, selected = 1u;
        const auto gu_spec = make_expert_spec(12u, channels, hidden);
        const auto dn_spec = make_expert_spec(7u, hidden, channels);
        const std::vector<std::byte> gate_ok(gu_spec.planar_expert_bytes * experts, std::byte{0});
        const std::vector<std::byte> up_ok(gu_spec.planar_expert_bytes * experts, std::byte{0});
        const std::vector<std::byte> dn_ok(dn_spec.planar_expert_bytes * experts, std::byte{0});
        const std::vector<float> clean_x(channels, 0.25f);
        const std::vector<uint32_t> clean_ids{0u};

        uint32_t refusals = 0;
        for (const uint16_t bad_bits : {uint16_t(0x7e00u), uint16_t(0x7c00u), uint16_t(0xfc00u)}) {
            for (const uint32_t target : {0u, 1u, 2u}) {
                auto g_b = gate_ok;
                auto u_b = up_ok;
                auto d_b = dn_ok;
                if (target == 0u) {
                    std::memcpy(g_b.data() + gu_spec.dm_offset, &bad_bits, 2);
                } else if (target == 1u) {
                    std::memcpy(u_b.data() + gu_spec.dm_offset + 2u, &bad_bits, 2);
                } else {
                    std::memcpy(d_b.data() + dn_spec.dm_offset, &bad_bits, 2);
                }

                tokspur::MetalGraph gpu(argv[1], 256ull << 20);
                Arena arena{gpu, {}};
                const auto v_lo = arena.upload(g_b, "gate_b");
                const auto v_hi = arena.upload(u_b, "up_b");
                const auto v_dn = arena.upload(d_b, "dn_b");
                const auto vx = arena.upload(std::as_bytes(std::span(clean_x)), "x");
                const auto vix = arena.upload(std::as_bytes(std::span(clean_ids)), "ids");
                const auto vh = arena.zeros(uint64_t(selected) * hidden * 4u, "h");
                const auto vy = arena.zeros(uint64_t(selected) * channels * 4u, "y");

                bool caught = false;
                try {
                    gpu.register_planar_matrix(v_lo, {12u, uint32_t(PlanarKind::q4k_coloc_gate_v3), channels, hidden, experts});
                    gpu.register_planar_matrix(v_hi, {12u, uint32_t(PlanarKind::q4k_coloc_up_v3), channels, hidden, experts});
                    gpu.register_planar_matrix(v_dn, {7u, uint32_t(PlanarKind::q51_planar_v3), hidden, channels, experts});
                    gpu.prepare_planar_moe_v3(12u, 7u, false, 32u, experts, selected, channels, hidden);
                    gpu.begin();
                    gpu.planar_moe_routed_m8(
                        v_lo, v_hi, v_dn, vix,
                        {vx, 1u, channels, uint64_t(channels) * 4u},
                        {vh, selected, hidden, uint64_t(hidden) * 4u},
                        {vy, selected, channels, uint64_t(channels) * 4u},
                        experts, selected);
                    static_cast<void>(gpu.submit());
                } catch (const tokspur::FormatError& e) {
                    if (std::string_view(e.what()).find("nonfinite") != std::string_view::npos) {
                        caught = true;
                    }
                }
                if (!caught) {
                    throw std::runtime_error("raw-upload nonfinite dm was not rejected with nonfinite FormatError");
                }
                ++refusals;
            }
        }
        std::cout << "PASS f14 raw_dm refusals=" << refusals << "\n";
        return 0;
    }

    std::cerr << "unknown mode: " << mode << "\n";
    return 2;
}
"""

_PLANAR_DM_CPU_PROBE_CPP = r"""
#include "tokspur/planar_pack_v3.hpp"
#include "tokspur/gguf.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
template <typename Fn>
void expect_nonfinite_format_error(Fn&& fn, const char* msg) {
    bool ok = false;
    try {
        fn();
    } catch (const tokspur::FormatError& e) {
        if (std::string_view(e.what()).find("nonfinite") != std::string_view::npos) {
            ok = true;
        }
    }
    if (!ok) throw std::runtime_error(msg);
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: planar_dm_cpu_probe <nan|inf>\n";
        return 2;
    }
    const std::string_view mode(argv[1]);
    std::vector<uint16_t> bad_patterns;
    if (mode == "nan") {
        bad_patterns = {uint16_t(0x7e00u)};
    } else if (mode == "inf") {
        bad_patterns = {uint16_t(0x7c00u), uint16_t(0xfc00u)};
    } else {
        std::cerr << "unknown mode: " << mode << "\n";
        return 2;
    }

    using tokspur::planar_v3::dequantize_expert_planar_v3;
    using tokspur::planar_v3::make_expert_spec;
    using tokspur::planar_v3::pack_expert_planar_v3;

    for (const uint32_t st : {12u, 7u, 8u}) {
        const uint32_t cols = (st == 12u) ? 256u : 32u;
        const uint32_t rows = (st == 8u) ? 8u : 4u;
        const auto spec = make_expert_spec(st, cols, rows);
        const std::vector<std::byte> gguf_ok(spec.source_expert_bytes, std::byte{0});
        std::vector<std::byte> planar_ok(spec.planar_expert_bytes, std::byte{0});
        pack_expert_planar_v3(spec, gguf_ok, planar_ok);
        std::vector<float> out_ok(size_t(cols) * rows, 0.0f);
        dequantize_expert_planar_v3(spec, planar_ok, out_ok);

        for (const uint16_t bad : bad_patterns) {
            const std::vector<uint64_t> offsets = (st == 8u) ? std::vector<uint64_t>{0u} : std::vector<uint64_t>{0u, 2u};
            for (const uint64_t off : offsets) {
                auto gguf_bad = gguf_ok;
                std::memcpy(gguf_bad.data() + off, &bad, 2);
                std::vector<std::byte> planar_scratch(spec.planar_expert_bytes, std::byte{0});
                expect_nonfinite_format_error(
                    [&] { pack_expert_planar_v3(spec, gguf_bad, planar_scratch); },
                    "pack_expert_planar_v3 must reject nonfinite dm");

                auto planar_bad = planar_ok;
                std::memcpy(planar_bad.data() + spec.dm_offset + off, &bad, 2);
                std::vector<float> out_bad(size_t(cols) * rows, 0.0f);
                expect_nonfinite_format_error(
                    [&] { dequantize_expert_planar_v3(spec, planar_bad, out_bad); },
                    "dequantize_expert_planar_v3 must reject nonfinite dm");
            }
        }
    }
    std::cout << "PASS f13 planar_dm_cpu " << mode << "\n";
    return 0;
}
"""


def _compile_native_probe_from_target(
    tmp_dir: Path,
    name: str,
    cpp_source: str,
    *,
    build_dir: Path,
    target_name: str,
    target_cpp_rel: str,
    extra_cxxflags: str | None = None,
) -> Path:
    """Compile a standalone C++ probe in tmp_dir using native/Makefile's link line."""
    probe_dir = tmp_dir / f"probe_{name}"
    probe_dir.mkdir(parents=True, exist_ok=True)
    driver_cpp = probe_dir / f"{name}.cpp"
    driver_bin = probe_dir / name
    driver_cpp.write_text(cpp_source, encoding="utf-8")

    target_path = build_dir / target_name
    make_args = [f"BUILD={build_dir}", str(target_path)]
    if extra_cxxflags:
        make_args.append(f"CXXFLAGS={extra_cxxflags}")

    if not target_path.exists() or extra_cxxflags:
        mk_res = run_make_native(["-j4", *make_args])
        assert mk_res.returncode == 0, mk_res.stderr + mk_res.stdout

    dry = run_make_native(["-n", *make_args])
    assert dry.returncode == 0, dry.stderr + dry.stdout
    link_line: str | None = None
    for raw in dry.stdout.splitlines():
        s = raw.strip()
        if target_cpp_rel in s and " -o " in s:
            link_line = s
    assert link_line is not None, (
        f"Could not find link command for {target_path} in make -n output"
    )
    tokens = shlex.split(link_line)
    replaced_src = False
    replaced_out = False
    out_tokens: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == target_cpp_rel:
            out_tokens.append(str(driver_cpp))
            replaced_src = True
        elif tok == "-o" and i + 1 < len(tokens):
            out_tokens.extend(["-o", str(driver_bin)])
            replaced_out = True
            i += 1
        else:
            out_tokens.append(tok)
        i += 1
    assert replaced_src and replaced_out, (
        f"Unexpected link tokens for {target_path}: {tokens}"
    )
    comp_res = run_cmd(out_tokens, cwd=NATIVE_DIR, timeout=180.0)
    assert comp_res.returncode == 0, (
        f"Probe {name} compile/link failed:\n{comp_res.stderr}\n{comp_res.stdout}"
    )
    assert driver_bin.exists(), f"Probe binary missing after link: {driver_bin}"
    return driver_bin


def build_metal_graph_probe(
    tmp_dir: Path,
    name: str,
    cpp_source: str,
    *,
    dispatch_graph_trace: bool = False,
) -> Path:
    """Compile a standalone MetalGraph probe binary inside tmp_dir."""
    build_dir = (
        (tmp_dir / "diag-build")
        if dispatch_graph_trace
        else (BUILD_DIR / "owned-native")
    )
    extra_cxxflags = (
        "-std=c++20 -O2 -g -Wall -Wextra -Wpedantic -Werror "
        "-DTOKSPUR_PRIVATE_DISPATCH_GRAPH=1"
        if dispatch_graph_trace
        else None
    )
    return _compile_native_probe_from_target(
        tmp_dir,
        name,
        cpp_source,
        build_dir=build_dir,
        target_name="test-graph",
        target_cpp_rel="tests/test_graph.cpp",
        extra_cxxflags=extra_cxxflags,
    )


def build_f11_metal_graph_probe(tmp_path: Path) -> Path:
    """Compile the F11 MetalGraph error_flag probe binary inside tmp_path."""
    return build_metal_graph_probe(
        tmp_path, "f11_error_flag_probe", _F11_PROBE_CPP, dispatch_graph_trace=False
    )


def build_planar_matrix_probe(tmp_path: Path) -> Path:
    """Compile the F14 MetalGraph::register_planar_matrix probe inside tmp_path."""
    return build_metal_graph_probe(
        tmp_path,
        "planar_matrix_probe",
        _PLANAR_MATRIX_PROBE_CPP,
        dispatch_graph_trace=False,
    )


def build_planar_dm_cpu_probe(tmp_path: Path) -> Path:
    """Compile the CPU-only Pack v3 dm finiteness probe inside tmp_path."""
    return _compile_native_probe_from_target(
        tmp_path,
        "planar_dm_cpu_probe",
        _PLANAR_DM_CPU_PROBE_CPP,
        build_dir=BUILD_DIR / "owned-native",
        target_name="test-derived_pack",
        target_cpp_rel="tests/test_derived_pack.cpp",
    )


def tamper_file_bit(path: Path, offset: int, xor_mask: int = 0x01) -> None:
    """Flip bits at byte offset in a file (handles 0444 by temp chmod)."""
    mode = path.stat().st_mode & 0o777
    os.chmod(path, 0o600)
    try:
        data = bytearray(path.read_bytes())
        data[offset] ^= xor_mask
        path.write_bytes(bytes(data))
    finally:
        os.chmod(path, mode)


class DispatchTraceBuilder:
    """Builder for schema-1 tokspur-dispatch-graph JSONL traces."""

    def __init__(self, producer: str = "tokspur-cpp-metal-graph") -> None:
        self.lines: list[dict[str, Any]] = [
            {
                "event": "trace",
                "schema": 1,
                "producer": producer,
                "device": "Apple M5 Max",
            }
        ]
        self._batch = 0

    def add_pipeline(
        self,
        key: str,
        function: str,
        bindings: list[tuple[int, str, str, bool]],
        threads: int = 256,
    ) -> None:
        self.lines.append(
            {
                "event": "pipeline",
                "key": key,
                "function": function,
                "constant": None,
                "threads": threads,
                "bindings": [
                    {"index": idx, "name": name, "access": access, "used": used}
                    for idx, name, access, used in bindings
                ],
            }
        )

    def add_allocation(
        self,
        slot: int,
        label: str,
        size_bytes: int,
        *,
        immutable: bool = False,
        kind: str = "buffer",
    ) -> None:
        self.lines.append(
            {
                "event": "allocation",
                "slot": slot,
                "label": label,
                "kind": kind,
                "bytes": size_bytes,
                "immutable": immutable,
                "sha256": sha256_bytes(label.encode()) if immutable else None,
            }
        )

    def add_host_write(self, slot: int, offset: int, size_bytes: int, tag: str) -> None:
        self.lines.append(
            {
                "event": "host_write",
                "slot": slot,
                "offset": offset,
                "bytes": size_bytes,
                "sha256": sha256_bytes(tag.encode()),
            }
        )

    def add_host_read(self, slot: int, offset: int, size_bytes: int, tag: str) -> None:
        self.lines.append(
            {
                "event": "host_read",
                "slot": slot,
                "offset": offset,
                "bytes": size_bytes,
                "sha256": sha256_bytes(tag.encode()),
            }
        )

    def begin_batch(self) -> int:
        batch_id = self._batch
        self._batch += 1
        self.lines.append({"event": "begin", "batch": batch_id})
        return batch_id

    def add_dispatch(
        self,
        pipeline: str,
        bindings: list[dict[str, Any]],
        *,
        grid: tuple[int, int, int] = (1, 1, 1),
        group: tuple[int, int, int] = (64, 1, 1),
        aligned: bool = True,
        constants: list[dict[str, Any]] | None = None,
    ) -> None:
        self.lines.append(
            {
                "event": "dispatch",
                "pipeline": pipeline,
                "aligned": aligned,
                "grid": list(grid),
                "group": list(group),
                "bindings": bindings,
                "constants": constants or [],
            }
        )

    def add_barrier(self) -> None:
        self.lines.append({"event": "barrier", "scope": "buffers"})

    def submit_batch(
        self,
        batch_id: int,
        dispatches: int,
        states: list[tuple[int, str]] | None = None,
    ) -> None:
        self.lines.append(
            {"event": "submit", "batch": batch_id, "dispatches": dispatches}
        )
        for slot, tag in states or []:
            self.lines.append(
                {"event": "state", "slot": slot, "sha256": sha256_bytes(tag.encode())}
            )

    def abort_batch(self, batch_id: int) -> None:
        self.lines.append({"event": "abort", "batch": batch_id})

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(item) for item in self.lines) + "\n"

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_jsonl(), encoding="utf-8")
        return path


def build_dag_pair_traces(tmp_dir: Path, *, depth: int = 3) -> tuple[Path, Path]:
    """Build diamond-DAG reference (serial barriers) and Phase B (depth-1) traces."""
    if depth < 2:
        raise ValueError(f"diamond DAG requires depth >= 2, got {depth}")
    ref = DispatchTraceBuilder()
    cand = DispatchTraceBuilder()
    total_slots = depth + 2
    for b in (ref, cand):
        b.add_pipeline(
            "rms_norm:1",
            "rms_norm",
            [
                (0, "buf", "read", True),
                (1, "aux", "read", True),
                (2, "out", "write", True),
            ],
        )
        for slot in range(total_slots):
            b.add_allocation(slot, f"s{slot}", 4096)
            b.add_host_write(slot, 0, 4096, f"init_{slot}")

    bid_r = ref.begin_batch()
    bid_c = cand.begin_batch()

    def _const(tag: str) -> list[dict[str, Any]]:
        return [{"index": 3, "bytes": 4, "sha256": sha256_bytes(tag.encode())}]

    # Level 0: 2 parallel dispatches (s0 -> s1) and (s0 -> s2) => widest_level == 2
    d0_bindings = [
        {"index": 0, "slot": 0, "offset": 0, "bytes": 4096, "access": "read"},
        {"index": 2, "slot": 1, "offset": 0, "bytes": 4096, "access": "write"},
        {"index": 7, "buffer": "error_flag"},
    ]
    d1_bindings = [
        {"index": 0, "slot": 0, "offset": 0, "bytes": 4096, "access": "read"},
        {"index": 2, "slot": 2, "offset": 0, "bytes": 4096, "access": "write"},
        {"index": 7, "buffer": "error_flag"},
    ]
    for builder in (ref, cand):
        builder.add_dispatch("rms_norm:1", d0_bindings, constants=_const("lvl0_a"))
    ref.add_barrier()
    for builder in (ref, cand):
        builder.add_dispatch("rms_norm:1", d1_bindings, constants=_const("lvl0_b"))
    ref.add_barrier()
    cand.add_barrier()

    # Level 1: fan-in join (s1, s2 -> s3)
    join_bindings = [
        {"index": 0, "slot": 1, "offset": 0, "bytes": 4096, "access": "read"},
        {"index": 1, "slot": 2, "offset": 0, "bytes": 4096, "access": "read"},
        {"index": 2, "slot": 3, "offset": 0, "bytes": 4096, "access": "write"},
        {"index": 7, "buffer": "error_flag"},
    ]
    for builder in (ref, cand):
        builder.add_dispatch("rms_norm:1", join_bindings, constants=_const("lvl1_join"))
    if depth > 2:
        ref.add_barrier()
        cand.add_barrier()

    # Levels 2..depth-1: serial chain (s_{lvl+1} -> s_{lvl+2})
    for lvl in range(2, depth):
        chain_bindings = [
            {
                "index": 0,
                "slot": lvl + 1,
                "offset": 0,
                "bytes": 4096,
                "access": "read",
            },
            {
                "index": 2,
                "slot": lvl + 2,
                "offset": 0,
                "bytes": 4096,
                "access": "write",
            },
            {"index": 7, "buffer": "error_flag"},
        ]
        for builder in (ref, cand):
            builder.add_dispatch(
                "rms_norm:1", chain_bindings, constants=_const(f"lvl{lvl}")
            )
        if lvl + 1 < depth:
            ref.add_barrier()
            cand.add_barrier()

    total_dispatches = depth + 1
    states = [(s, f"final_{s}") for s in range(total_slots)]
    ref.submit_batch(bid_r, total_dispatches, states)
    cand.submit_batch(bid_c, total_dispatches, states)
    ref_path = ref.write(tmp_dir / "ref.trace.jsonl")
    cand_path = cand.write(tmp_dir / "cand.trace.jsonl")
    return ref_path, cand_path


EPISTEMIC_VERIFICATION_HEADER = (
    "VERIFICATION ONLY: Deductive/Numerical Consistency Under Declared Assumptions; "
    "NOT Empirical Validation (Roache 1998; Oberkampf & Roy 2010)."
)


def build_holdout_manifest(
    base_dir: Path,
    *,
    pre_reg_id: str = "prereg-2026-01",
    num_holdouts: int = 2,
    max_relative_l2_error: float = 0.02,
    leak_calibration: bool = False,
    reference_error: float = 0.005,
) -> dict[str, Any]:
    """Create a Track 4 schema-compliant empirical_vv_manifest.v1 + splits."""
    base_dir.mkdir(parents=True, exist_ok=True)
    cal_file = base_dir / "cal_0.json"

    def wave_split(split_id: str, role: str, amplitude: float, error: float) -> bytes:
        # A single sine mode after one finite-difference step has this amplification factor.
        nx, cfl = 8, 0.5
        factor = 1.0 - 2.0 * cfl**2 * math.sin(math.pi / (2 * nx)) ** 2
        reference = [
            0.0,
            *(
                amplitude * math.sin(math.pi * j / nx) * factor / (1 + error)
                for j in range(1, nx)
            ),
            0.0,
        ]
        case = {
            "case_id": split_id,
            "wave_speed_c": 1.0,
            "domain_length_l": 1.0,
            "nx": nx,
            "nt": 1,
            "cfl": cfl,
            "modes": [{"mode_k": 1, "amplitude": amplitude}],
            "reference_u_final": reference,
        }
        return json.dumps(
            {"split_id": split_id, "role": role, "domain": "wave1d", "cases": [case]}
        ).encode()

    cal_bytes = wave_split("cal_0", "calibration", 1.0, 0.0)
    cal_file.write_bytes(cal_bytes)
    cal_sha = sha256_bytes(cal_bytes)

    calibration_splits = [
        {
            "split_id": "cal_0",
            "relative_path": "cal_0.json",
            "expected_sha256": cal_sha,
        }
    ]
    holdout_splits = []
    for idx in range(num_holdouts):
        hid = f"holdout_{idx}"
        hfile = base_dir / f"{hid}.json"
        if leak_calibration and idx == 0:
            hbytes = cal_bytes
        else:
            hbytes = wave_split(hid, "holdout", 2.0 + idx, reference_error)
        hfile.write_bytes(hbytes)
        hsha = sha256_bytes(hbytes)
        decimal = f"{max_relative_l2_error:.6f}"
        if float(decimal) == max_relative_l2_error:
            lock_input = (
                f"track4.vv.lock.v1|{pre_reg_id}|{hid}|{hsha}|{decimal}".encode()
            )
        else:
            bits = struct.pack(">d", max_relative_l2_error).hex()
            lock_input = f"track4.vv.lock.v2|{pre_reg_id}|{hid}|{hsha}|{bits}".encode()
        holdout_splits.append(
            {
                "holdout_id": hid,
                "relative_path": f"{hid}.json",
                "expected_sha256": hsha,
                "max_relative_l2_error": max_relative_l2_error,
                "single_use": True,
                "pre_registered_lock_commitment": sha256_bytes(lock_input),
            }
        )

    manifest = {
        "manifest_version": "1.0.0",
        "schema_id": "track4.empirical_vv_manifest.v1",
        "pre_registration_id": pre_reg_id,
        "created_utc": "2026-11-21T19:00:00Z",
        "epistemic_header": EPISTEMIC_VERIFICATION_HEADER,
        "split_provenance": {
            "partition_strategy": "stratified-deterministic",
            "calibration_regime": "synthetic-calibration",
            "holdout_regime": "single-use-locked-holdout",
            "zero_leakage_policy": "sha256-disjoint-calibration-holdout",
        },
        "calibration_splits": calibration_splits,
        "holdout_splits": holdout_splits,
    }
    manifest_path = base_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "manifest_path": manifest_path,
        "ledger_path": base_dir / "ledger.json",
        "manifest": manifest,
    }


def build_paired_block_series(kind: str) -> list[dict[str, Any]]:
    """Return deterministic paired-block metrics for R5 Wilcoxon/BH-FDR tests."""
    if kind == "loser_2block":
        return [
            {
                "block": 1,
                "baseline": [10.0, 10.2, 9.8, 10.1, 9.9, 10.3],
                "candidate": [12.5, 12.8, 12.2, 12.6, 12.4, 12.9],
            },
            {
                "block": 2,
                "baseline": [10.0, 10.1, 9.9, 10.2, 10.0, 10.1],
                "candidate": [12.6, 12.7, 12.5, 12.8, 12.4, 12.7],
            },
        ]
    if kind == "winner_2block":
        return [
            {
                "block": 1,
                "baseline": [12.0, 12.1, 11.9, 12.2, 12.0, 12.3],
                "candidate": [9.5, 9.4, 9.6, 9.5, 9.3, 9.7],
            },
            {
                "block": 2,
                "baseline": [12.1, 12.0, 12.2, 11.9, 12.1, 12.0],
                "candidate": [9.4, 9.5, 9.3, 9.6, 9.4, 9.5],
            },
        ]
    if kind == "exact_tie":
        return [
            {
                "block": 1,
                "baseline": [10.0, 10.0, 10.0, 10.0],
                "candidate": [10.0, 10.0, 10.0, 10.0],
            },
            {
                "block": 2,
                "baseline": [10.0, 10.0, 10.0, 10.0],
                "candidate": [10.0, 10.0, 10.0, 10.0],
            },
        ]
    if kind == "single_block_only":
        return [
            {
                "block": 1,
                "baseline": [10.0, 10.1, 10.2, 9.9, 10.0, 10.1],
                "candidate": [12.0, 12.1, 12.2, 12.0, 12.1, 12.3],
            },
        ]
    if kind == "borderline_nonfutile":
        return [
            {
                "block": 1,
                "baseline": [12.0, 12.0, 12.0, 12.0, 12.0, 12.0],
                "candidate": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            },
            {
                "block": 2,
                "baseline": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                "candidate": [10.1, 9.9, 10.0, 10.1, 9.9, 10.0],
            },
        ]
    raise ValueError(f"Unknown paired-block series kind: {kind}")


def reference_exact_wilcoxon_signed_rank(
    diffs: list[float],
) -> tuple[float, float, float, float, int]:
    """Pure-Python reference for exact two-sided Wilcoxon signed-rank test."""
    nonzero = [d for d in diffs if abs(d) > 1e-12]
    n = len(nonzero)
    if n == 0:
        return (0.0, 0.0, 0.0, 1.0, 0)
    ordered = sorted((abs(d), d > 0.0) for d in nonzero)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs(ordered[j][0] - ordered[i][0]) <= 1e-12:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    w_pos = sum(ranks[idx] for idx, (_, pos) in enumerate(ordered) if pos)
    w_neg = sum(ranks[idx] for idx, (_, pos) in enumerate(ordered) if not pos)
    w_min = min(w_pos, w_neg)
    total_patterns = 1 << n
    extreme = 0
    for mask in range(total_patterns):
        w_m = sum(ranks[bit] for bit in range(n) if (mask >> bit) & 1)
        if min(w_m, (n * (n + 1) / 2.0) - w_m) <= w_min + 1e-12:
            extreme += 1
    p_value = min(1.0, extreme / total_patterns)
    return (w_pos, w_neg, w_min, p_value, n)


def reference_apply_bh_fdr(
    p_values: list[float], alpha: float
) -> list[tuple[float, bool]]:
    """Pure-Python reference for Benjamini-Hochberg step-up adjusted q-values."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"Invalid BH-FDR alpha: {alpha}")
    for p in p_values:
        if math.isnan(p) or p < 0.0 or p > 1.0:
            raise ValueError(f"Invalid p-value: {p}")
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda item: (item[1], item[0]))
    q_sorted = [0.0] * m
    running = 1.0
    for rev_idx in range(m - 1, -1, -1):
        rank = rev_idx + 1
        raw_q = min(1.0, m * indexed[rev_idx][1] / rank)
        running = min(running, raw_q)
        q_sorted[rev_idx] = running
    out = [(0.0, False)] * m
    for pos, (orig_idx, _) in enumerate(indexed):
        q = q_sorted[pos]
        out[orig_idx] = (q, q <= alpha)
    return out


def find_rust_module(filename: str) -> Path | None:
    """Locate an R5 Rust source module under rust/crates/."""
    matches = sorted((RUST_DIR / "crates").glob(f"*/src/**/{filename}"))
    if matches:
        return matches[0]
    direct = sorted((RUST_DIR / "crates").glob(f"*/src/{filename}"))
    return direct[0] if direct else None


def discover_r5_crate(kind: Literal["holdout", "futility"]) -> tuple[str, Path] | None:
    """Discover the workspace crate package name and root path containing R5 modules."""
    cargo_toml = tomllib.loads((RUST_DIR / "Cargo.toml").read_text(encoding="utf-8"))
    members = cargo_toml.get("workspace", {}).get("members", [])
    targets = (
        ("holdout.rs",)
        if kind == "holdout"
        else ("futility.rs", "transfer.rs", "ci_tests.rs")
    )
    for rel in members:
        crate_dir = RUST_DIR / rel
        src_dir = crate_dir / "src"
        if not src_dir.is_dir():
            continue
        for fname in targets:
            if (src_dir / fname).is_file() or list(src_dir.rglob(fname)):
                pkg_meta = tomllib.loads(
                    (crate_dir / "Cargo.toml").read_text(encoding="utf-8")
                )
                pkg_name = pkg_meta.get("package", {}).get("name")
                if pkg_name:
                    return str(pkg_name), crate_dir
    return None


def list_crate_tests(crate_name: str) -> list[str]:
    """List test names in a Rust workspace crate via `cargo test -- --list`."""
    res = run_cargo_test(["-p", crate_name, "--", "--list"])
    assert res.returncode == 0, f"cargo test --list failed: {res.stderr}\n{res.stdout}"
    tests: list[str] = []
    for line in res.stdout.splitlines():
        s = line.strip()
        if s.endswith(": test"):
            tests.append(s[: -len(": test")])
    return tests


def discover_r5_cli(kind: Literal["holdout", "futility"]) -> list[str] | None:
    """Locate a compiled R5 CLI binary or cargo bin target for holdout/futility."""
    hint = "tokspur-holdout" if kind == "holdout" else "tokspur-futility"
    for candidate in ("tokspur-eval", hint, "tokspur-r5"):
        bin_path = find_r5_cli_bin(candidate)
        if bin_path is not None:
            return [str(bin_path)]
    discovered = discover_r5_crate(kind)
    if discovered is not None:
        pkg_name, crate_dir = discovered
        crate_toml = tomllib.loads(
            (crate_dir / "Cargo.toml").read_text(encoding="utf-8")
        )
        bins = [b.get("name") for b in crate_toml.get("bin", []) if b.get("name")]
        if not bins:
            bins = [p.stem for p in sorted((crate_dir / "src/bin").glob("*.rs"))]
        if bins:
            chosen = next((b for b in bins if kind in b or "eval" in b), bins[0])
            return [
                "cargo",
                "run",
                "--frozen",
                "-q",
                "-p",
                pkg_name,
                "--bin",
                chosen,
                "--",
            ]
    tokspur_bin = find_tokspur_cli()
    if tokspur_bin is not None:
        help_res = run_cmd([str(tokspur_bin), "--help"], timeout=15.0)
        if help_res.returncode == 0 and re.search(rf"\b{kind}\b", help_res.stdout):
            return [str(tokspur_bin), kind]
    return None


def _run_r5_driver(
    tmp_dir: Path,
    pkg_name: str,
    crate_dir: Path,
    main_rs: str,
    driver_args: list[str],
) -> CmdResult:
    """Build and run a standalone offline Rust driver crate in tmp_dir."""
    driver_dir = tmp_dir / f"driver_{pkg_name.replace('-', '_')}"
    src_dir = driver_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    ident = pkg_name.replace("-", "_")
    (driver_dir / "Cargo.toml").write_text(
        f'[package]\nname = "e2e-r5-driver"\nversion = "0.1.0"\n'
        f'edition = "2024"\n\n[workspace]\n\n[dependencies]\n'
        f'{pkg_name} = {{ path = "{crate_dir}" }}\n'
        f'tokspur-json = {{ path = "{RUST_DIR / "crates/tokspur-json"}" }}\n',
        encoding="utf-8",
    )
    (src_dir / "main.rs").write_text(
        main_rs.replace("__CRATE__", ident), encoding="utf-8"
    )
    env = {"RUSTUP_TOOLCHAIN": "1.99.1"}
    return run_cmd(
        [
            "cargo",
            "run",
            "--offline",
            "--manifest-path",
            str(driver_dir / "Cargo.toml"),
            "--target-dir",
            str(driver_dir / "target"),
            "--",
            *driver_args,
        ],
        cwd=driver_dir,
        env=env,
        timeout=300.0,
    )


def run_r5_holdout_eval(
    tmp_dir: Path,
    manifest_dir: Path,
    holdout_id: str,
    ledger_path: Path,
) -> CmdResult:
    """Execute R5 holdout evaluation against a tmp_path manifest and ledger."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cli = discover_r5_cli("holdout")
    args = [
        "holdout-check",
        str(manifest_dir),
        holdout_id,
        str(ledger_path),
    ]
    if cli is not None:
        cwd = RUST_DIR if cli[0] == "cargo" else REPO_ROOT
        return run_cmd([*cli, *args], cwd=cwd, timeout=180.0)
    discovered = discover_r5_crate("holdout")
    if discovered is None:
        require_artifact(find_rust_module("holdout.rs"), "rust/crates/*/src/holdout.rs")
        raise AssertionError("holdout crate not found")
    pkg_name, crate_dir = discovered
    main_rs = (
        "use std::path::Path;\n"
        "fn main() {\n"
        "    let args: Vec<String> = std::env::args().collect();\n"
        "    let mdir = Path::new(&args[2]);\n"
        "    let hid = &args[3];\n"
        "    let ledger = Path::new(&args[4]);\n"
        "    match __CRATE__::holdout::unlock_and_evaluate_holdout(mdir, hid, ledger) {\n"
        '        Ok(o) => println!("{{\\"holdout_id\\":\\"{}\\",'
        '\\"consumption_token\\":\\"{}\\",'
        '\\"zero_calibration_leakage\\":{}}}", '
        "o.holdout_id, o.consumption_token, o.zero_calibration_leakage),\n"
        '        Err(err) => { eprintln!("{err}"); std::process::exit(1); }\n'
        "    }\n"
        "}\n"
    )
    return _run_r5_driver(tmp_dir, pkg_name, crate_dir, main_rs, args)


def run_r5_futility_eval(tmp_dir: Path, payload: dict[str, Any]) -> CmdResult:
    """Execute R5 Wilcoxon/BH-FDR/futility evaluation against a tmp_path JSON."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    mode = payload.get("mode", "futility")
    if mode in ("futility", "wilcoxon"):
        cli = discover_r5_cli("futility")
        if cli is None:
            found = (
                find_rust_module("futility.rs")
                or find_rust_module("transfer.rs")
                or find_rust_module("ci_tests.rs")
            )
            require_artifact(found, "R5 futility/transfer/ci_tests Rust module")
            raise AssertionError("futility CLI not found")
        if mode == "futility":
            sweep = {
                "min_blocks": int(payload.get("min_blocks", 2)),
                "fdr_alpha": float(payload.get("alpha", 0.05)),
                "min_worthwhile_gain_pct": float(
                    payload.get("min_worthwhile_gain_pct", 0.0)
                ),
                "candidates": payload.get("candidates")
                or [
                    {
                        "candidate_id": "cand_0",
                        "direction": payload.get("direction", "lower_is_better"),
                        "blocks": payload["blocks"],
                    }
                ],
            }
        else:
            diffs = [float(x) for x in payload["diffs"]]
            sweep = {
                "min_blocks": 2,
                "fdr_alpha": 0.05,
                "min_worthwhile_gain_pct": 0.0,
                "candidates": [
                    {
                        "candidate_id": "wilcoxon_0",
                        "direction": "higher_is_better",
                        "blocks": [
                            {
                                "block": 1,
                                "baseline": [0.0] * len(diffs),
                                "candidate": diffs,
                            }
                        ],
                    }
                ],
            }
        input_path = tmp_dir / "futility_sweep.json"
        input_path.write_text(json.dumps(sweep, indent=2), encoding="utf-8")
        cwd = RUST_DIR if cli[0] == "cargo" else REPO_ROOT
        res = run_cmd([*cli, "futility-eval", str(input_path)], cwd=cwd, timeout=180.0)
        if res.returncode != 0:
            return res
        raw = json.loads(res.stdout)
        cand = raw["candidates"][0]
        normalized = {
            "any_stopped": raw["any_stopped"],
            "all_stopped": raw["all_stopped"],
            "stop_for_futility": cand["verdict"] == "stop_futile",
            "verdict": cand["verdict"],
            "continue_reason": cand["continue_reason"],
            "blocks_evaluated": cand["blocks_observed"],
            "n_eff": cand["n_effective"],
            "w_pos": cand["w_pos"],
            "w_neg": cand["w_neg"],
            "w_min": cand["w_min"],
            "p_value": cand["p_two_sided"],
            "p_futility": cand["p_futility"],
            "q_futility": cand["q_futility"],
        }
        return CmdResult(
            argv=res.argv,
            returncode=res.returncode,
            stdout=json.dumps(normalized),
            stderr=res.stderr,
            duration_s=res.duration_s,
        )

    discovered = discover_r5_crate("futility")
    if discovered is None:
        found = (
            find_rust_module("ci_tests.rs")
            or find_rust_module("futility.rs")
            or find_rust_module("transfer.rs")
        )
        require_artifact(found, "R5 ci_tests/futility/transfer Rust module")
        raise AssertionError("futility crate not found")
    pkg_name, crate_dir = discovered
    alpha_str = str(payload["alpha"])
    p_strs = [str(p) for p in payload["p_values"]]
    main_rs = (
        "fn main() {\n"
        "    let args: Vec<String> = std::env::args().collect();\n"
        "    let alpha: f64 = match args.get(1).and_then(|s| s.parse().ok()) {\n"
        "        Some(v) => v,\n"
        '        None => { eprintln!("invalid alpha"); std::process::exit(1); }\n'
        "    };\n"
        "    let pvals: Vec<f64> = args[2..]\n"
        "        .iter()\n"
        "        .map(|s| s.parse().unwrap_or(f64::NAN))\n"
        "        .collect();\n"
        "    match __CRATE__::ci_tests::apply_bh_fdr_pvalues(&pvals, alpha) {\n"
        "        Ok(res) => {\n"
        '            let qs: Vec<String> = res.iter().map(|(q, _)| format!("{q:.16}")).collect();\n'
        '            let rs: Vec<String> = res.iter().map(|(_, r)| format!("{r}")).collect();\n'
        '            println!("{{\\"q_values\\":[{}],\\"rejected\\":[{}]}}", qs.join(","), rs.join(","));\n'
        "        }\n"
        '        Err(err) => { eprintln!("{err}"); std::process::exit(1); }\n'
        "    }\n"
        "}\n"
    )
    return _run_r5_driver(tmp_dir, pkg_name, crate_dir, main_rs, [alpha_str, *p_strs])


def run_r5_holdout_suite(filter_expr: str | None = None) -> CmdResult:
    """Verify holdout.rs exists and run its checked cargo test suite."""
    discovered = discover_r5_crate("holdout")
    if discovered is None:
        require_artifact(find_rust_module("holdout.rs"), "rust/crates/*/src/holdout.rs")
        raise AssertionError("holdout crate not found")
    pkg_name, _ = discovered
    cargo_args = ["-p", pkg_name]
    if filter_expr:
        available = list_crate_tests(pkg_name)
        assert any(filter_expr in t for t in available), (
            f"No test matching '{filter_expr}' in crate {pkg_name}: {available}"
        )
        cargo_args.extend(["--", filter_expr])
    return run_cargo_test_checked(cargo_args, min_passed=1)


def run_r5_futility_suite(filter_expr: str | None = None) -> CmdResult:
    """Verify R5 futility/transfer/ci_tests exist and pass checked cargo test."""
    discovered = discover_r5_crate("futility")
    if discovered is None:
        found = (
            find_rust_module("futility.rs")
            or find_rust_module("transfer.rs")
            or find_rust_module("ci_tests.rs")
        )
        require_artifact(found, "R5 futility/transfer/ci_tests Rust module")
        raise AssertionError("futility crate not found")
    pkg_name, _ = discovered
    cargo_args = ["-p", pkg_name]
    if filter_expr:
        available = list_crate_tests(pkg_name)
        assert any(filter_expr in t for t in available), (
            f"No test matching '{filter_expr}' in crate {pkg_name}: {available}"
        )
        cargo_args.extend(["--", filter_expr])
    return run_cargo_test_checked(cargo_args, min_passed=1)


def tamper_holdout_ledger(ledger_path: Path, field: str = "record_chain_hash") -> None:
    """Mutate a field in an existing HoldoutLedger JSON file to test tamper checks."""
    data = json.loads(ledger_path.read_text(encoding="utf-8"))
    entries = data.get("entries") or []
    if not entries:
        raise ValueError(f"No entries to tamper in {ledger_path}")
    val = str(entries[-1].get(field, "0" * 64))
    flipped = ("0" if val.startswith("1") else "1") + val[1:]
    entries[-1][field] = flipped
    ledger_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def make_chat_payload(
    prompt: str,
    *,
    history: list[dict[str, Any]] | None = None,
    model: str = "tokspur",
    enable_thinking: bool = True,
    seed: int | None = 9983,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 1,
    min_p: float = 0.0,
    max_tokens: int = 48,
    stream: bool = False,
    include_usage: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an OpenAI-compatible POST /v1/chat/completions JSON body."""
    messages = [*(history or []), {"role": "user", "content": prompt}]
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "max_tokens": max_tokens,
        "enable_thinking": enable_thinking,
        "stream": stream,
    }
    if seed is not None:
        body["seed"] = seed
    if stream and include_usage:
        body["stream_options"] = {"include_usage": True}
    if extra:
        body.update(extra)
    return body


def make_long_repeated_prompt(label: str, approx_lines: int = 160) -> str:
    """Generate deterministic multi-line prompt exceeding parked-lane threshold."""
    lines = [f"Context header for {label}: analyze every numbered record carefully."]
    for i in range(approx_lines):
        lines.append(
            f"Record {i:04d}: module=tokspur_lane_{label} step={(i * 7) % 97} "
            f"checksum={(i * 13 + 5) % 251} status=nominal."
        )
    lines.append(f"Question for {label}: summarize the checksum recurrence concisely.")
    return "\n".join(lines)


def list_metallib_symbols(metallib: Path) -> set[str]:
    """List defined kernel function names via xcrun metal-nm."""
    res = run_cmd(
        ["xcrun", "metal-nm", "--defined-only", "--just-symbol-name", str(metallib)]
    )
    assert res.returncode == 0, f"metal-nm failed: {res.stderr}"
    return {line.strip() for line in res.stdout.splitlines() if line.strip()}


def query_make_dry_run_flags(
    target_rel_path: str, extra_vars: list[str] | None = None
) -> str:
    """Return full dry-run compile commands from native/Makefile via make -B -n."""
    args = ["-B", "-n", target_rel_path, *(extra_vars or [])]
    res = run_make_native(args)
    assert res.returncode == 0, f"make -B -n failed: {res.stderr}"
    return res.stdout


def parse_metal_compile_commands(dry_run_stdout: str) -> list[str]:
    """Extract xcrun metal -c commands for Metal 3.2 and 4.0."""
    return [
        line.strip()
        for line in dry_run_stdout.splitlines()
        if "xcrun metal " in line
        and " -c " in line
        and ("-std=metal3.2" in line or "-std=metal4.0" in line)
    ]


def assert_metal_fp_policy(dry_run_stdout: str) -> None:
    """Check the released arithmetic for each production Metal source."""
    cmds = parse_metal_compile_commands(dry_run_stdout)
    m32 = [c for c in cmds if "-std=metal3.2" in c]
    m40 = [c for c in cmds if "-std=metal4.0" in c]
    assert len(m32) >= 1 and len(m40) >= 1, (len(m32), len(m40))
    for cmd in cmds:
        assert "-fno-fast-math" in cmd, f"Missing -fno-fast-math: {cmd}"
        args = shlex.split(cmd)
        source = Path(args[args.index("-c") + 1]).name
        strict = "-std=metal4.0" in args or source in {
            "raw_decode.metal",
            "logit_topk.metal",
        }
        contract_flags = [arg for arg in args if arg.startswith("-ffp-contract=")]
        assert contract_flags == (["-ffp-contract=off"] if strict else []), cmd
