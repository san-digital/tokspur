# SPDX-License-Identifier: Apache-2.0
"""Offline structural, coverage, R6-exclusion, and contract self-checks."""

from __future__ import annotations

import ast
import copy
import json
import re
import struct
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from . import harness

SUITE_DIR = Path(__file__).resolve().parent
TIER_FILES = {
    1: SUITE_DIR / "test_tier1_features.py",
    2: SUITE_DIR / "test_tier2_boundaries.py",
    3: SUITE_DIR / "test_tier3_cross_feature.py",
    4: SUITE_DIR / "test_tier4_scenarios.py",
}


def _collect_ast_test_functions(path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    funcs: dict[str, ast.FunctionDef] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            funcs[node.name] = node
    return funcs


def _decorator_names(func: ast.FunctionDef) -> set[str]:
    names: set[str] = set()
    for dec in func.decorator_list:
        if isinstance(dec, ast.Attribute):
            names.add(dec.attr)
        elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
            names.add(dec.func.attr)
    return names


def _has_nontrivial_assertion(func: ast.FunctionDef) -> bool:
    for node in ast.walk(func):
        if isinstance(node, ast.Assert):
            if isinstance(node.test, ast.Constant) and node.test.value is True:
                continue
            return True
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr in (
                "raises",
                "fail",
            ):
                return True
            if isinstance(node.func, ast.Attribute) and node.func.attr.startswith(
                "assert_"
            ):
                return True
            if isinstance(node.func, ast.Name) and node.func.id.startswith("assert_"):
                return True
    return False


def _uses_mocking(func: ast.FunctionDef) -> bool:
    for node in ast.walk(func):
        if isinstance(node, ast.Name) and node.id in (
            "monkeypatch",
            "MagicMock",
            "patch",
        ):
            return True
        if isinstance(node, ast.Attribute) and node.attr in ("monkeypatch", "patch"):
            return True
    return False


@pytest.mark.e2e_offline
def test_registry_covers_all_f1_to_f20_thresholds() -> None:
    registry = harness.FEATURE_CASE_REGISTRY
    tier_counts = Counter(c.tier for c in registry)
    assert tier_counts[1] == 100
    assert tier_counts[2] == 100
    assert tier_counts[3] == 20
    assert tier_counts[4] == 10
    assert len(registry) == 230

    case_ids = [c.case_id for c in registry]
    assert len(case_ids) == len(set(case_ids)), "Duplicate case_id in registry"

    for idx in range(1, 21):
        fid = f"F{idx}"
        t1 = [c for c in registry if c.tier == 1 and fid in c.feature_ids]
        t2 = [c for c in registry if c.tier == 2 and fid in c.feature_ids]
        t3 = [c for c in registry if c.tier == 3 and fid in c.feature_ids]
        t4 = [c for c in registry if c.tier == 4 and fid in c.feature_ids]
        assert len(t1) == 5, f"{fid} Tier 1 count={len(t1)}, expected 5"
        assert len(t2) == 5, f"{fid} Tier 2 count={len(t2)}, expected 5"
        assert len(t3) >= 1, f"{fid} missing from Tier 3 pairwise coverage"
        assert len(t4) >= 1, f"{fid} missing from Tier 4 scenario coverage"


@pytest.mark.e2e_offline
def test_ast_matches_registry_and_forbids_facade_tests() -> None:
    registry_by_tier = {
        tier: {c.case_id: c for c in harness.FEATURE_CASE_REGISTRY if c.tier == tier}
        for tier in (1, 2, 3, 4)
    }
    for tier, path in TIER_FILES.items():
        assert path.exists(), f"Missing tier file {path}"
        funcs = _collect_ast_test_functions(path)
        expected_names = {f"test_{cid}" for cid in registry_by_tier[tier]}
        assert set(funcs.keys()) == expected_names, (
            f"Tier {tier} test functions do not match FEATURE_CASE_REGISTRY"
        )
        required_tier_marker = f"tier{tier}"
        for fname, fnode in funcs.items():
            decs = _decorator_names(fnode)
            assert required_tier_marker in decs, (
                f"{fname} missing @{required_tier_marker}"
            )
            case = registry_by_tier[tier][fname.removeprefix("test_")]
            marker = (
                "e2e_offline"
                if tier <= 2 and case.feature_ids == ("F2",)
                else "e2e_live"
            )
            assert decs & {"e2e_offline", "e2e_live"} == {marker}, (
                f"{fname} needs {marker}"
            )
            assert _has_nontrivial_assertion(fnode), (
                f"{fname} lacks a genuine non-trivial assertion"
            )
            assert not _uses_mocking(fnode), (
                f"{fname} uses forbidden mocking in opaque-box E2E test"
            )
            for stmt in fnode.body:
                assert not isinstance(stmt, ast.Pass), f"{fname} contains pass"


@pytest.mark.e2e_offline
def test_r6_exclusions_absent_from_repo_config_and_cli() -> None:
    cargo_toml = (harness.RUST_DIR / "Cargo.toml").read_text(encoding="utf-8")
    rust_mk = (harness.NATIVE_DIR / "rust.mk").read_text(encoding="utf-8")
    cli_py = (harness.REPO_ROOT / "src/tokspur/cli.py").read_text(encoding="utf-8")
    rust_cli_dir = harness.RUST_DIR / "crates/tokspur-cli/src"
    rust_cli_src = "\n".join(
        p.read_text(encoding="utf-8") for p in sorted(rust_cli_dir.glob("*.rs"))
    )

    lean_forbidden = re.compile(r"\b(nanoda_lib|lean4|lean-extract|olean_verify)\b")
    assert not lean_forbidden.search(cargo_toml), "R6 EX1 found in rust/Cargo.toml"
    assert not lean_forbidden.search(rust_mk), "R6 EX1 found in native/rust.mk"
    assert not lean_forbidden.search(rust_cli_src), "R6 EX1 found in tokspur-cli"

    early_cap_forbidden = re.compile(
        r"(dflash2[-_]early[-_]cap|history[-_]only[-_]head[-_]cap|"
        r"cap[-_]after[-_]two[-_]short[-_]rounds)"
    )
    assert not early_cap_forbidden.search(cli_py), "R6 EX2 found in cli.py"
    assert not early_cap_forbidden.search(rust_cli_src), "R6 EX2 found in tokspur-cli"


@pytest.mark.e2e_offline
def test_pyproject_ruff_and_check_sources_offline() -> None:
    pyproject = tomllib.loads(
        (harness.REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert pyproject["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]
    assert pyproject["tool"]["ruff"]["line-length"] == 88
    assert pyproject["tool"]["ruff"]["lint"]["select"] == ["E4", "E7", "E9", "F", "I"]

    res = harness.run_script("check_sources.py", [])
    assert res.returncode == 0, f"check_sources.py failed: {res.stderr}"


@pytest.mark.e2e_offline
def test_cli_offline_help_inspection_without_gpu() -> None:
    for help_args in (["--help"], ["serve", "--help"]):
        py_res = harness.run_tokspur(help_args, cli_impl="python")
        assert py_res.returncode == 0, py_res.stderr
        assert "usage:" in py_res.stdout.lower()

    cli_bin = harness.find_tokspur_cli()
    if cli_bin is not None:
        for help_args in (["--help"], ["pack", "--help"], ["serve", "--help"]):
            rs_res = harness.run_tokspur(help_args, cli_impl="rust")
            assert rs_res.returncode == 0, rs_res.stderr
            assert "usage:" in rs_res.stdout.lower()
    else:
        venv_tokspur = harness.REPO_ROOT / ".venv/bin/tokspur"
        if venv_tokspur.exists():
            miss_res = harness.run_cmd(
                [str(venv_tokspur), "--help"], env={"TOKSPUR_CLI_IMPL": "rust"}
            )
            assert miss_res.returncode == 1


@pytest.mark.e2e_offline
def test_synthetic_builders_and_reference_oracles(tmp_path: Path) -> None:
    gguf_path = harness.build_synthetic_gguf_v3(tmp_path / "synth.gguf")
    gguf_bytes = gguf_path.read_bytes()
    assert struct.unpack("<IIQ", gguf_bytes[:16]) == (0x46554747, 3, 1)

    ref_p, cand_p = harness.build_dag_pair_traces(tmp_path / "traces", depth=3)
    ref_lines = [json.loads(line) for line in ref_p.read_text().splitlines()]
    cand_lines = [json.loads(line) for line in cand_p.read_text().splitlines()]
    assert ref_lines[0]["event"] == "trace" and ref_lines[0]["schema"] == 1
    assert sum(1 for item in cand_lines if item["event"] == "barrier") == 2
    assert any(
        item["event"] == "dispatch" and len(item.get("constants") or []) >= 1
        for item in cand_lines
    )
    assert any(item["event"] == "state" for item in cand_lines)

    manifest_info = harness.build_holdout_manifest(tmp_path / "holdout")
    manifest = manifest_info["manifest"]
    assert manifest["schema_id"] == "track4.empirical_vv_manifest.v1"
    assert manifest["epistemic_header"] == harness.EPISTEMIC_VERIFICATION_HEADER
    assert len(manifest["holdout_splits"]) == 2
    assert "expected_sha256" in manifest["holdout_splits"][0]
    assert "sha256" not in manifest["holdout_splits"][0]

    w_pos, w_neg, w_min, p_val, n_eff = harness.reference_exact_wilcoxon_signed_rank(
        [-2.0, -2.5, -1.8, -2.1, -2.3, -1.9]
    )
    assert n_eff == 6 and w_pos == 0.0 and w_neg == 21.0 and w_min == 0.0
    assert p_val < 0.05
    q_vals = harness.reference_apply_bh_fdr([0.01, 0.04, 0.50], 0.05)
    assert q_vals[0][1] is True and q_vals[2][1] is False


class _AlphaRenamer(ast.NodeTransformer):
    """Alpha-rename locally bound identifiers while preserving globals and attrs."""

    def __init__(self, func: ast.FunctionDef) -> None:
        self.mapping: dict[str, str] = {}
        for arg in (
            *func.args.posonlyargs,
            *func.args.args,
            *func.args.kwonlyargs,
        ):
            self._bind(arg.arg)
        if func.args.vararg is not None:
            self._bind(func.args.vararg.arg)
        if func.args.kwarg is not None:
            self._bind(func.args.kwarg.arg)
        for node in ast.walk(func):
            if isinstance(node, ast.Name) and isinstance(
                node.ctx, (ast.Store, ast.Del)
            ):
                self._bind(node.id)
            elif isinstance(node, ast.ExceptHandler) and isinstance(node.name, str):
                self._bind(node.name)

    def _bind(self, ident: str) -> str:
        if ident not in self.mapping:
            self.mapping[ident] = f"_v{len(self.mapping)}"
        return self.mapping[ident]

    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.arg = self.mapping.get(node.arg, node.arg)
        node.annotation = None
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id in self.mapping:
            node.id = self.mapping[node.id]
        return node

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.ExceptHandler:
        if isinstance(node.name, str) and node.name in self.mapping:
            node.name = self.mapping[node.name]
        self.generic_visit(node)
        return node


def _normalize_test_ast(func: ast.FunctionDef) -> str:
    cloned = copy.deepcopy(func)
    cloned.name = "__test__"
    cloned.decorator_list = []
    cloned.returns = None
    if (
        cloned.body
        and isinstance(cloned.body[0], ast.Expr)
        and isinstance(cloned.body[0].value, ast.Constant)
        and isinstance(cloned.body[0].value.value, str)
    ):
        cloned.body = cloned.body[1:]
    cloned = _AlphaRenamer(cloned).visit(cloned)
    return ast.dump(cloned, annotate_fields=True, include_attributes=False)


def _is_returncode_attr(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "returncode"


def _is_status_or_returncode_expr(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute) and node.attr in ("status", "returncode"):
        return node.attr
    if isinstance(node, ast.Name) and node.id in ("status", "returncode"):
        return node.id
    return None


def _is_zero_literal(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and node.value == 0
    )


def _is_non_none_get_call(node: ast.AST) -> bool:
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
    ):
        return False
    if len(node.args) >= 2:
        arg1 = node.args[1]
        if not (isinstance(arg1, ast.Constant) and arg1.value is None):
            return True
    for kw in node.keywords:
        if kw.arg == "default":
            if not (isinstance(kw.value, ast.Constant) and kw.value.value is None):
                return True
    return False


def _is_zero_get_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and len(node.args) >= 2
        and _is_zero_literal(node.args[1])
    )


def _function_has_existence_or_upper_bound(
    func: ast.FunctionDef, target: ast.AST
) -> bool:
    target_dump = ast.dump(target, include_attributes=False)
    for node in ast.walk(func):
        if isinstance(node, ast.Compare):
            for op in node.ops:
                if isinstance(op, (ast.In, ast.Lt, ast.LtE, ast.Eq)):
                    if any(
                        ast.dump(sub, include_attributes=False) == target_dump
                        for sub in ast.walk(node)
                    ):
                        return True
        if isinstance(node, ast.Subscript):
            if ast.dump(node.value, include_attributes=False) == target_dump:
                return True
    return False


@pytest.mark.e2e_offline
def test_ast_guardrail_a_no_cross_tier_duplicates() -> None:
    seen: dict[str, list[str]] = defaultdict(list)
    for tier, path in TIER_FILES.items():
        for fname, fnode in _collect_ast_test_functions(path).items():
            seen[_normalize_test_ast(fnode)].append(f"T{tier}:{fname}")
    dups = [group for group in seen.values() if len(group) > 1]
    assert dups == [], f"Cross-tier normalized AST duplicates detected: {dups}"


@pytest.mark.e2e_offline
def test_ast_guardrail_b_no_tautological_assertions() -> None:
    violations: list[str] = []
    for tier, path in TIER_FILES.items():
        for fname, fnode in _collect_ast_test_functions(path).items():
            for node in ast.walk(fnode):
                if isinstance(node, ast.Assert):
                    test_expr = node.test
                    if isinstance(test_expr, ast.BoolOp) and isinstance(
                        test_expr.op, ast.Or
                    ):
                        for val in test_expr.values:
                            if (
                                isinstance(val, ast.Compare)
                                and len(val.ops) == 1
                                and isinstance(val.ops[0], ast.Eq)
                                and _is_returncode_attr(val.left)
                                and _is_zero_literal(val.comparators[0])
                            ):
                                violations.append(
                                    f"T{tier}:{fname}: tautological `or returncode == 0`"
                                )
                    for sub in ast.walk(test_expr):
                        if (
                            isinstance(sub, ast.Compare)
                            and len(sub.ops) == 1
                            and isinstance(sub.ops[0], ast.In)
                            and isinstance(
                                sub.comparators[0], (ast.Tuple, ast.List, ast.Set)
                            )
                        ):
                            kind = _is_status_or_returncode_expr(sub.left)
                            elts = {
                                e.value
                                for e in sub.comparators[0].elts
                                if isinstance(e, ast.Constant)
                                and isinstance(e.value, int)
                            }
                            if kind == "returncode" and (
                                0 in elts and any(v != 0 for v in elts)
                            ):
                                violations.append(
                                    f"T{tier}:{fname}: mixed pass/fail `returncode in {sorted(elts)}`"
                                )
                            if kind == "status" and (
                                any(200 <= v < 300 for v in elts)
                                and any(v >= 400 for v in elts)
                            ):
                                violations.append(
                                    f"T{tier}:{fname}: mixed pass/fail `status in {sorted(elts)}`"
                                )
                        if _is_non_none_get_call(sub):
                            violations.append(
                                f"T{tier}:{fname}: `.get(key, <non-None>)` default masking in assert"
                            )
                        if (
                            isinstance(sub, ast.Compare)
                            and len(sub.ops) == 1
                            and isinstance(sub.ops[0], ast.GtE)
                            and _is_zero_literal(sub.comparators[0])
                        ):
                            if _is_zero_get_call(sub.left):
                                violations.append(
                                    f"T{tier}:{fname}: tautological `.get(..., 0) >= 0`"
                                )
                            elif not _function_has_existence_or_upper_bound(
                                fnode, sub.left
                            ):
                                violations.append(
                                    f"T{tier}:{fname}: unbounded `>= 0` without key check"
                                )
    assert violations == [], f"Tautological assertion(s): {violations}"


_SUT_CALL_NAMES = frozenset(
    {
        "ServerProcess",
        "run_tokspur",
        "run_cmd",
        "run_dispatch_graph",
        "run_make_native",
        "run_script",
        "run_cargo_test_checked",
        "run_cargo_metadata",
        "run_r5_holdout_eval",
        "run_r5_futility_eval",
        "run_r5_holdout_suite",
        "run_r5_futility_suite",
        "list_metallib_symbols",
        "query_make_dry_run_flags",
        "build_metal_graph_probe",
        "build_f11_metal_graph_probe",
        "build_planar_matrix_probe",
        "build_planar_dm_cpu_probe",
        "build_dag_pair_traces",
        "build_holdout_manifest",
        "build_synthetic_gguf_v3",
    }
)


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


def _propagate_derived_vars(func: ast.FunctionDef, seed_vars: set[str]) -> set[str]:
    tracked = set(seed_vars)
    changed = True
    while changed:
        changed = False
        for node in ast.walk(func):
            targets: list[ast.AST] = []
            value: ast.AST | None = None
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
                value = node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets = [node.target]
                value = node.value
            elif isinstance(node, ast.For):
                targets = [node.target]
                value = node.iter
            if value is None:
                continue
            val_names = {sub.id for sub in ast.walk(value) if isinstance(sub, ast.Name)}
            if val_names & tracked:
                for tgt in targets:
                    for sub in ast.walk(tgt):
                        if isinstance(sub, ast.Name) and sub.id not in tracked:
                            tracked.add(sub.id)
                            changed = True
    return tracked


def _var_flows_into_sut_call(func: ast.FunctionDef, tracked_vars: set[str]) -> bool:
    for node in ast.walk(func):
        if isinstance(node, ast.Call) and _call_name(node) in _SUT_CALL_NAMES:
            arg_nodes = list(node.args) + [kw.value for kw in node.keywords]
            for arg_node in arg_nodes:
                for sub in ast.walk(arg_node):
                    if isinstance(sub, ast.Name) and sub.id in tracked_vars:
                        return True
    return False


@pytest.mark.e2e_offline
def test_ast_guardrail_c_no_unused_fixtures_or_tmp_path() -> None:
    unused: list[str] = []
    for tier, path in TIER_FILES.items():
        for fname, fnode in _collect_ast_test_functions(path).items():
            param_names = [arg.arg for arg in fnode.args.args]
            loaded_names = {
                node.id
                for stmt in fnode.body
                for node in ast.walk(stmt)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            }
            for param in param_names:
                if param not in loaded_names:
                    unused.append(f"T{tier}:{fname}: unused parameter `{param}`")
                if param == "synthetic_pack_v3_factory":
                    unused.append(f"T{tier}:{fname}: forbidden dead fixture `{param}`")

            if "tmp_path" in param_names:
                tmp_vars = _propagate_derived_vars(fnode, {"tmp_path"})
                if not _var_flows_into_sut_call(fnode, tmp_vars):
                    unused.append(
                        f"T{tier}:{fname}: `tmp_path` never flows into a SUT call"
                    )

            for idx, stmt in enumerate(fnode.body):
                for node in ast.walk(stmt):
                    if (
                        isinstance(node, ast.Call)
                        and _call_name(node) == "tamper_file_bit"
                        and node.args
                    ):
                        tgt_dump = ast.dump(node.args[0], include_attributes=False)
                        tgt_names = {
                            sub.id
                            for sub in ast.walk(node.args[0])
                            if isinstance(sub, ast.Name)
                            and isinstance(sub.ctx, ast.Load)
                        }
                        used_after = False
                        for later in fnode.body[idx + 1 :]:
                            for lnode in ast.walk(later):
                                if (
                                    isinstance(lnode, ast.Call)
                                    and _call_name(lnode) in _SUT_CALL_NAMES
                                ):
                                    if any(
                                        ast.dump(sub, include_attributes=False)
                                        == tgt_dump
                                        or (
                                            isinstance(sub, ast.Name)
                                            and sub.id in tgt_names
                                        )
                                        for arg in (
                                            list(lnode.args)
                                            + [kw.value for kw in lnode.keywords]
                                        )
                                        for sub in ast.walk(arg)
                                    ):
                                        used_after = True
                        if not used_after:
                            unused.append(
                                f"T{tier}:{fname}: `tamper_file_bit` target never passed to subsequent SUT call"
                            )
    assert unused == [], f"Unused or dead test fixture violations: {unused}"


@pytest.mark.e2e_offline
def test_ast_guardrail_d_no_whitebox_source_reads_in_tier2() -> None:
    tier2_path = TIER_FILES[2]
    tree = ast.parse(tier2_path.read_text(encoding="utf-8"), filename=str(tier2_path))
    forbidden_attrs = {"REPO_ROOT", "NATIVE_DIR", "RUST_DIR"}
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in forbidden_attrs:
            violations.append(f"white-box path reference: {node.attr}")
        if isinstance(node, ast.Name) and node.id in forbidden_attrs:
            violations.append(f"white-box symbol reference: {node.id}")
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and any(
                frag in node.value
                for frag in (
                    "src/tokspur/",
                    "native/src/",
                    "native/rust.mk",
                    "rust/Cargo.toml",
                )
            )
        ):
            violations.append(f"white-box repo source literal: {node.value!r}")
    assert violations == [], f"Tier 2 white-box violations: {violations}"


@pytest.mark.e2e_offline
def test_ast_guardrail_e_no_zero_sut_or_unchecked_cargo_test() -> None:
    sut_primitives = {
        "ServerProcess",
        "run_tokspur",
        "run_cmd",
        "run_dispatch_graph",
        "run_make_native",
        "run_script",
        "run_cargo_test_checked",
        "run_cargo_metadata",
        "run_r5_holdout_eval",
        "run_r5_futility_eval",
        "run_r5_holdout_suite",
        "run_r5_futility_suite",
        "list_metallib_symbols",
        "query_make_dry_run_flags",
    }
    case_features = {
        c.case_id: set(c.feature_ids) for c in harness.FEATURE_CASE_REGISTRY
    }
    zero_sut: list[str] = []
    unchecked_cargo: list[str] = []
    misattributed: list[str] = []
    for tier, path in TIER_FILES.items():
        for fname, fnode in _collect_ast_test_functions(path).items():
            called: set[str] = set()
            str_literals: set[str] = set()
            for node in ast.walk(fnode):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        called.add(node.func.attr)
                    elif isinstance(node.func, ast.Name):
                        called.add(node.func.id)
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    str_literals.add(node.value)
            if not (called & sut_primitives):
                zero_sut.append(f"T{tier}:{fname}")
            if "run_cargo_test" in called:
                unchecked_cargo.append(f"T{tier}:{fname}")

            cid = fname[len("test_") :] if fname.startswith("test_") else fname
            fids = case_features.get(cid, set())
            if "F11" in fids:
                bad_f11 = {
                    "test-dflash2_context_stage",
                    "test-challenger_m4_2",
                    "test-numeric",
                    "ci-components",
                } & str_literals
                if bad_f11:
                    misattributed.append(
                        f"T{tier}:{fname} (F11) uses unrelated target(s): {sorted(bad_f11)}"
                    )
            if "F14" in fids:
                bad_f14 = {
                    "test-challenger_m4_2",
                    "test-numeric",
                    "ci-components",
                } & str_literals
                if bad_f14:
                    misattributed.append(
                        f"T{tier}:{fname} (F14) uses unrelated target(s): {sorted(bad_f14)}"
                    )
            if "F10" in fids and "test-graph" in str_literals:
                if "run_dispatch_graph" not in called:
                    misattributed.append(
                        f"T{tier}:{fname} (F10) invokes test-graph without run_dispatch_graph"
                    )
            if "F13" in fids and (
                "build_synthetic_pack_v3" in called
                or "synthetic_pack_v3_factory" in called
            ):
                misattributed.append(
                    f"T{tier}:{fname} (F13) uses dead synthetic Pack v3 builder"
                )

    assert zero_sut == [], f"Zero-SUT facade tests: {zero_sut}"
    assert unchecked_cargo == [], f"Unchecked run_cargo_test calls: {unchecked_cargo}"
    assert misattributed == [], f"Feature-to-binary misattributions: {misattributed}"
