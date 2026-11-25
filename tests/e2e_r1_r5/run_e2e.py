#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unified E2E runner for tokspur R1-R5 (--mode offline|live, --tier 1|2|3|4|all)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_DIR = REPO_ROOT / "tests/e2e_r1_r5"
repo_python = REPO_ROOT / ".venv/bin/python"
TEST_PYTHON = str(repo_python) if repo_python.is_file() else sys.executable


def _run_step(
    name: str, argv: list[str], *, env: dict[str, str] | None = None
) -> dict[str, Any]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    proc = subprocess.run(
        argv,
        cwd=str(REPO_ROOT),
        env=merged_env,
        text=True,
        capture_output=True,
        check=False,
    )
    print(f"[{name}] exit={proc.returncode}")
    if proc.stdout.strip():
        print(proc.stdout.rstrip())
    if proc.stderr.strip():
        print(proc.stderr.rstrip(), file=sys.stderr)
    return {
        "step": name,
        "argv": argv,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _count_collected(collect_stdout: str) -> int:
    count = 0
    for line in collect_stdout.splitlines():
        s = line.strip()
        if s.startswith("tests/e2e_r1_r5/test_tier") and "::" in s:
            count += 1
    return count


def run_offline(json_report: Path | None) -> int:
    steps = [
        _run_step(
            "ruff-check", [TEST_PYTHON, "-m", "ruff", "check", "tests/e2e_r1_r5"]
        ),
        _run_step(
            "ruff-format",
            [TEST_PYTHON, "-m", "ruff", "format", "--check", "tests/e2e_r1_r5"],
        ),
        _run_step(
            "collect-tiers",
            [
                TEST_PYTHON,
                "-m",
                "pytest",
                "tests/e2e_r1_r5",
                "-m",
                "tier1 or tier2 or tier3 or tier4",
                "--collect-only",
                "-q",
            ],
            env={"TOKSPUR_E2E_MODE": "offline"},
        ),
        _run_step(
            "offline-tests",
            [
                TEST_PYTHON,
                "-m",
                "pytest",
                "tests/e2e_r1_r5",
                "-m",
                "e2e_offline",
                "--e2e-mode=offline",
                "-v",
            ],
            env={"TOKSPUR_E2E_MODE": "offline"},
        ),
    ]
    collected = _count_collected(steps[2]["stdout"])
    ok = all(s["returncode"] == 0 for s in steps) and collected == 230
    if collected != 230:
        print(
            f"[ERROR] Expected 230 Tier 1-4 cases collected, got {collected}",
            file=sys.stderr,
        )
    if json_report is not None:
        json_report.parent.mkdir(parents=True, exist_ok=True)
        json_report.write_text(
            json.dumps(
                {
                    "mode": "offline",
                    "collected_tier_cases": collected,
                    "passed": ok,
                    "steps": steps,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0 if ok else 1


def run_live(tier: str, json_report: Path | None) -> int:
    pytest_args = [
        TEST_PYTHON,
        "-m",
        "pytest",
        "tests/e2e_r1_r5",
        "--e2e-mode=live",
        "--strict-live",
        "-v",
    ]
    if tier == "all":
        pytest_args.extend(["-m", "tier1 or tier2 or tier3 or tier4 or e2e_offline"])
    else:
        pytest_args.extend(["-m", f"tier{tier}"])

    step = _run_step(
        f"live-tier-{tier}",
        pytest_args,
        env={"TOKSPUR_E2E_MODE": "live"},
    )
    ok = step["returncode"] == 0
    if json_report is not None:
        json_report.parent.mkdir(parents=True, exist_ok=True)
        json_report.write_text(
            json.dumps(
                {"mode": "live", "tier": tier, "passed": ok, "steps": [step]},
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run tokspur R1-R5 E2E suite in offline or live mode."
    )
    parser.add_argument(
        "--mode",
        choices=("offline", "live"),
        default="offline",
        help="Execution mode (default: offline).",
    )
    parser.add_argument(
        "--tier",
        choices=("1", "2", "3", "4", "all"),
        default="all",
        help="Tier selection for live mode (default: all).",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        default=None,
        help="Optional path to write structured JSON summary report.",
    )
    args = parser.parse_args(argv)
    if args.mode == "offline":
        return run_offline(args.json_report)
    return run_live(args.tier, args.json_report)


if __name__ == "__main__":
    raise SystemExit(main())
