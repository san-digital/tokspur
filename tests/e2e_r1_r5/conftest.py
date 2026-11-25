# SPDX-License-Identifier: Apache-2.0
"""Pytest configuration, live/offline gating, and E2E fixtures for R1-R5."""

from __future__ import annotations

import functools
import os
import subprocess
from pathlib import Path
from typing import Any, Callable

import pytest

from . import harness


def _default_e2e_mode() -> str:
    env_mode = os.environ.get("TOKSPUR_E2E_MODE", "offline").strip().lower()
    return env_mode if env_mode in ("offline", "live") else "offline"


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("tokspur-e2e")
    group.addoption(
        "--e2e-mode",
        action="store",
        choices=("offline", "live"),
        default=_default_e2e_mode(),
        help="E2E execution mode: offline (default, zero GPU) or live (M6 full run)",
    )
    group.addoption(
        "--strict-live",
        action="store_true",
        default=False,
        help="Fail if any @pytest.mark.e2e_live test skips (auto-enabled in live mode)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "tier1: Tier 1 feature happy-path E2E test")
    config.addinivalue_line(
        "markers", "tier2: Tier 2 boundary/negative rejection E2E test"
    )
    config.addinivalue_line("markers", "tier3: Tier 3 pairwise cross-feature E2E test")
    config.addinivalue_line(
        "markers", "tier4: Tier 4 realistic application scenario E2E test"
    )
    config.addinivalue_line(
        "markers", "e2e_live: Requires built binaries/GPU/live execution (M6)"
    )
    config.addinivalue_line(
        "markers", "e2e_offline: Safe offline structural and contract self-check"
    )


def _is_live_mode(config: pytest.Config) -> bool:
    opt = config.getoption("--e2e-mode")
    return str(opt).strip().lower() == "live"


def _is_strict_live(config: pytest.Config) -> bool:
    return bool(config.getoption("--strict-live")) or _is_live_mode(config)


def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("e2e_live") is None:
        if (
            "_f2_" in item.name
            and not _is_strict_live(item.config)
            and not _cargo_frozen_ready()
        ):
            pytest.skip(
                "frozen Cargo workspace cache not provisioned in this environment"
            )
        return
    if not _is_live_mode(item.config):
        pytest.skip("offline mode: live M6 execution deferred")
    harness.assert_no_competing_inference_processes()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[Any]):
    outcome = yield
    report = outcome.get_result()
    if (
        report.skipped
        and item.get_closest_marker("e2e_live") is not None
        and _is_strict_live(item.config)
    ):
        report.outcome = "failed"
        report.longrepr = (
            "Strict-live mode violation: @pytest.mark.e2e_live test "
            f"{item.nodeid} skipped instead of executing/failing"
        )


@pytest.fixture(scope="session")
def e2e_mode(request: pytest.FixtureRequest) -> str:
    return str(request.config.getoption("--e2e-mode"))


@pytest.fixture
def isolated_state_root(tmp_path: Path) -> Path:
    """Isolated clean absolute --state-root path inside tmp_path."""
    state_root = (tmp_path / "tokspur-state-root").resolve()
    state_root.mkdir(parents=True, mode=0o700, exist_ok=True)
    os.chmod(state_root, 0o700)
    return state_root


@pytest.fixture
def http_client_factory() -> Callable[[int, Path], harness.HttpClient]:
    """Create an HttpClient authenticated by <state_dir>/api-key."""

    def _make(port: int, state_root: Path) -> harness.HttpClient:
        state_dir = state_root / ".tokspur/state"
        key = harness.read_api_key(state_dir)
        return harness.HttpClient(port=port, api_key=key)

    return _make


@pytest.fixture
def synthetic_gguf_factory(tmp_path: Path) -> Callable[..., Path]:
    def _build(name: str = "synthetic.gguf", **kwargs: Any) -> Path:
        return harness.build_synthetic_gguf_v3(tmp_path / name, **kwargs)

    return _build


@pytest.fixture
def synthetic_trace_factory(tmp_path: Path) -> Callable[..., Path]:
    def _build(
        name: str = "trace.jsonl", builder: harness.DispatchTraceBuilder | None = None
    ) -> Path:
        tb = builder or harness.DispatchTraceBuilder()
        return tb.write(tmp_path / name)

    return _build


@functools.lru_cache(maxsize=1)
def _cargo_frozen_ready() -> bool:
    rust_dir = Path(__file__).resolve().parents[2] / "rust"
    env = {**os.environ, "RUSTUP_AUTO_INSTALL": "0", "CARGO_NET_OFFLINE": "true"}
    try:
        proc = subprocess.run(
            ["cargo", "metadata", "--frozen", "--format-version", "1"],
            cwd=rust_dir,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.fixture
def synthetic_r5_factory(tmp_path: Path) -> Callable[..., dict[str, Any]]:
    def _build(subdir: str = "holdout", **kwargs: Any) -> dict[str, Any]:
        return harness.build_holdout_manifest(tmp_path / subdir, **kwargs)

    return _build
