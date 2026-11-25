import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rust_receipt_cache", ROOT / "scripts" / "rust_receipt_cache.py"
)
rc_mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rc_mod)


def run_cache(cache, action, command, extra_env=None):
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [
            sys.executable,
            "-I",
            str(ROOT / "scripts" / "rust_receipt_cache.py"),
            "--repo-root",
            str(ROOT),
            "--cache-dir",
            str(cache),
            *action,
            "--",
            sys.executable,
            "-c",
            command,
        ],
        env=env,
        capture_output=True,
        text=True,
    )


def test_cached_command_includes_its_arguments_and_preserves_siblings(tmp_path):
    action = ["--action", "test", "--arm", "ordinary", "--cpp-oracle", "0"]
    assert "STORE" in run_cache(tmp_path, action, "print('first')").stdout
    assert "HIT" in run_cache(tmp_path, action, "print('first')").stdout
    assert "STORE" in run_cache(tmp_path, action, "print('second')").stdout
    # Targeted invocation must not overwrite the earlier command's receipt.
    assert "HIT" in run_cache(tmp_path, action, "print('first')").stdout
    assert "HIT" in run_cache(tmp_path, action, "print('second')").stdout
    assert len(list((tmp_path / "receipts").glob("*.json"))) == 2
    assert run_cache(tmp_path, action, "raise SystemExit(3)").returncode == 3


def test_fuzz_receipt_requires_every_requested_output(tmp_path):
    cache = tmp_path / "cache"
    built = tmp_path / "built"
    restored = tmp_path / "restored"
    built.mkdir()
    (built / "json").write_bytes(b"json")
    (built / "wire").write_bytes(b"wire")
    action = [
        "--action",
        "fuzz-build",
        "--arm",
        "cpp-oracle",
        "--fuzz-dir",
        str(restored),
        "--fuzz-out",
        str(built),
        "--fuzz-bins",
        "json wire",
    ]
    assert "STORE" in run_cache(cache, action, "pass").stdout
    assert "HIT" in run_cache(cache, action, "pass").stdout
    receipt = next((cache / "receipts").glob("*.json"))
    original = json.loads(receipt.read_bytes())
    for names in [["bin/json"], ["bin/wire", "bin/wire"], ["bin/unused", "bin/wire"]]:
        changed = dict(original)
        changed["artifacts"] = [
            dict(item, rel_path=name)
            for item, name in zip(original["artifacts"], names, strict=False)
        ]
        receipt.write_text(json.dumps(changed), encoding="utf-8")
        assert "STORE" in run_cache(cache, action, "pass").stdout
        assert not (restored / "bin" / "unused").exists()


def test_workspace_crate_deps_parses_table_and_target_cfg_sections(tmp_path):
    rust_dir = tmp_path / "rust"
    a_dir = rust_dir / "crates" / "tokspur-alpha"
    b_dir = rust_dir / "crates" / "tokspur-beta"
    c_dir = rust_dir / "crates" / "tokspur-gamma"
    for d in (a_dir, b_dir, c_dir):
        d.mkdir(parents=True)
    (b_dir / "Cargo.toml").write_text(
        '[package]\nname = "tokspur-beta"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    (c_dir / "Cargo.toml").write_text(
        '[package]\nname = "tokspur-gamma"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    (a_dir / "Cargo.toml").write_text(
        '[package]\nname = "tokspur-alpha"\nversion = "0.1.0"\n\n'
        "[dependencies.tokspur-beta]\nworkspace = true\n\n"
        "[target.'cfg(target_os = \"macos\")'.dependencies]\n"
        'gamma_alias = { package = "tokspur-gamma", path = "../tokspur-gamma" }\n',
        encoding="utf-8",
    )
    deps = rc_mod._workspace_crate_deps(rust_dir)
    assert deps["tokspur-alpha"] == {"tokspur-beta", "tokspur-gamma"}


def test_fuzz_build_unsets_shadowing_rustflags_and_env_paths_hash_cleanly(tmp_path):
    cache = tmp_path / "cache"
    built = tmp_path / "built"
    restored = tmp_path / "restored"
    env_dump = tmp_path / "env_dump.json"
    built.mkdir()
    (built / "wire").write_bytes(b"wire-bin")
    action = [
        "--action",
        "fuzz-build",
        "--arm",
        "fast-math",
        "--fuzz-dir",
        str(restored),
        "--fuzz-out",
        str(built),
        "--fuzz-rustflags",
        "--cfg fuzzing -Coverflow-checks",
    ]
    cmd = (
        f"import json, os, pathlib; "
        f"pathlib.Path({str(env_dump)!r}).write_text(json.dumps({{"
        f"'RUSTFLAGS': os.environ.get('RUSTFLAGS'), "
        f"'ENCODED': os.environ.get('CARGO_ENCODED_RUSTFLAGS'), "
        f"'TARGET': os.environ.get('CARGO_TARGET_AARCH64_APPLE_DARWIN_RUSTFLAGS')"
        f"}}))"
    )
    res = run_cache(
        cache,
        action,
        cmd,
        extra_env={
            "CARGO_ENCODED_RUSTFLAGS": "-Copt-level=0",
            "CARGO_TARGET_AARCH64_APPLE_DARWIN_RUSTFLAGS": "-Copt-level=1",
            "TOKSPUR_FAKE_TARGET": "nonexistent/relative/target-bin",
        },
    )
    assert res.returncode == 0, res.stderr
    recorded = json.loads(env_dump.read_text(encoding="utf-8"))
    assert recorded["RUSTFLAGS"] == "--cfg fuzzing -Coverflow-checks"
    assert recorded["ENCODED"] is None
    assert recorded["TARGET"] is None

    fake_file = tmp_path / "fake_target_bin"
    fake_file.write_bytes(b"v1")
    test_action = ["--action", "test", "--arm", "fast-math", "--cpp-oracle", "0"]
    r1 = run_cache(
        cache,
        test_action,
        "pass",
        extra_env={"TOKSPUR_FAKE_TARGET": str(fake_file)},
    )
    assert "STORE" in r1.stdout
    fake_file.write_bytes(b"v2")
    r2 = run_cache(
        cache,
        test_action,
        "pass",
        extra_env={"TOKSPUR_FAKE_TARGET": str(fake_file)},
    )
    assert "STORE" in r2.stdout

    inputs = rc_mod._collect_inputs(ROOT, "clippy", "fast-math", "0")
    for profile_rel in (
        "profiles/flash-next-assets.json",
        "profiles/flash-next-text-assets.json",
        "profiles/m5-max-128.json",
        "profiles/owned-target-experimental.json",
    ):
        assert profile_rel in inputs
