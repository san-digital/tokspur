#!/usr/bin/env python3
"""Content-addressed SHA-256 receipt cache for native/rust.mk."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Dict, Set

DOMAIN_TAG = b"tokspur-rust-receipt-v2\0"
TOOLCHAIN_TAG = b"tokspur-toolchain-v1\0"
SOURCES_TAG = b"tokspur-sources-v1\0"

ENV_FLAG_KEYS = (
    "RUSTFLAGS",
    "CARGO_ENCODED_RUSTFLAGS",
    "CARGO_TARGET_AARCH64_APPLE_DARWIN_RUSTFLAGS",
    "RUSTC",
    "RUSTC_WRAPPER",
    "RUSTC_WORKSPACE_WRAPPER",
    "CARGO_BUILD_TARGET",
    "CARGO_INCREMENTAL",
    "CARGO_TARGET_DIR",
    "CC",
    "CXX",
    "CFLAGS",
    "CXXFLAGS",
    "LDFLAGS",
    "SDKROOT",
    "MACOSX_DEPLOYMENT_TARGET",
    "TOKSPUR_TEST_DERIVED_PACK_BIN",
    "TOKSPUR_FAKE_TARGET",
    "TOKSPUR_PINNED_TEMPLATE",
    "TOKSPUR_TEST_TEXT_METADATA",
    "TOKSPUR_ALLOW_KNOWN_PROFILE",
    "TOKSPUR_DUMP_VERIFY_PROFILE",
    "TOKSPUR_DUMP_PLANAR_UNION_PROFILE",
)

ENV_PATH_KEYS = (
    "TOKSPUR_FAKE_TARGET",
    "TOKSPUR_PINNED_TEMPLATE",
    "TOKSPUR_TEST_TEXT_METADATA",
    "TOKSPUR_TEST_DERIVED_PACK_BIN",
)

IGNORED_DIRS = {"target", ".git", "__pycache__"}
IGNORED_NAMES = {".DS_Store"}


def _push_len_prefixed(h: "hashlib._Hash", data: bytes) -> None:
    h.update(len(data).to_bytes(8, "little"))
    h.update(data)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    total = 0
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
            total += len(chunk)
    return h.hexdigest(), total


def _run_tool_stdout(cmd: list[str], cwd: Path) -> bytes:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        return b""
    return proc.stdout


def _workspace_crate_deps(rust_dir: Path) -> Dict[str, Set[str]]:
    crates_dir = rust_dir / "crates"
    parsed: list[tuple[str, dict[str, object]]] = []
    workspace_names: Set[str] = set()
    for cargo_toml in sorted(crates_dir.glob("*/Cargo.toml")):
        data = tomllib.loads(cargo_toml.read_text(encoding="utf-8"))
        pkg = data.get("package")
        pkg_name = pkg.get("name") if isinstance(pkg, dict) else None
        if isinstance(pkg_name, str):
            workspace_names.add(pkg_name)
            parsed.append((pkg_name, data))

    def _extract_from_dep_table(tbl: object, out: Set[str]) -> None:
        if not isinstance(tbl, dict):
            return
        for dep_key, spec in tbl.items():
            actual = (
                spec.get("package")
                if isinstance(spec, dict) and isinstance(spec.get("package"), str)
                else dep_key
            )
            if actual in workspace_names or actual.startswith("tokspur-"):
                out.add(actual)

    deps_map: Dict[str, Set[str]] = {}
    for pkg_name, data in parsed:
        deps: Set[str] = set()
        for sec in ("dependencies", "dev-dependencies", "build-dependencies"):
            _extract_from_dep_table(data.get(sec), deps)
        targets = data.get("target")
        if isinstance(targets, dict):
            for target_cfg in targets.values():
                if isinstance(target_cfg, dict):
                    for sec in (
                        "dependencies",
                        "dev-dependencies",
                        "build-dependencies",
                    ):
                        _extract_from_dep_table(target_cfg.get(sec), deps)
        deps_map[pkg_name] = deps
    return deps_map


def _transitive_crates(rust_dir: Path, seeds: list[str]) -> list[str]:
    graph = _workspace_crate_deps(rust_dir)
    seen: set[str] = set()
    stack = list(seeds)
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for nxt in graph.get(cur, ()):
            if nxt not in seen:
                stack.append(nxt)
    return sorted(seen)


def _collect_tree_files(repo_root: Path, base: Path, out: dict[str, str]) -> None:
    if not base.exists():
        return
    if base.is_symlink():
        raise RuntimeError(f"Refusing symlink in cache input tree: {base}")
    if base.is_file():
        rel = base.relative_to(repo_root).as_posix()
        digest, _ = _sha256_file(base)
        out[rel] = digest
        return
    for root, dirs, files in os.walk(base, followlinks=False):
        root_path = Path(root)
        dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIRS)
        for d in dirs:
            dp = root_path / d
            if dp.is_symlink():
                raise RuntimeError(f"Refusing symlink directory in cache input: {dp}")
        for fname in sorted(files):
            if fname in IGNORED_NAMES:
                continue
            fp = root_path / fname
            if fp.is_symlink():
                raise RuntimeError(f"Refusing symlink file in cache input: {fp}")
            if not fp.is_file():
                continue
            rel = fp.relative_to(repo_root).as_posix()
            digest, _ = _sha256_file(fp)
            out[rel] = digest


def _collect_inputs(
    repo_root: Path, action: str, arm: str, cpp_oracle: str
) -> dict[str, str]:
    rust_dir = repo_root / "rust"
    files: dict[str, str] = {}

    common_native = (
        "native/data/unicode9.inc",
        "profiles/flash-next-assets.json",
        "profiles/flash-next-text-assets.json",
        "profiles/m5-max-128.json",
        "profiles/owned-target-experimental.json",
        "tests/fixtures/native-build-inputs.txt",
    )
    for rel in common_native:
        p = repo_root / rel
        if p.exists():
            _collect_tree_files(repo_root, p, files)

    if arm == "ordinary":
        _collect_tree_files(repo_root, rust_dir / "crates", files)
        if action == "test":
            for sub in (
                "native/Makefile",
                "native/src",
                "native/include",
                "native/kernels",
                "native/diagnostics",
                "native/tests/test_derived_pack.cpp",
                "native/tests/grouped_mm_fixture.hpp",
            ):
                _collect_tree_files(repo_root, repo_root / sub, files)
        return files

    if arm == "cpp-oracle":
        crates = _transitive_crates(rust_dir, ["tokspur-fuzz"])
        for crate in crates:
            _collect_tree_files(repo_root, rust_dir / "crates" / crate, files)
        for name in ("json", "unicode", "gguf", "response"):
            _collect_tree_files(
                repo_root, repo_root / f"native/include/tokspur/{name}.hpp", files
            )
            _collect_tree_files(repo_root, repo_root / f"native/src/{name}.cpp", files)
        return files

    if action == "fuzz-build":
        crates = _transitive_crates(rust_dir, ["tokspur-fuzz"])
        for crate in crates:
            _collect_tree_files(repo_root, rust_dir / "crates" / crate, files)
        return files

    # Feature-level clippy / test arms: -p tokspur-wire -p tokspur-server -p tokspur-fuzz
    crates = _transitive_crates(
        rust_dir, ["tokspur-wire", "tokspur-server", "tokspur-fuzz"]
    )
    for crate in crates:
        _collect_tree_files(repo_root, rust_dir / "crates" / crate, files)
    if cpp_oracle == "1":
        # Keep distinct dependency sensitivity clean; cpp-oracle itself is hashed in key.
        pass
    return files


def _resolve_env_path_digest(val: str, repo_root: Path, rust_dir: Path) -> bytes:
    raw_p = Path(val)
    if raw_p.is_absolute():
        candidates = [raw_p]
    else:
        candidates = [
            Path.cwd() / raw_p,
            rust_dir / raw_p,
            repo_root / raw_p,
            *(
                crate_dir / raw_p
                for crate_dir in sorted((rust_dir / "crates").glob("*"))
                if crate_dir.is_dir()
            ),
        ]
    for cand in candidates:
        try:
            if cand.is_file():
                return _sha256_file(cand)[0].encode("ascii")
        except OSError:
            continue
    return b"missing"


def _compute_key(
    repo_root: Path,
    action: str,
    arm: str,
    cpp_oracle: str,
    features: str,
    fuzz_triple: str,
    fuzz_rustflags: str,
    command: list[str],
    fuzz_bins: str = "",
) -> dict[str, str]:
    rust_dir = repo_root / "rust"

    tc_hash = hashlib.sha256(TOOLCHAIN_TAG)
    tc_entries: list[tuple[str, bytes]] = [
        ("rust-toolchain.toml", (rust_dir / "rust-toolchain.toml").read_bytes()),
        ("rustc_vV", _run_tool_stdout(["rustc", "-vV"], rust_dir)),
        ("cargo_V", _run_tool_stdout(["cargo", "--version"], rust_dir)),
    ]
    needs_cxx = (
        cpp_oracle == "1"
        or arm == "cpp-oracle"
        or action == "fuzz-build"
        or (action == "test" and arm == "ordinary")
    )
    if needs_cxx:
        tc_entries.append(
            ("cxx_version", _run_tool_stdout(["c++", "--version"], rust_dir))
        )
        if sys.platform == "darwin":
            tc_entries.append(
                (
                    "sdk_version",
                    _run_tool_stdout(["xcrun", "--show-sdk-version"], rust_dir),
                )
            )
    search_path = os.pathsep.join(str(rust_dir / part) for part in os.get_exec_path())
    program = str(rust_dir / command[0]) if os.path.dirname(command[0]) else command[0]
    executable = shutil.which(program, path=search_path)
    if executable is None:
        raise FileNotFoundError(command[0])
    tc_entries.append(
        ("command_binary", _sha256_file(Path(executable))[0].encode("ascii"))
    )
    for k, v in sorted(tc_entries):
        _push_len_prefixed(tc_hash, k.encode("utf-8"))
        _push_len_prefixed(tc_hash, v)
    toolchain_sha256 = tc_hash.hexdigest()

    cargo_lock_sha256 = _sha256_bytes((rust_dir / "Cargo.lock").read_bytes())
    rust_mk_sha256 = _sha256_bytes((repo_root / "native/rust.mk").read_bytes())

    config_files: list[tuple[str, bytes]] = [
        ("rust/Cargo.toml", (rust_dir / "Cargo.toml").read_bytes()),
        ("rust/.cargo/config.toml", (rust_dir / ".cargo/config.toml").read_bytes()),
        ("command", json.dumps(command, ensure_ascii=True).encode("ascii")),
        ("runner", Path(__file__).read_bytes()),
    ]
    clippy_toml = rust_dir / "clippy.toml"
    if clippy_toml.exists():
        config_files.append(("rust/clippy.toml", clippy_toml.read_bytes()))
    for env_k in ENV_FLAG_KEYS:
        config_files.append((f"env:{env_k}", os.environ.get(env_k, "").encode("utf-8")))
    for key, value in os.environ.items():
        if (
            key.startswith(("CARGO_PROFILE_", "CARGO_TARGET_"))
            and key not in ENV_FLAG_KEYS
        ):
            config_files.append((f"env:{key}", value.encode("utf-8")))
    for path_key in ENV_PATH_KEYS:
        val = os.environ.get(path_key, "")
        if val:
            config_files.append(
                (
                    f"env_file:{path_key}",
                    _resolve_env_path_digest(val, repo_root, rust_dir),
                )
            )
    if action == "fuzz-build":
        config_files.append(("fuzz_triple", fuzz_triple.encode("utf-8")))
        config_files.append(("fuzz_rustflags", fuzz_rustflags.encode("utf-8")))
        config_files.append(("fuzz_bins", fuzz_bins.encode("utf-8")))

    source_digests = _collect_inputs(repo_root, action, arm, cpp_oracle)
    src_hash = hashlib.sha256(SOURCES_TAG)
    for rel_path in sorted(source_digests):
        _push_len_prefixed(src_hash, rel_path.encode("utf-8"))
        _push_len_prefixed(src_hash, source_digests[rel_path].encode("ascii"))
    sources_sha256 = src_hash.hexdigest()

    key_hash = hashlib.sha256(DOMAIN_TAG)
    for item in (
        action,
        arm,
        cpp_oracle,
        features,
        toolchain_sha256,
        cargo_lock_sha256,
        rust_mk_sha256,
    ):
        _push_len_prefixed(key_hash, item.encode("utf-8"))
    _push_len_prefixed(key_hash, len(config_files).to_bytes(8, "little"))
    for k, v in sorted(config_files):
        _push_len_prefixed(key_hash, k.encode("utf-8"))
        _push_len_prefixed(key_hash, v)
    _push_len_prefixed(key_hash, sources_sha256.encode("ascii"))

    return {
        "content_key": key_hash.hexdigest(),
        "toolchain_sha256": toolchain_sha256,
        "cargo_lock_sha256": cargo_lock_sha256,
        "rust_mk_sha256": rust_mk_sha256,
        "sources_sha256": sources_sha256,
    }


def _atomic_write_bytes(path: Path, data: bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".tmp-{os.getpid()}-{secrets.token_hex(6)}"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fchmod(fd, mode)
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    os.close(fd)
    os.replace(tmp, path)


def _atomic_copy_file(src: Path, dst: Path, mode: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".tmp-{os.getpid()}-{secrets.token_hex(6)}"
    try:
        shutil.copyfile(src, tmp)
        os.chmod(tmp, mode)
        os.replace(tmp, dst)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _verify_hit(
    receipt_path: Path,
    cache_dir: Path,
    action: str,
    arm: str,
    content_key: str,
    fuzz_dir: Path | None,
    expected_outputs: list[str],
) -> bool:
    if not receipt_path.is_file():
        return False
    try:
        meta = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    if not isinstance(meta, dict):
        return False
    if (
        meta.get("schema_version") != "1.0.0"
        or meta.get("status") != "passed"
        or meta.get("action") != action
        or meta.get("arm") != arm
        or meta.get("content_key") != content_key
    ):
        return False

    artifacts = meta.get("artifacts", [])
    if not isinstance(artifacts, list) or len(artifacts) != len(expected_outputs):
        return False
    if action == "fuzz-build":
        if not artifacts or fuzz_dir is None:
            return False
        blob_dir = cache_dir / "blobs" / f"{action}--{arm}" / content_key
        to_restore: list[tuple[Path, Path]] = []
        for item, filename in zip(artifacts, expected_outputs, strict=True):
            if not isinstance(item, dict) or item.get("rel_path") != f"bin/{filename}":
                return False
            exp_sha = item.get("sha256")
            exp_bytes = item.get("bytes")
            if not isinstance(exp_sha, str) or type(exp_bytes) is not int:
                return False
            blob_path = blob_dir / filename
            if not blob_path.is_file():
                return False
            actual_sha, actual_bytes = _sha256_file(blob_path)
            if actual_sha != exp_sha or actual_bytes != exp_bytes:
                return False
            dst_path = fuzz_dir / "bin" / filename
            if not dst_path.is_file() or _sha256_file(dst_path)[0] != exp_sha:
                to_restore.append((blob_path, dst_path))
        for src, dst in to_restore:
            _atomic_copy_file(src, dst, 0o755)
    return True


def _receipt_path(
    cache_dir: Path, action: str, arm: str, cpp_oracle: str, content_key: str
) -> Path:
    safe_arm = re.sub(r"[^A-Za-z0-9_.-]+", "_", arm)
    return (
        cache_dir
        / "receipts"
        / f"{action}--{safe_arm}--oracle{cpp_oracle}--{content_key[:16]}.json"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Content-addressed Cargo receipt cache"
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--enabled", default="1")
    parser.add_argument(
        "--action", choices=("clippy", "test", "fuzz-build"), required=True
    )
    parser.add_argument("--arm", required=True)
    parser.add_argument("--cpp-oracle", default="1")
    parser.add_argument("--features", default="")
    parser.add_argument("--fuzz-dir", type=Path, default=None)
    parser.add_argument("--fuzz-out", type=Path, default=None)
    parser.add_argument("--fuzz-bins", default="")
    parser.add_argument("--fuzz-triple", default="")
    parser.add_argument("--fuzz-rustflags", default="")
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise SystemExit("rust_receipt_cache.py: missing command after --")

    outputs: list[tuple[str, str]] = []
    if args.action == "fuzz-build":
        if args.fuzz_dir is None or args.fuzz_out is None:
            parser.error("fuzz-build requires --fuzz-dir and --fuzz-out")
        names = args.fuzz_bins.split() if args.arm == "cpp-oracle" else [args.arm]
        if (
            not names
            or len(names) != len(set(names))
            or any(
                not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) for name in names
            )
        ):
            parser.error("expected distinct fuzz target names")
        if (args.fuzz_dir / "bin").is_symlink():
            parser.error("fuzz output directory must be a directory within --fuzz-dir")
        outputs = (
            [(name, name) for name in names]
            if args.arm == "cpp-oracle"
            else [("wire", f"wire-{args.arm}")]
        )
    repo_root = args.repo_root.resolve()
    rust_dir = repo_root / "rust"
    enabled = str(args.enabled).strip() != "0"

    digests = None
    receipt_path = None
    if enabled:
        digests = _compute_key(
            repo_root=repo_root,
            action=args.action,
            arm=args.arm,
            cpp_oracle=args.cpp_oracle,
            features=args.features,
            fuzz_triple=args.fuzz_triple,
            fuzz_rustflags=args.fuzz_rustflags,
            command=cmd,
            fuzz_bins=args.fuzz_bins,
        )
        receipt_path = _receipt_path(
            args.cache_dir,
            args.action,
            args.arm,
            args.cpp_oracle,
            digests["content_key"],
        )
        if _verify_hit(
            receipt_path=receipt_path,
            cache_dir=args.cache_dir,
            action=args.action,
            arm=args.arm,
            content_key=digests["content_key"],
            fuzz_dir=args.fuzz_dir,
            expected_outputs=[name for _, name in outputs],
        ):
            print(
                f"[rust-receipt-cache] HIT {args.action}:{args.arm} "
                f"key={digests['content_key'][:12]}"
            )
            return 0

    env = os.environ.copy()
    if args.action == "fuzz-build" and args.fuzz_rustflags:
        env.pop("CARGO_ENCODED_RUSTFLAGS", None)
        for key in [
            k for k in env if k.startswith("CARGO_TARGET_") and k.endswith("_RUSTFLAGS")
        ]:
            env.pop(key, None)
        env["RUSTFLAGS"] = args.fuzz_rustflags

    proc = subprocess.run(cmd, cwd=str(rust_dir), env=env, check=False)
    if proc.returncode != 0:
        return proc.returncode

    artifacts_meta: list[dict[str, object]] = []
    if args.action == "fuzz-build":
        bin_dir = args.fuzz_dir / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)

        blob_dir = None
        if enabled and digests is not None:
            blob_dir = (
                args.cache_dir
                / "blobs"
                / f"{args.action}--{args.arm}"
                / digests["content_key"]
            )
        for src_name, dst_name in outputs:
            src = args.fuzz_out / src_name
            dst = bin_dir / dst_name
            _atomic_copy_file(src, dst, 0o755)
            sha, size = _sha256_file(dst)
            if blob_dir is not None:
                _atomic_copy_file(dst, blob_dir / dst_name, 0o755)
            artifacts_meta.append(
                {"rel_path": f"bin/{dst_name}", "sha256": sha, "bytes": size}
            )

    if enabled and digests is not None and receipt_path is not None:
        payload = {
            "schema_version": "1.0.0",
            "action": args.action,
            "arm": args.arm,
            "cpp_oracle": args.cpp_oracle,
            "features": args.features,
            "content_key": digests["content_key"],
            "toolchain_sha256": digests["toolchain_sha256"],
            "cargo_lock_sha256": digests["cargo_lock_sha256"],
            "rust_mk_sha256": digests["rust_mk_sha256"],
            "sources_sha256": digests["sources_sha256"],
            "status": "passed",
            "artifacts": artifacts_meta,
        }
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        _atomic_write_bytes(receipt_path, encoded, 0o600)
        print(
            f"[rust-receipt-cache] STORE {args.action}:{args.arm} "
            f"key={digests['content_key'][:12]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
