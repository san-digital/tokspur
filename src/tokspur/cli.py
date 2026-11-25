# SPDX-License-Identifier: Apache-2.0
"""Explicit preparation and foreground serving. No implicit service switch."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import secrets
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

from . import __version__

DEFAULT_CLI_IMPL = "rust"
from .local import (  # noqa: E402
    ROOT,
    candidate_file_specs,
    check_adoption,
    dflash2_profile_group_key,
    inference_processes,
    isolated_environment,
    port_available,
    private_dir,
    profile,
    read_json,
    state_lock,
    verify_asset,
    write_json,
)

DEFAULT_HF_BASE_URL = "https://huggingface.co"
QWEN_LICENCE_URL = (
    "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/"
    "34567a4712bc9766c4449e2e98e4468bfa24d915/LICENSE"
)


_PATH_COMPONENT = re.compile(r"[A-Za-z0-9._-]+")


def _repo_leaf_name(repo: str) -> str:
    parts = repo.split("/")
    if len(parts) != 2 or any(
        part in (".", "..") or not _PATH_COMPONENT.fullmatch(part) for part in parts
    ):
        raise ValueError(f"invalid repository identifier '{repo}'")
    return parts[1]


def _validate_asset_rel_path(name: str) -> Path:
    """A pinned asset name is a relative path of plain components (`pull.rs`)."""
    if not name:
        raise ValueError("asset filename must not be empty")
    if name.startswith("/"):
        raise ValueError("asset filename must be a relative path")
    for part in name.split("/"):
        if part in ("", ".", ".."):
            raise ValueError(f"asset filename must not traverse directories: {name}")
        if not _PATH_COMPONENT.fullmatch(part):
            raise ValueError(f"invalid asset filename component '{part}'")
    return Path(name)


def _ensure_no_symlink_components(path: Path, label: str) -> None:
    """No existing ancestor or leaf of `path` may be a symlink (`layout.rs`)."""
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a clean absolute path")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} must not traverse symlinks")
        if not current.exists():
            break


# The least `--max-filesize` given to curl. Before 8.11.0 curl also counts the
# cap against the body of each redirect it follows, and the model host answers
# `/resolve/` with a redirect whose body is a kilobyte or two, so a smaller cap
# fails the transfer on the redirect. The size check after the transfer still
# holds a file to its pin.
_MIN_TRANSFER_CAP = 64 << 10


def _partial_download_size(part_path: Path, name: str) -> int:
    """The bytes a resumable `.part` holds; curl is never pointed at anything but a plain file."""
    try:
        found = part_path.lstat()
    except FileNotFoundError:
        return 0
    if not stat.S_ISREG(found.st_mode):
        raise ValueError(f"Partial download path must be a regular file: {name}.part")
    if found.st_nlink != 1:
        raise ValueError(
            f"Partial download file must not have multiple hard links: {name}.part"
        )
    return found.st_size


def _preserved_dflash2_group(result: dict) -> dict | None:
    """The DFlash2 checkpoint the current adoption records, if it still checks out beside `result`."""
    with contextlib.suppress(FileNotFoundError, KeyError, ValueError):
        existing = read_json(ROOT / ".tokspur/adoption.json")
        if "dflash2_draft" in existing:
            check_adoption({**result, "dflash2_draft": existing["dflash2_draft"]})
            return existing["dflash2_draft"]
    return None


def _directory_has_candidate_assets(specs: dict, group: str, directory: Path) -> bool:
    if not directory.is_dir():
        return False
    return any(
        all((directory / entry["name"]).is_file() for entry in cand)
        for cand in candidate_file_specs(specs, group)
    )


def _verify_group_directory(
    specs: dict, profile_group: str, label: str, directory: Path, *, out=None
) -> dict:
    stream = sys.stdout if out is None else out
    directory = directory.resolve(strict=True)
    candidates = candidate_file_specs(specs, profile_group)
    if len(candidates) == 1:
        selected_specs = candidates[0]
    else:
        present = [
            cand
            for cand in candidates
            if all((directory / spec["name"]).exists() for spec in cand)
        ]
        if len(present) > 1:
            raise ValueError(f"Ambiguous {label} asset variants present in {directory}")
        selected_specs = present[0] if present else candidates[0]
    assets = []
    for spec in selected_specs:
        print(f"Verifying {label}: {spec['name']}", file=stream, flush=True)
        assets.append(verify_asset(directory / spec["name"], spec))
    return {"directory": str(directory), "assets": assets}


def adopt(
    target: Path,
    draft: Path | None = None,
    dflash2_dir: Path | None = None,
    *,
    out=None,
) -> None:
    stream = sys.stdout if out is None else out
    if draft is None:
        draft = target
    specs = read_json(ROOT / "profiles/flash-next-assets.json")
    result: dict = {"schema": 1, "verification": "full-sha256"}
    for group, directory in (("target", target), ("draft", draft)):
        result[group] = _verify_group_directory(
            specs, group, group, directory, out=stream
        )
    if dflash2_dir is not None:
        df_key = dflash2_profile_group_key(specs)
        if df_key is None:
            raise ValueError("flash-next-assets.json missing dflash2_draft")
        result["dflash2_draft"] = _verify_group_directory(
            specs, df_key, "dflash2", dflash2_dir, out=stream
        )
    else:
        preserved = _preserved_dflash2_group(result)
        if preserved is not None:
            result["dflash2_draft"] = preserved
    write_json(ROOT / ".tokspur/adoption.json", result)
    print(
        "Existing weights verified in place. No model files copied or downloaded.",
        file=stream,
    )


def valid_api_key(path: Path) -> bool:
    """Whether `path` holds a key the server admits: 64 hex digits and a newline."""
    try:
        status = path.lstat()
        text = path.read_bytes()
    except FileNotFoundError:
        return False
    return (
        (status.st_mode & 0o170777) == 0o100600
        and status.st_nlink == 1
        and len(text) == 65
        and text.endswith(b"\n")
        and all(c in b"0123456789abcdef" for c in text[:64])
    )


def configure(port: int, *, require_port_free: bool = True, out=None) -> None:
    stream = sys.stdout if out is None else out
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError("Use an unprivileged local port")
    if require_port_free and not port_available(port):
        raise ValueError(f"Port {port} is occupied; configuration left unchanged")
    data = read_json(ROOT / ".tokspur/adoption.json")
    check_adoption(data)
    p = profile()
    state = private_dir(ROOT / ".tokspur/state")
    if (state / "owner.json").exists() and read_json(state / "owner.json") != {
        "project": "tokspur",
        "schema": 1,
    }:
        raise ValueError("Unknown state owner")
    write_json(state / "owner.json", {"project": "tokspur", "schema": 1})

    # Keep a key only if the server would admit it (64 hex characters + newline,
    # 0600, one link); replace anything else atomically so an interrupted write
    # cannot leave an empty key behind.
    key_file = state / "api-key"
    if key_file.is_symlink():
        raise ValueError("state/api-key must not be a symlink")
    if not valid_api_key(key_file):
        temporary = state / f".api-key.{os.getpid()}.tmp"
        fd = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_WRONLY | os.O_CLOEXEC,
            0o600,
        )
        try:
            with os.fdopen(fd, "w") as key_stream:
                key_stream.write(secrets.token_hex(32) + "\n")
                key_stream.flush()
                os.fchmod(key_stream.fileno(), 0o600)
                os.fsync(key_stream.fileno())
            os.replace(temporary, key_file)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    target_dir = Path(data["target"]["directory"])
    if target_dir.is_dir():
        model_dir = private_dir(state / "models")
        link = model_dir / p["public_model"]
        if link.is_symlink():
            if link.resolve() != target_dir.resolve():
                link.unlink()
                link.symlink_to(target_dir, target_is_directory=True)
        elif link.exists():
            raise ValueError("Refusing to replace an existing model directory")
        else:
            link.symlink_to(target_dir, target_is_directory=True)

    write_json(
        ROOT / ".tokspur/configured.json",
        {
            "port": port,
            "profile": p,
            "model": p["public_model"],
        },
    )
    print(
        f"Configured tokspur at http://127.0.0.1:{port}; inference not started.",
        file=stream,
    )


def _resolve_hf_base_url() -> tuple[str, bool]:
    raw = os.environ.get("TOKSPUR_HF_BASE_URL", "").rstrip("/")
    if not raw:
        return DEFAULT_HF_BASE_URL, False
    for prefix in ("http://127.0.0.1:", "http://[::1]:"):
        if raw.startswith(prefix):
            port_s = raw[len(prefix) :]
            if port_s.isdigit() and 1 <= int(port_s) <= 65535:
                return raw, True
    raise ValueError(
        "TOKSPUR_HF_BASE_URL only permits loopback http://127.0.0.1:<port> or http://[::1]:<port>"
    )


def pull(
    directory: Path | None = None,
    *,
    dflash2: bool = False,
    dflash2_dir: Path | None = None,
    port: int | None = None,
) -> None:
    if port is not None and (type(port) is not int or not 1024 <= port <= 65535):
        raise ValueError("Use an unprivileged local port")
    base_url, allow_http = _resolve_hf_base_url()
    blocked = ("HF_", "HUGGING_FACE_", "HUGGINGFACE_")
    curl_env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(blocked)
        and k not in ("PYTHONPATH", "PYTHONHOME", "TOKSPUR_HF_BASE_URL")
    }
    curl_env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    curl_env["DO_NOT_TRACK"] = "1"

    include_dflash2 = bool(dflash2 or dflash2_dir is not None)
    with state_lock():
        specs = read_json(ROOT / "profiles/flash-next-assets.json")
        managed_root = private_dir(ROOT / ".tokspur/models")
        target_repo = specs["target"]["repo"]
        draft_repo = specs["draft"]["repo"]
        # Not resolved: a symlinked destination is refused, not followed.
        target_dest = (
            Path(directory).absolute()
            if directory is not None
            else (managed_root / _repo_leaf_name(target_repo))
        )
        draft_dest = (
            target_dest
            if (directory is not None or target_repo == draft_repo)
            else (managed_root / _repo_leaf_name(draft_repo))
        )

        print(
            f"Model licence: Qwen Community License Agreement 1.0 ({QWEN_LICENCE_URL})",
            flush=True,
        )
        if include_dflash2:
            print(
                "Drafter licence: Apache-2.0 checkpoint subject to Qwen Community License Agreement 1.0",
                flush=True,
            )

        cache: dict[Path, dict] = {}

        def _pull_group(group_key: str, dest: Path) -> dict:
            group_obj = specs[group_key]
            repo = group_obj["repo"]
            revision = group_obj["revision"]
            if len(revision) != 40 or any(
                c not in "0123456789abcdef" for c in revision
            ):
                raise ValueError(
                    f"asset profile requires a 40-character lowercase hex revision, found '{revision}'"
                )
            _ensure_no_symlink_components(dest, "model destination directory")
            dest.mkdir(parents=True, exist_ok=True)
            verified_assets = []
            for spec in group_obj["files"]:
                name = spec["name"]
                dest_path = dest / _validate_asset_rel_path(name)
                if dest_path in cache:
                    verified_assets.append(cache[dest_path])
                    continue
                # Checked again once it exists: curl writes through this path.
                _ensure_no_symlink_components(dest_path.parent, "download directory")
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                _ensure_no_symlink_components(dest_path.parent, "download directory")
                part_path = dest_path.parent / f"{dest_path.name}.part"
                if dest_path.is_symlink():
                    raise ValueError(f"Refusing symlink asset path: {name}")
                if dest_path.exists():
                    rec = verify_asset(dest_path, spec)
                    part_path.unlink(missing_ok=True)
                    print(f"Already verified: {name}", flush=True)
                    cache[dest_path] = rec
                    verified_assets.append(rec)
                    continue
                expected_size = spec["size"]
                part_len = _partial_download_size(part_path, name)
                if part_len > expected_size:
                    part_path.unlink()
                    part_len = 0
                if part_len < expected_size:
                    url = f"{base_url}/{repo}/resolve/{revision}/{name}"
                    print(f"Downloading {repo}/{name}...", flush=True)
                    # `-q` only works as the first argument: it keeps the
                    # caller's ~/.curlrc (netrc, headers, proxy, insecure) out
                    # of a download promised to send no tokens.
                    cmd = [
                        "/usr/bin/curl",
                        "-q",
                        "--fail",
                        "--location",
                        "--proto",
                        "=http,https" if allow_http else "=https",
                        "--retry",
                        "3",
                        "--retry-delay",
                        "2",
                        "--silent",
                        "--show-error",
                    ]
                    if part_len > 0:
                        cmd.extend(["--continue-at", "-"])
                    # curl counts the cap against the body of this response,
                    # so on a resume it is what the partial file still lacks.
                    cap = max(expected_size - part_len, _MIN_TRANSFER_CAP)
                    cmd.extend(["--max-filesize", str(cap)])
                    cmd.extend(["--output", str(part_path), "--", url])
                    subprocess.run(cmd, check=True, env=curl_env)
                print(f"Verifying: {name}", flush=True)
                try:
                    verify_asset(part_path, spec)
                except BaseException:
                    part_path.unlink(missing_ok=True)
                    raise
                os.replace(part_path, dest_path)
                rec = verify_asset(dest_path, spec)
                cache[dest_path] = rec
                verified_assets.append(rec)
            return {
                "directory": str(dest.resolve(strict=True)),
                "assets": verified_assets,
            }

        result: dict = {
            "schema": 1,
            "verification": "full-sha256",
            "target": _pull_group("target", target_dest),
            "draft": _pull_group("draft", draft_dest),
        }
        if include_dflash2:
            df_key = dflash2_profile_group_key(specs)
            if df_key is None:
                raise ValueError("flash-next-assets.json missing dflash2_draft")
            df_dest = (
                Path(dflash2_dir).absolute()
                if dflash2_dir is not None
                else (managed_root / _repo_leaf_name(specs[df_key]["repo"]))
            )
            result["dflash2_draft"] = _pull_group(df_key, df_dest)
        else:
            # A plain pull keeps the checkpoint an earlier pull or adopt recorded.
            preserved = _preserved_dflash2_group(result)
            if preserved is not None:
                result["dflash2_draft"] = preserved
        write_json(ROOT / ".tokspur/adoption.json", result)

        resolved_port = port
        if resolved_port is None:
            with contextlib.suppress(FileNotFoundError, KeyError):
                existing_cfg = read_json(ROOT / ".tokspur/configured.json")
                if isinstance(existing_cfg.get("port"), int):
                    resolved_port = existing_cfg["port"]
        if resolved_port is None:
            resolved_port = profile()["port"]
        configure(resolved_port, require_port_free=(port is not None))
        suffix = " --drafter dflash2" if include_dflash2 else ""
        print(f"Ready. Start serving with: tokspur serve --speed-preset{suffix}")


def default_derived_pack_path(
    root: Path | None = None, *, pack_v3: bool = False, prefer_v3: bool = False
) -> Path:
    base = ROOT if root is None else root
    derived_dir = base / ".tokspur" / "derived"
    if not pack_v3 and prefer_v3:
        for v3_name in ("tokspur-pack-v3.asa.gguf", "tokspur-pack-v3.gguf"):
            v3_path = derived_dir / v3_name
            if v3_path.is_file():
                return v3_path.resolve()
        # With neither format installed, name the default format (v3) in the refusal.
        if not any(
            (derived_dir / name).is_file()
            for name in ("tokspur-pack-v2.asa.gguf", "tokspur-pack-v2.gguf")
        ):
            return (derived_dir / "tokspur-pack-v3.asa.gguf").resolve()
    asa_name = "tokspur-pack-v3.asa.gguf" if pack_v3 else "tokspur-pack-v2.asa.gguf"
    legacy_name = "tokspur-pack-v3.gguf" if pack_v3 else "tokspur-pack-v2.gguf"
    asa_path = derived_dir / asa_name
    if asa_path.is_file():
        return asa_path.resolve()
    legacy_path = derived_dir / legacy_name
    if legacy_path.is_file():
        return legacy_path.resolve()
    return asa_path.resolve()


def _is_valid_dflash2_vocab_gather(value: str, vocabulary: int = 248320) -> bool:
    if value in ("control32k", "cjk48k", "cjk64k", "cjk80k"):
        return vocabulary > 248076
    if not value or len(value) > 65536 or vocabulary <= 0:
        return False
    prev_last = -1
    count = 0
    for part in value.split(","):
        if not part:
            return False
        if "-" in part:
            pieces = part.split("-")
            if len(pieces) != 2:
                return False
            first_s, last_s = pieces
        else:
            first_s = last_s = part
        for s in (first_s, last_s):
            if (
                not s
                or not s.isascii()
                or not s.isdigit()
                or (s[0] == "0" and len(s) > 1)
            ):
                return False
        first = int(first_s)
        last = int(last_s)
        if last < first or first <= prev_last or last >= vocabulary:
            return False
        count += last - first + 1
        prev_last = last
    return 256 <= count < vocabulary and count % 32 == 0


def serve(
    allow_raw_gguf: bool = False,
    pack_v3: bool = False,
    derived_pack: Path | str | None = None,
    gdn_prefill_only: bool = False,
    derived_pack_trust_mode: str = "full",
    output_packet: int | None = None,
    first_output_packet: int | None = None,
    prefill_dense: str | None = None,
    prefill_grouped: str | None = None,
    known_token_capacity: int | None = 2048,
    context_length: int = 65536,
    mtp_draft_slice: bool | None = None,
    verify_nax: bool | None = None,
    verify_nax_m8: bool | None = None,
    speed_preset: bool = False,
    mtp_depth: int | None = None,
    dflash2_precision: str | None = None,
    dflash2_max_drafts: int | None = None,
    dflash2_vocab_slice: int | None = None,
    dflash2_vocab_gather: str | None = None,
    native_host_logit_views: str | None = None,
    native_logit_topk: str | None = None,
    native_ple_async_gather: str | None = None,
    ple_ram_resident: bool | None = None,
    native_ple_ram_resident: str | None = None,
    native_mtp_fast_round: str | None = None,
    native_prefill_router_fast: str | None = None,
    native_verify_hc_gdn_fused: str | None = None,
    native_dflash2_defer_context: str | None = None,
    native_dflash2_verify_rows: str | None = None,
    native_prepared_validation: str | None = None,
    native_mtp_draft_slice: str | None = None,
    native_verify_nax: str | None = None,
    native_verify_nax_m8: str | None = None,
    drafter: str = "mtp",
    dflash2_dir: Path | str | None = None,
    concurrent_dispatch: bool = True,
    q8_gemv_rg4: str = "off",
    q8_three_msg: str = "off",
    prefill_multiseg: bool = False,
    prefill_swizzle: bool = True,
    qsa_prefill_fused: bool = True,
    hc_consumer_fused: bool = True,
    ple_overlap: bool = True,
    shared_prefix_cache: bool | None = None,
    parked_lanes_mib: int | None = None,
    mtp_lookup: str | None = None,
    mtp_lookup_match: int | None = None,
    dflash2_lookup: str | None = None,
    dflash2_lookup_match: int | None = None,
    tool_enum_validation: str | None = None,
    tool_arg_validation: str | None = None,
    private_prefix_trace: bool = False,
    telemetry_stdout: bool = False,
    checkpoint_dir: Path | str | None = None,
    checkpoint_cap_gib: float | None = None,
    checkpoint_floor_gib: float | None = None,
    checkpoint_policy: str | None = None,
    checkpoint_ssd: bool | None = None,
    checkpoint_ssd_budget_gib: float | None = None,
    checkpoint_ssd_floor_gib: float = 20.0,
    checkpoint_ssd_dir: str | Path | None = None,
    overlap_prefill: bool = False,
    model_dir: Path | str | None = None,
    port: int | None = None,
) -> None:
    if port is not None and (type(port) is not int or not 1024 <= port <= 65535):
        raise ValueError("Use an unprivileged local port")
    for flag, value in (
        ("native_host_logit_views", native_host_logit_views),
        ("native_logit_topk", native_logit_topk),
        ("native_ple_async_gather", native_ple_async_gather),
        ("native_ple_ram_resident", native_ple_ram_resident),
        ("native_mtp_fast_round", native_mtp_fast_round),
        ("native_prefill_router_fast", native_prefill_router_fast),
        ("native_verify_hc_gdn_fused", native_verify_hc_gdn_fused),
        ("native_dflash2_defer_context", native_dflash2_defer_context),
        ("native_prepared_validation", native_prepared_validation),
        ("native_mtp_draft_slice", native_mtp_draft_slice),
        ("native_verify_nax", native_verify_nax),
        ("native_verify_nax_m8", native_verify_nax_m8),
    ):
        if value not in (None, "off", "on"):
            raise ValueError(f"Unsupported {flag} mode: {value}")
    if gdn_prefill_only:
        raise ValueError(
            "--gdn-prefill-only was removed because its source could not be verified"
        )
    if native_prefill_router_fast == "on":
        raise ValueError(
            "--native-prefill-router-fast requires the removed router-window implementation"
        )
    if output_packet is None:
        output_packet = 64 if speed_preset else 8
    if prefill_dense is None:
        prefill_dense = "nax" if speed_preset else "f32-staged"
    if prefill_grouped is None:
        prefill_grouped = "nax" if speed_preset else "f32"
    if native_mtp_draft_slice is not None:
        mtp_draft_slice_enabled = native_mtp_draft_slice == "on"
    elif mtp_draft_slice is not None:
        mtp_draft_slice_enabled = bool(mtp_draft_slice)
    else:
        mtp_draft_slice_enabled = bool(speed_preset and drafter == "mtp")
    if native_ple_ram_resident is not None:
        ple_ram_resident_flag: str | None = native_ple_ram_resident
    elif ple_ram_resident is not None:
        ple_ram_resident_flag = "on" if ple_ram_resident else "off"
    else:
        ple_ram_resident_flag = None
    if native_verify_nax is not None:
        verify_nax_enabled = native_verify_nax == "on"
    elif verify_nax is not None:
        verify_nax_enabled = bool(verify_nax)
    else:
        verify_nax_enabled = bool(speed_preset)
    if native_verify_nax_m8 is not None:
        verify_nax_m8_enabled = native_verify_nax_m8 == "on"
        verify_nax_m8_explicit = True
    elif verify_nax_m8 is not None:
        verify_nax_m8_enabled = bool(verify_nax_m8)
        verify_nax_m8_explicit = True
    else:
        verify_nax_m8_enabled = False
        verify_nax_m8_explicit = False
    verify_nax_enabled = verify_nax_enabled or verify_nax_m8_enabled
    if speed_preset:
        if native_verify_hc_gdn_fused is None and verify_nax_enabled:
            native_verify_hc_gdn_fused = "on"
        if native_host_logit_views is None:
            native_host_logit_views = "on"
        if native_logit_topk is None:
            native_logit_topk = "on"
        if native_ple_async_gather is None:
            native_ple_async_gather = "on"
        if native_prepared_validation is None:
            native_prepared_validation = "on"
        if drafter == "mtp":
            if verify_nax_enabled and not verify_nax_m8_explicit:
                verify_nax_m8_enabled = True
            if native_mtp_fast_round is None:
                native_mtp_fast_round = "on"
            if mtp_depth is None:
                mtp_depth = 3
        elif drafter == "dflash2":
            if verify_nax_enabled and not verify_nax_m8_explicit:
                verify_nax_m8_enabled = True
            if dflash2_precision is None:
                dflash2_precision = "q8_0"
            if native_dflash2_defer_context is None:
                native_dflash2_defer_context = "on"
            if dflash2_max_drafts is None:
                dflash2_max_drafts = 7
            if dflash2_vocab_slice is None and dflash2_vocab_gather is None:
                dflash2_vocab_slice = 32768
            if native_dflash2_verify_rows is None:
                native_dflash2_verify_rows = "auto"
        if output_packet == 64 and first_output_packet is None:
            first_output_packet = 8
    if drafter == "dflash2" and native_dflash2_defer_context is None:
        native_dflash2_defer_context = "on"
    if tool_enum_validation is None:
        tool_enum_validation = "executor" if speed_preset else "strict"
    if tool_arg_validation is None:
        tool_arg_validation = "executor" if speed_preset else "strict"
    if native_mtp_fast_round == "on" and drafter != "mtp":
        raise ValueError(
            "--native-mtp-fast-round on is incompatible with --drafter dflash2"
        )
    if (
        native_prefill_router_fast == "on"
        and prefill_dense != "nax"
        and prefill_grouped != "nax"
    ):
        raise ValueError(
            "--native-prefill-router-fast on requires --prefill-dense nax or --prefill-grouped nax"
        )
    if (
        native_verify_hc_gdn_fused == "on"
        and not verify_nax_enabled
        and drafter != "dflash2"
    ):
        raise ValueError(
            "--native-verify-hc-gdn-fused on requires --verify-nax or --drafter dflash2"
        )
    if native_dflash2_verify_rows not in (None, "all", "auto"):
        raise ValueError(
            f"Unsupported native_dflash2_verify_rows: {native_dflash2_verify_rows} (expected all or auto)"
        )
    if native_dflash2_verify_rows is not None and drafter != "dflash2":
        raise ValueError("--native-dflash2-verify-rows requires --drafter dflash2")
    if dflash2_max_drafts is not None:
        if drafter != "dflash2":
            raise ValueError("--dflash2-max-drafts requires --drafter dflash2")
        if dflash2_max_drafts not in range(1, 16):
            raise ValueError(
                f"Unsupported dflash2_max_drafts: {dflash2_max_drafts} (expected 1..15)"
            )
    if dflash2_vocab_slice is not None:
        if drafter != "dflash2":
            raise ValueError("--dflash2-vocab-slice requires --drafter dflash2")
        if not (
            dflash2_vocab_slice == 0
            or (256 <= dflash2_vocab_slice < 248320 and dflash2_vocab_slice % 32 == 0)
        ):
            raise ValueError(
                f"Unsupported dflash2_vocab_slice: {dflash2_vocab_slice} (expected 0 or a multiple of 32 in 256..248319)"
            )
    if dflash2_vocab_gather is not None:
        if drafter != "dflash2":
            raise ValueError("--dflash2-vocab-gather requires --drafter dflash2")
        if dflash2_vocab_slice is not None:
            raise ValueError(
                "--dflash2-vocab-gather and --dflash2-vocab-slice are mutually exclusive"
            )
        if not isinstance(
            dflash2_vocab_gather, str
        ) or not _is_valid_dflash2_vocab_gather(dflash2_vocab_gather):
            raise ValueError(
                f"Unsupported dflash2_vocab_gather: {dflash2_vocab_gather} (expected control32k, cjk48k, cjk64k, cjk80k, or ascending IDs/ranges multiple of 32 in 256..248319)"
            )
    if mtp_depth not in (None, 2, 3):
        raise ValueError(f"Unsupported mtp_depth: {mtp_depth} (expected 2 or 3)")
    if dflash2_precision not in (None, "q8_0", "bf16"):
        raise ValueError(
            f"Unsupported dflash2_precision: {dflash2_precision} (expected q8_0 or bf16)"
        )
    if allow_raw_gguf and (pack_v3 or derived_pack is not None):
        raise ValueError(
            "--raw-gguf cannot be combined with --pack-v3 or --derived-pack"
        )
    if derived_pack_trust_mode not in ("full", "local-seal"):
        raise ValueError(
            f"Unsupported derived_pack_trust_mode: {derived_pack_trust_mode}"
        )
    if output_packet not in (8, 64):
        raise ValueError(f"Unsupported output_packet: {output_packet}")
    if first_output_packet is not None:
        if not (1 <= first_output_packet <= 64):
            raise ValueError(
                f"Unsupported first_output_packet: {first_output_packet} (expected 1..64)"
            )
        if first_output_packet > output_packet:
            raise ValueError(
                f"--first-output-packet ({first_output_packet}) cannot exceed --output-packet ({output_packet})"
            )
    if prefill_dense not in ("f32-staged", "nax", "f32", "f16"):
        raise ValueError(f"Unsupported prefill_dense mode: {prefill_dense}")
    if prefill_grouped not in ("f32", "auto", "serial", "nax"):
        raise ValueError(f"Unsupported prefill_grouped mode: {prefill_grouped}")
    dense_staged_or_nax = prefill_dense in ("f32-staged", "nax")
    if dense_staged_or_nax and prefill_grouped == "serial":
        raise ValueError("--prefill-grouped serial requires --prefill-dense f32 or f16")
    if prefill_grouped == "nax" and not dense_staged_or_nax:
        raise ValueError(
            "--prefill-grouped nax requires --prefill-dense f32-staged or nax"
        )
    resolved_known_capacity = (
        2048 if known_token_capacity is None else known_token_capacity
    )
    if resolved_known_capacity not in (512, 1024, 2048):
        raise ValueError(
            f"Unsupported known_token_capacity: {resolved_known_capacity} (expected 512, 1024 or 2048)"
        )
    if type(context_length) is not int or context_length not in (65536, 131072):
        raise ValueError(
            f"Unsupported context_length: {context_length} (expected 65536 or 131072)"
        )
    if drafter not in ("mtp", "dflash2"):
        raise ValueError(f"Unsupported drafter mode: {drafter}")
    if q8_gemv_rg4 not in ("off", "on", "wide"):
        raise ValueError(f"Unsupported q8_gemv_rg4 mode: {q8_gemv_rg4}")
    if q8_three_msg not in ("off", "on", "wide"):
        raise ValueError(f"Unsupported q8_three_msg mode: {q8_three_msg}")
    if mtp_lookup not in (None, "off", "bypass", "hybrid"):
        raise ValueError(f"Unsupported mtp_lookup mode: {mtp_lookup}")
    if mtp_lookup_match is not None and not (2 <= mtp_lookup_match <= 16):
        raise ValueError(
            f"Unsupported mtp_lookup_match: {mtp_lookup_match} (expected 2..16)"
        )
    if dflash2_lookup not in (None, "off", "bypass", "hybrid"):
        raise ValueError(f"Unsupported dflash2_lookup mode: {dflash2_lookup}")
    if dflash2_lookup_match is not None and not (2 <= dflash2_lookup_match <= 16):
        raise ValueError(
            f"Unsupported dflash2_lookup_match: {dflash2_lookup_match} (expected 2..16)"
        )
    if tool_enum_validation not in ("strict", "executor"):
        raise ValueError(
            f"Unsupported tool_enum_validation mode: {tool_enum_validation}"
        )
    if tool_arg_validation not in ("strict", "executor"):
        raise ValueError(f"Unsupported tool_arg_validation mode: {tool_arg_validation}")
    if checkpoint_policy not in (None, "recompute-cost", "lru"):
        raise ValueError(f"Unsupported checkpoint_policy: {checkpoint_policy}")
    if drafter == "dflash2":
        if mtp_depth is not None:
            raise ValueError("--mtp-depth is incompatible with --drafter dflash2")
        if mtp_draft_slice_enabled:
            raise ValueError("--mtp-draft-slice is incompatible with --drafter dflash2")
        # An explicit OFF is refused too, so it cannot be mistaken for a full DFlash2 draft vocabulary: that is a separate setting.
        if mtp_draft_slice is not None or native_mtp_draft_slice is not None:
            raise ValueError(
                "--no-mtp-draft-slice and --native-mtp-draft-slice do not apply to --drafter dflash2; use --dflash2-vocab-slice 0 for the full draft vocabulary"
            )
        if mtp_lookup not in (None, "off") or mtp_lookup_match is not None:
            raise ValueError("--mtp-lookup is incompatible with --drafter dflash2")
    elif dflash2_dir is not None:
        raise ValueError("--dflash2-dir requires --drafter dflash2")
    elif dflash2_precision is not None:
        raise ValueError("--dflash2-precision requires --drafter dflash2")
    elif dflash2_lookup not in (None, "off") or dflash2_lookup_match is not None:
        raise ValueError("--dflash2-lookup requires --drafter dflash2")
    effective_mtp_lookup = (
        ("hybrid" if drafter == "mtp" else "off") if mtp_lookup is None else mtp_lookup
    )
    effective_dflash2_lookup = (
        ("hybrid" if drafter == "dflash2" else "off")
        if dflash2_lookup is None
        else dflash2_lookup
    )

    if checkpoint_cap_gib is not None:
        effective_checkpoint_cap_gib = float(checkpoint_cap_gib)
    elif checkpoint_ssd is False:
        effective_checkpoint_cap_gib = 0.0
    elif checkpoint_ssd_budget_gib is not None:
        effective_checkpoint_cap_gib = float(checkpoint_ssd_budget_gib)
    elif checkpoint_ssd is True:
        effective_checkpoint_cap_gib = 8.0
    else:
        effective_checkpoint_cap_gib = 0.0

    effective_checkpoint_floor_gib = (
        float(checkpoint_floor_gib)
        if checkpoint_floor_gib is not None
        else float(checkpoint_ssd_floor_gib)
    )
    if not (0.0 <= effective_checkpoint_cap_gib <= 32.0):
        raise ValueError("checkpoint_cap_gib must be between 0.0 and 32.0 GiB")
    if not (0.0 <= effective_checkpoint_floor_gib <= 100.0):
        raise ValueError("checkpoint_floor_gib must be between 0.0 and 100.0 GiB")
    # Checkpoints are off until a positive capacity is given, so a checkpoint
    # setting without one would be dropped silently.
    if effective_checkpoint_cap_gib <= 0.0:
        for flag, given in (
            (
                "--checkpoint-dir",
                checkpoint_dir is not None or checkpoint_ssd_dir is not None,
            ),
            ("--checkpoint-policy", checkpoint_policy is not None),
            ("--checkpoint-floor-gib", checkpoint_floor_gib is not None),
        ):
            if given:
                raise ValueError(
                    f"{flag} has no effect unless --checkpoint-cap-gib is greater than 0"
                )
    effective_shared_prefix = (
        True if shared_prefix_cache is None else bool(shared_prefix_cache)
    )
    if parked_lanes_mib is None:
        effective_parked_lanes_mib = 4096 if effective_shared_prefix else 0
    else:
        effective_parked_lanes_mib = int(parked_lanes_mib)
    if not (0 <= effective_parked_lanes_mib <= 16384):
        raise ValueError("--parked-lanes-mib must be between 0 and 16384")
    if effective_parked_lanes_mib > 0 and not effective_shared_prefix:
        raise ValueError("--parked-lanes-mib requires the shared prefix cache")

    state = private_dir(ROOT / ".tokspur/state")
    with state_lock():
        auto_adopted = False
        if model_dir is not None:
            resolved_dir = Path(model_dir).resolve(strict=True)
            reuse_existing = False
            with contextlib.suppress(FileNotFoundError, KeyError, ValueError):
                existing_adopt = read_json(ROOT / ".tokspur/adoption.json")
                if (
                    existing_adopt.get("target", {}).get("directory")
                    == str(resolved_dir)
                    and existing_adopt.get("draft", {}).get("directory")
                    == str(resolved_dir)
                    and (
                        dflash2_dir is None
                        or existing_adopt.get("dflash2_draft", {}).get("directory")
                        == str(Path(dflash2_dir).resolve(strict=True))
                    )
                ):
                    check_adoption(existing_adopt)
                    adoption = existing_adopt
                    reuse_existing = True
            if not reuse_existing:
                auto_adopted = True
                adopt(
                    resolved_dir,
                    resolved_dir,
                    Path(dflash2_dir) if dflash2_dir is not None else None,
                    out=sys.stderr,
                )
                adoption = read_json(ROOT / ".tokspur/adoption.json")
        else:
            try:
                adoption = read_json(ROOT / ".tokspur/adoption.json")
            except FileNotFoundError:
                specs = read_json(ROOT / "profiles/flash-next-assets.json")
                managed_root = ROOT / ".tokspur/models"
                managed_target = managed_root / _repo_leaf_name(specs["target"]["repo"])
                managed_draft = managed_root / _repo_leaf_name(specs["draft"]["repo"])
                if _directory_has_candidate_assets(
                    specs, "target", managed_target
                ) and _directory_has_candidate_assets(specs, "draft", managed_draft):
                    auto_df: Path | None = (
                        Path(dflash2_dir) if dflash2_dir is not None else None
                    )
                    if auto_df is None:
                        df_key = dflash2_profile_group_key(specs)
                        if df_key is not None and "repo" in specs[df_key]:
                            cand_df = managed_root / _repo_leaf_name(
                                specs[df_key]["repo"]
                            )
                            if _directory_has_candidate_assets(specs, df_key, cand_df):
                                auto_df = cand_df
                    auto_adopted = True
                    adopt(
                        managed_target,
                        managed_draft,
                        auto_df,
                        out=sys.stderr,
                    )
                    adoption = read_json(ROOT / ".tokspur/adoption.json")
                else:
                    raise ValueError(
                        "No adopted model weights found. Run 'tokspur pull' or 'tokspur serve /path/to/Qwen3.8-Flash-Next-Q4_K-Tokspur-GGUF'."
                    ) from None
            else:
                if dflash2_dir is not None and adoption.get("dflash2_draft", {}).get(
                    "directory"
                ) != str(Path(dflash2_dir).resolve(strict=True)):
                    # An explicit checkpoint the adoption does not record is
                    # hashed against the pinned profile and recorded, as the
                    # two other branches do. It replaces any recorded before.
                    specs = read_json(ROOT / "profiles/flash-next-assets.json")
                    df_key = dflash2_profile_group_key(specs)
                    if df_key is None:
                        raise ValueError("flash-next-assets.json missing dflash2_draft")
                    adoption = {
                        **adoption,
                        "dflash2_draft": _verify_group_directory(
                            specs, df_key, "dflash2", Path(dflash2_dir), out=sys.stderr
                        ),
                    }
                    check_adoption(adoption)
                    write_json(ROOT / ".tokspur/adoption.json", adoption)
        check_adoption(adoption)

        if port is not None or model_dir is not None or auto_adopted:
            cfg_port = port
            if cfg_port is None:
                with contextlib.suppress(FileNotFoundError, KeyError):
                    prev_cfg = read_json(ROOT / ".tokspur/configured.json")
                    if isinstance(prev_cfg.get("port"), int):
                        cfg_port = prev_cfg["port"]
            if cfg_port is None:
                cfg_port = profile()["port"]
            configure(cfg_port, out=sys.stderr)
            config = read_json(ROOT / ".tokspur/configured.json")
        else:
            try:
                config = read_json(ROOT / ".tokspur/configured.json")
            except FileNotFoundError:
                configure(profile()["port"], out=sys.stderr)
                config = read_json(ROOT / ".tokspur/configured.json")

        if drafter == "dflash2" and dflash2_dir is None:
            adopted_df = adoption.get("dflash2_draft", {}).get("directory")
            if adopted_df:
                dflash2_dir = adopted_df
            else:
                with contextlib.suppress(FileNotFoundError, KeyError):
                    specs = read_json(ROOT / "profiles/flash-next-assets.json")
                    df_key = dflash2_profile_group_key(specs)
                    if df_key is not None and "repo" in specs[df_key]:
                        cand_df = (
                            ROOT
                            / ".tokspur/models"
                            / _repo_leaf_name(specs[df_key]["repo"])
                        )
                        if _directory_has_candidate_assets(specs, df_key, cand_df):
                            df_group = _verify_group_directory(
                                specs, df_key, "dflash2", cand_df, out=sys.stderr
                            )
                            adoption = {**adoption, "dflash2_draft": df_group}
                            check_adoption(adoption)
                            write_json(ROOT / ".tokspur/adoption.json", adoption)
                            dflash2_dir = df_group["directory"]
            if dflash2_dir is None:
                raise ValueError(
                    "--drafter dflash2 requires --dflash2-dir (or run 'tokspur pull --dflash2')"
                )
        if not port_available(config["port"]):
            raise ValueError("Requested port is occupied")
        workers = inference_processes()
        if workers:
            raise ValueError(
                f"Other inference workers found: {workers}. Stop them explicitly first."
            )
        import psutil

        if psutil.virtual_memory().total < 128 * 1024**3:
            raise ValueError(
                "The selected 96 GiB profile requires 128 GiB unified memory"
            )
        # Validate source provenance before loading hundreds of GB of weights.
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/bootstrap.py"), "--verify-only"],
            check=True,
        )
        # The server reads and checks state/api-key itself.
        env = isolated_environment()

        server_bin = None
        for candidate in (
            ROOT / "build/rust-fast-math/release/tokspur-server",
            ROOT / "build/rust-target/release/tokspur-server",
            ROOT / "rust/target/release/tokspur-server",
        ):
            if candidate.is_file():
                server_bin = candidate
                break
        if server_bin is None:
            raise FileNotFoundError(
                "Rust release tokspur-server binary not found; build it with make -C native -f rust.mk rust-server"
            )

        build_dir = ROOT / "build/owned-native"
        worker_bin = build_dir / "scalar-channel-target"
        if not worker_bin.is_file():
            worker_bin = build_dir / "tokspur-native"

        metallib = build_dir / "tokspur-owned.metallib"
        policy = ROOT / "profiles/owned-target-experimental.json"
        if not policy.is_file():
            policy = ROOT / "profiles/m5-max-128.json"

        target_info = adoption.get("target", {})
        target_dir = Path(target_info.get("directory", ""))
        # The pair the Rust CLI derives from the verified GGUF metadata comes
        # first, as in `serve.rs`.
        derived_text = state / "gguf" / "derived-text"
        tokenizer = derived_text / "tokenizer.json"
        template = derived_text / "chat_template.jinja"
        if not tokenizer.is_file() or not template.is_file():
            tokenizer = target_dir / "tokenizer.json"
            template = target_dir / "chat_template.jinja"
        if not tokenizer.is_file() or not template.is_file():
            gguf_dir = state / "gguf"
            candidates: list[Path] = []
            internal_model = config.get("internal_model")
            if internal_model and gguf_dir.is_dir():
                candidates.extend(sorted(gguf_dir.glob(f"{internal_model}*")))
            if gguf_dir.is_dir():
                candidates.extend(
                    sorted(
                        p
                        for p in gguf_dir.iterdir()
                        if p.is_dir() and p not in candidates
                    )
                )
            if target_dir.parent.is_dir() and target_dir.parent not in candidates:
                candidates.append(target_dir.parent)
            for candidate in candidates:
                cand_tok = candidate / "tokenizer.json"
                cand_tmpl = candidate / "chat_template.jinja"
                if cand_tok.is_file() and not tokenizer.is_file():
                    tokenizer = cand_tok
                if cand_tmpl.is_file() and not template.is_file():
                    template = cand_tmpl
                if tokenizer.is_file() and template.is_file():
                    break
        if not tokenizer.is_file() or not template.is_file():
            # This CLI cannot derive the pair, so a pull or adopt done only
            # here leaves none; the server would be handed a missing path.
            raise FileNotFoundError(
                "Tokenizer and chat template not found. The Python reference CLI does not derive them; "
                "run 'tokspur pull' or 'tokspur adopt' once with the Rust CLI (TOKSPUR_CLI_IMPL unset) "
                f"to write them to {derived_text}"
            )

        shards = []
        adopted_pack: Path | None = None
        for asset in target_info.get("assets", []):
            if asset.get("role") == "derived_pack":
                if adopted_pack is not None:
                    raise ValueError("multiple derived packs in adopted target assets")
                adopted_pack = Path(asset["path"])
            else:
                shards.append(asset["path"])
        if adopted_pack is not None and allow_raw_gguf:
            raise ValueError(
                "--raw-gguf is unavailable for pack-only adopted weights (no raw GPU GGUF shards)"
            )

        if derived_pack is not None:
            dp = Path(derived_pack)
            pack_file = dp if dp.is_absolute() else (Path.cwd() / dp).resolve()
        elif adopted_pack is not None:
            pack_file = adopted_pack
        else:
            pack_file = default_derived_pack_path(ROOT, pack_v3=pack_v3, prefer_v3=True)

        if not allow_raw_gguf:
            if not pack_file.is_file():
                raise FileNotFoundError(
                    f"Lossless derived weight pack not found at {pack_file}; "
                    "run pack construction or pass --raw-gguf to allow raw GGUF fallback"
                )
            derived_pack_flags = [
                "--derived-pack",
                str(pack_file),
                "--derived-pack-auth-workers",
                "16",
                *(
                    ["--derived-pack-trust-mode", derived_pack_trust_mode]
                    if derived_pack_trust_mode != "full"
                    else []
                ),
            ]
        else:
            derived_pack_flags = []

        if drafter == "dflash2":
            assert dflash2_dir is not None
            dflash2_path = Path(dflash2_dir).resolve()
            if not dflash2_path.is_dir():
                raise FileNotFoundError(
                    f"DFlash2 checkpoint directory not found: {dflash2_path}"
                )
            # The native drafter hashes only config.json, so the weights are
            # pinned here: the worker is given a directory only if the checked
            # adoption records it.
            if adoption.get("dflash2_draft", {}).get("directory") != str(dflash2_path):
                raise ValueError(
                    "DFlash2 checkpoint directory is not the verified adoption; re-run 'tokspur adopt' with --dflash2-dir"
                )
            speculative_flags = [
                "--native-drafter",
                "dflash2",
                "--native-dflash2-dir",
                str(dflash2_path),
                "--native-mtp-depth",
                "7",
                "--native-mtp-request-carry",
                "on",
                "--native-mtp-draft-attention",
                "split16",
                *(
                    ["--native-dflash2-precision", dflash2_precision]
                    if dflash2_precision is not None
                    else []
                ),
                *(
                    ["--native-dflash2-max-drafts", str(dflash2_max_drafts)]
                    if dflash2_max_drafts is not None
                    else []
                ),
                *(
                    ["--native-dflash2-vocab-slice", str(dflash2_vocab_slice)]
                    if dflash2_vocab_slice is not None
                    else []
                ),
                *(
                    ["--native-dflash2-vocab-gather", dflash2_vocab_gather]
                    if dflash2_vocab_gather is not None
                    else []
                ),
            ]
        else:
            draft_assets = adoption.get("draft", {}).get("assets", [])
            mtp_asset = None
            for asset in draft_assets:
                if (
                    asset.get("sha256")
                    == "5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6"
                ):
                    mtp_asset = asset
                    break
            if mtp_asset is None and len(draft_assets) == 1:
                mtp_asset = draft_assets[0]

            if mtp_asset is None:
                raise ValueError(
                    "Adopted draft assets do not contain verified MTP sidecar (blk.48); re-run tokspur adopt"
                )
            mtp_sidecar = Path(mtp_asset.get("path", ""))
            if not mtp_sidecar.is_file():
                raise FileNotFoundError(
                    f"Adopted MTP sidecar file not found: {mtp_sidecar}"
                )
            speculative_flags = [
                "--native-mtp-sidecar",
                str(mtp_sidecar),
                "--native-mtp-depth",
                str(2 if mtp_depth is None else mtp_depth),
                "--native-mtp-request-carry",
                "on",
                "--native-mtp-draft-attention",
                "tiled128",
            ]

        staged_carrier = (
            "on"
            if dense_staged_or_nax and prefill_grouped in ("f32", "auto", "nax")
            else "off"
        )
        raw_ckpt_dir = checkpoint_dir or checkpoint_ssd_dir
        if raw_ckpt_dir is None:
            resolved_ckpt_dir = state / "checkpoints"
        else:
            ckpt_path = Path(raw_ckpt_dir)
            resolved_ckpt_dir = (
                ckpt_path if ckpt_path.is_absolute() else (Path.cwd() / ckpt_path)
            )

        carrier_flags = [
            "--native-scalar-channel",
            "on",
            "--native-fast-math",
            "channel",
            "--native-qsa-stage-v8",
            "on",
            *speculative_flags,
            "--native-mtp-output-packet",
            str(output_packet),
            *(
                ["--native-mtp-first-output-packet", str(first_output_packet)]
                if first_output_packet is not None
                else []
            ),
            "--native-qsa-sparse",
            "off",
            "--native-gdn-ported",
            "off",
            "--native-prefill-input512",
            "on",
            "--native-known-token-capacity",
            str(resolved_known_capacity),
            *(
                ["--context-length", str(context_length)]
                if context_length != 65536
                else []
            ),
            *(["--native-raw-decode", "on"] if allow_raw_gguf else []),
            *(["--native-mtp-draft-slice", "on"] if mtp_draft_slice_enabled else []),
            *(["--native-verify-nax", "on"] if verify_nax_enabled else []),
            *(
                ["--native-verify-nax-m8", "on"]
                if verify_nax_m8_enabled
                else (
                    ["--native-verify-nax-m8", "off"] if verify_nax_m8_explicit else []
                )
            ),
            *(
                ["--native-dflash2-verify-rows", native_dflash2_verify_rows]
                if native_dflash2_verify_rows is not None
                else []
            ),
            *(
                ["--native-prepared-validation", "on"]
                if native_prepared_validation == "on"
                else []
            ),
            *(
                ["--native-host-logit-views", native_host_logit_views]
                if native_host_logit_views is not None
                else []
            ),
            *(
                ["--native-logit-topk", native_logit_topk]
                if native_logit_topk is not None
                else []
            ),
            *(
                ["--native-ple-async-gather", native_ple_async_gather]
                if native_ple_async_gather is not None
                else []
            ),
            *(
                ["--native-ple-ram-resident", ple_ram_resident_flag]
                if ple_ram_resident_flag is not None
                else []
            ),
            *(
                ["--native-mtp-fast-round", native_mtp_fast_round]
                if native_mtp_fast_round is not None
                else []
            ),
            *(
                ["--native-prefill-router-fast", native_prefill_router_fast]
                if native_prefill_router_fast is not None
                else []
            ),
            *(
                ["--native-verify-hc-gdn-fused", native_verify_hc_gdn_fused]
                if native_verify_hc_gdn_fused is not None
                else []
            ),
            *(
                ["--native-dflash2-defer-context", native_dflash2_defer_context]
                if native_dflash2_defer_context is not None
                else []
            ),
            "--native-prefill-dense",
            prefill_dense,
            "--native-prefill-grouped",
            prefill_grouped,
            "--native-prefill-hc-rows",
            staged_carrier,
            "--native-prefill-router-window",
            "off",
            "--native-prefill-hc-down-window",
            "off",
            "--native-prefill-hc-down-f32",
            "off",
            "--native-prefill-final-logits",
            "on",
            "--native-q8-three-rows",
            "on",
            *(["--native-concurrent-dispatch", "on"] if concurrent_dispatch else []),
            *(["--native-q8-gemv-rg4", q8_gemv_rg4] if q8_gemv_rg4 != "off" else []),
            *(["--native-q8-three-msg", q8_three_msg] if q8_three_msg != "off" else []),
            *(["--native-prefill-multiseg", "on"] if prefill_multiseg else []),
            *(["--native-prefill-swizzle", "on"] if prefill_swizzle else []),
            *(["--native-qsa-prefill-fused", "on"] if qsa_prefill_fused else []),
            *(["--native-hc-consumer-fused", "on"] if hc_consumer_fused else []),
            *(["--native-ple-overlap", "on"] if ple_overlap else []),
            "--shared-prefix-cache",
            "on" if effective_shared_prefix else "off",
            "--native-parked-lanes-mib",
            str(effective_parked_lanes_mib),
            *(
                ["--native-mtp-lookup", effective_mtp_lookup]
                if (
                    effective_mtp_lookup != "off"
                    or (drafter == "mtp" and mtp_lookup == "off")
                )
                else []
            ),
            *(
                ["--native-mtp-lookup-match", str(mtp_lookup_match)]
                if mtp_lookup_match is not None
                else []
            ),
            *(
                ["--native-dflash2-lookup", effective_dflash2_lookup]
                if (
                    effective_dflash2_lookup != "off"
                    or (drafter == "dflash2" and dflash2_lookup == "off")
                )
                else []
            ),
            *(
                ["--native-dflash2-lookup-match", str(dflash2_lookup_match)]
                if dflash2_lookup_match is not None
                else []
            ),
            "--tool-enum-validation",
            tool_enum_validation,
            *(
                ["--tool-arg-validation", tool_arg_validation]
                if tool_arg_validation != "strict"
                else []
            ),
            *(["--private-prefix-trace", "on"] if private_prefix_trace else []),
            *(["--telemetry-stdout", "on"] if telemetry_stdout else []),
            *(
                [
                    "--checkpoint-ssd",
                    "on",
                    "--checkpoint-cap-gib",
                    str(effective_checkpoint_cap_gib),
                    "--checkpoint-floor-gib",
                    str(effective_checkpoint_floor_gib),
                    "--checkpoint-dir",
                    str(resolved_ckpt_dir),
                    "--checkpoint-policy",
                    checkpoint_policy or "recompute-cost",
                ]
                if effective_checkpoint_cap_gib > 0.0
                else []
            ),
            *(["--overlap-prefill", "on"] if overlap_prefill else []),
        ]

        command = [
            str(server_bin),
            "--worker",
            str(worker_bin),
            "--policy",
            str(policy),
            "--library",
            str(metallib),
            "--tokenizer",
            str(tokenizer),
            "--template",
            str(template),
            "--state-dir",
            str(state),
            "--model-alias",
            config.get("model", "tokspur"),
            "--port",
            str(config["port"]),
            "--allow-full-model",
            "true",
            *derived_pack_flags,
            *carrier_flags,
            *shards,
        ]
        child = None
        previous = {}
        owned = {}
        stop_requested = False

        def group_signal(number):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, number)

        def stop(number, frame):
            nonlocal stop_requested
            # Signal callbacks only request shutdown. Process operations and
            # their bounded TERM/KILL waits happen in one cleanup path below.
            # Repeated signals cannot interrupt that drain or reenter Popen.
            stop_requested = True

        def live_owned():
            remaining = []
            for (pid, created), owned_proc in owned.items():
                try:
                    if (
                        owned_proc.is_running()
                        and owned_proc.create_time() == created
                        and owned_proc.status() != psutil.STATUS_ZOMBIE
                    ):
                        remaining.append((pid, owned_proc))
                except psutil.NoSuchProcess:
                    continue
            return remaining

        def drain_owned(seconds):
            deadline = time.monotonic() + seconds
            while True:
                remaining = live_owned()
                if not remaining or time.monotonic() >= deadline:
                    return remaining
                time.sleep(0.1)

        # Install the handlers before the child exists, so an early Ctrl-C is
        # recorded and acted on instead of raised between Popen and the loop.
        for number in (signal.SIGINT, signal.SIGTERM):
            previous[number] = signal.signal(number, stop)
        try:
            # Reuse the pinned supervisor primitive: an abruptly killed tokspur
            # parent must not leave its API (and therefore native worker) orphaned.
            # A signal delivered after handler installation but before launch
            # must not start a model just to terminate it. If it arrives during
            # Popen, the assigned child still reaches the same bounded finally.
            if not stop_requested:
                child = subprocess.Popen(command, env=env, start_new_session=True)
                write_json(
                    state / "process.json",
                    {"pid": child.pid, "supervisor": os.getpid()},
                )
            while child is not None and not stop_requested:
                try:
                    for process in psutil.Process(child.pid).children(recursive=True):
                        # macOS XPC shader compiler services are system-owned,
                        # not model workers; do not manage their lifetime.
                        if not process.exe().startswith(("/System/", "/usr/libexec/")):
                            owned[(process.pid, process.create_time())] = process
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                try:
                    result = child.wait(timeout=0.2)
                    break
                except subprocess.TimeoutExpired:
                    continue
        finally:
            try:
                if child is not None:
                    if child.poll() is None:
                        group_signal(signal.SIGTERM)
                        try:
                            child.wait(timeout=60)
                        except subprocess.TimeoutExpired:
                            # The API group ignored SIGTERM; never leave it running.
                            group_signal(signal.SIGKILL)
                            child.wait(timeout=15)
                    remaining = drain_owned(45)
                    if remaining:
                        # Model workers hold read-only mappings, so a forced exit
                        # loses nothing; an orphan would block the one-model rule.
                        for _pid, process in remaining:
                            with contextlib.suppress(
                                psutil.NoSuchProcess, psutil.AccessDenied
                            ):
                                process.kill()
                        remaining = drain_owned(15)
                    if remaining:
                        pids = [pid for pid, _process in remaining]
                        raise ValueError(
                            f"Owned processes have not exited: {pids}; retained process receipt"
                        )
                    (state / "process.json").unlink(missing_ok=True)
            finally:
                # Keep our non-raising handlers installed through every wait,
                # escalation and descendant drain, even when cleanup fails.
                for number, handler in previous.items():
                    signal.signal(number, handler)
        raise SystemExit(0 if stop_requested else result)


def main() -> None:
    if os.environ.get("TOKSPUR_CLI_IMPL") == "python" and any(
        arg.split("=", 1)[0]
        in {
            "--native-prepared-validation",
            "--native-verify-cost-profile",
            "--native-dflash2-draft-policy",
            "--native-verify-m8-splits",
            "--worker-burst-pacing",
        }
        for arg in sys.argv[1:]
    ):
        raise SystemExit("these runtime-check controls require the Rust CLI")
    mode = os.environ.get("TOKSPUR_CLI_IMPL") or DEFAULT_CLI_IMPL
    if mode not in ("rust", "python"):
        sys.stderr.write(
            "tokspur: invalid TOKSPUR_CLI_IMPL (expected rust or python)\n"
        )
        raise SystemExit(2)
    if mode == "rust" or sys.argv[1:2] == ["check-model-config"]:
        rust_bin = ROOT / "build/rust-fast-math/release/tokspur"
        if not rust_bin.is_file():
            sys.stderr.write(
                f"tokspur: Rust CLI binary not found at {rust_bin}; run scripts/bootstrap.py\n"
            )
            raise SystemExit(1)
        os.execv(str(rust_bin), [str(rust_bin), *sys.argv[1:]])

    parser = argparse.ArgumentParser(description="Isolated tokspur runtime experiments")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Read-only readiness summary; never loads a model")
    sub.add_parser(
        "check-model-config",
        help="Compare a local config.json with the Flash-Next text profile",
    )
    p_pull = sub.add_parser(
        "pull",
        help="Download and verify pinned weights; configures local state",
    )
    p_pull.add_argument("dir_pos", nargs="?", type=Path, default=None)
    p_pull.add_argument("--dir", dest="dir_flag", type=Path, default=None)
    p_pull.add_argument(
        "--dflash2",
        "--all",
        dest="dflash2",
        action="store_true",
        default=False,
    )
    p_pull.add_argument("--dflash2-dir", type=Path, default=None)
    p_pull.add_argument("--port", type=int, default=None)
    a = sub.add_parser(
        "adopt", help="Verify existing weights in place without copying them"
    )
    a.add_argument("dir_pos", nargs="?", type=Path, default=None)
    a.add_argument("--target", type=Path, required=False, default=None)
    a.add_argument("--draft", type=Path, required=False, default=None)
    a.add_argument("--dflash2-dir", type=Path, required=False, default=None)
    a.add_argument(
        "--check-only",
        action="store_true",
        default=False,
        help="Check existing adopted weights without rehashing",
    )
    c = sub.add_parser(
        "configure", help="Write isolated settings; never starts inference"
    )
    c.add_argument("--port", type=int, default=profile()["port"])
    sub.add_parser("plan", help="Show the selected native policy")
    sub.add_parser(
        "verify-build", help="Verify build-manifest.json and launch artefacts"
    )
    s = sub.add_parser("serve", help="Foreground API; refuses other inference workers")
    s.add_argument("model_dir_pos", nargs="?", type=Path, default=None)
    s.add_argument("--model-dir", dest="model_dir_flag", type=Path, default=None)
    s.add_argument("--port", type=int, default=None)
    s.add_argument(
        "--raw-gguf",
        action="store_true",
        default=False,
        help="Explicit opt-in to raw GGUF instead of derived pack",
    )
    s.add_argument(
        "--pack-v3",
        action="store_true",
        default=False,
        help="Require planar derived pack v3 (the default when installed; v2 is the fallback)",
    )
    s.add_argument(
        "--derived-pack",
        type=Path,
        default=None,
        help="Explicit derived pack GGUF file path",
    )
    s.add_argument(
        "--gdn-prefill-only",
        action="store_true",
        default=False,
        help="Retired GDN prefill experiment",
    )
    s.add_argument(
        "--derived-pack-trust-mode",
        choices=["full", "local-seal"],
        default="full",
        help=(
            "Derived pack verification at start (default: full). full reads and "
            "checks the whole pack. local-seal skips that when the seal beside the "
            "pack records this exact file (inode, size and times): the file passed "
            "full verification and is unchanged since. The seal also records which "
            "source files the pack was compared with and is accepted only by a start "
            "that has no source beyond those; otherwise that start does the full "
            "verification and widens the seal. Any change to the pack file costs one "
            "full verification. No protection against a process running as the same "
            "user, which can rewrite both files"
        ),
    )
    s.add_argument(
        "--output-packet",
        type=int,
        default=None,
        choices=[8, 64],
        help="MTP output packet size (8 or 64)",
    )
    s.add_argument(
        "--first-output-packet",
        "--native-mtp-first-output-packet",
        dest="first_output_packet",
        type=int,
        default=None,
        help="Opt-in initial speculative output packet cap (1..64) before steady-state packets",
    )
    s.add_argument(
        "--prefill-dense",
        choices=["f32-staged", "nax", "f32", "f16"],
        default=None,
        help="Dense Q8 prefill kernel mode (default: f32-staged)",
    )
    s.add_argument(
        "--prefill-grouped",
        choices=["f32", "auto", "serial", "nax"],
        default=None,
        help="Grouped MoE prefill kernel mode (default: f32)",
    )
    s.add_argument(
        "--known-token-capacity",
        type=int,
        choices=[512, 1024, 2048],
        default=2048,
        help="Known-token IPC packet capacity (default: 2048)",
    )
    s.add_argument(
        "--context-length",
        "--native-context-length",
        dest="context_length",
        type=int,
        choices=[65536, 131072],
        default=65536,
        help="Active prompt and generation context length (default: 65536)",
    )
    s.add_argument(
        "--mtp-draft-slice",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Opt-in top-32K MTP draft vocabulary slice (default: off)",
    )
    s.add_argument(
        "--native-mtp-draft-slice",
        choices=["off", "on"],
        default=None,
        help="Explicit native top-32K MTP draft vocabulary slice (default: off)",
    )
    s.add_argument(
        "--verify-nax",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Opt-in Metal 4 NAX batched speculative verification (default: off)",
    )
    s.add_argument(
        "--native-verify-nax",
        choices=["off", "on"],
        default=None,
        help="Explicit native batched verification mode (default: off)",
    )
    s.add_argument(
        "--verify-nax-m8",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Opt-in Metal 4 NAX m=8 dense tensor-ops speculative verification (default: off)",
    )
    s.add_argument(
        "--native-verify-nax-m8",
        choices=["off", "on"],
        default=None,
        help="Explicit native m=8 dense tensor-ops verification mode (default: off)",
    )
    s.add_argument(
        "--speed-preset",
        action="store_true",
        default=False,
        help="Opt-in Pack v3, NAX, MTP3/DFlash2 and executor speed preset",
    )
    s.add_argument(
        "--mtp-depth",
        type=int,
        choices=[2, 3],
        default=None,
        help="Opt-in MTP speculative lookahead depth (default: 2)",
    )
    s.add_argument(
        "--dflash2-precision",
        choices=["q8_0", "bf16"],
        default=None,
        help="Opt-in DFlash2 dense projection precision (default: q8_0)",
    )
    s.add_argument(
        "--dflash2-max-drafts",
        "--native-dflash2-max-drafts",
        dest="dflash2_max_drafts",
        type=int,
        choices=range(1, 16),
        default=None,
        help="Opt-in DFlash2 maximum drafted proposals per round (1..15)",
    )
    s.add_argument(
        "--dflash2-vocab-slice",
        "--native-dflash2-vocab-slice",
        dest="dflash2_vocab_slice",
        type=int,
        default=None,
        help="Opt-in DFlash2 draft vocabulary slice (0 or multiple of 32)",
    )
    s.add_argument(
        "--dflash2-vocab-gather",
        "--native-dflash2-vocab-gather",
        dest="dflash2_vocab_gather",
        default=None,
        help="Opt-in DFlash2 gathered draft vocabulary (control32k, cjk48k, cjk64k, cjk80k, or ID ranges)",
    )
    s.add_argument(
        "--native-host-logit-views",
        choices=["off", "on"],
        default=None,
        help="Opt-in native host logit views (default: off)",
    )
    s.add_argument(
        "--native-logit-topk",
        choices=["off", "on"],
        default=None,
        help="Opt-in GPU two-stage logit top-k verification (default: off)",
    )
    s.add_argument(
        "--native-ple-async-gather",
        choices=["off", "on"],
        default=None,
        help="Opt-in asynchronous 16-bucket PLE gather overlap (default: off)",
    )
    s.add_argument(
        "--ple-ram-resident",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Opt-in clean read-only mmap PLE RAM residency (default: off)",
    )
    s.add_argument(
        "--native-ple-ram-resident",
        choices=["off", "on"],
        default=None,
        help="Explicit native PLE RAM residency mode (default: off)",
    )
    s.add_argument(
        "--native-mtp-fast-round",
        "--mtp-fast-round",
        dest="native_mtp_fast_round",
        choices=["off", "on"],
        default=None,
        help="Opt-in GPU-resident MTP fast round and fused priming (default: off)",
    )
    s.add_argument(
        "--native-prefill-router-fast",
        "--prefill-router-fast",
        dest="native_prefill_router_fast",
        choices=["off", "on"],
        default=None,
        help="Opt-in 1-pass BF16 NAX MoE router projection in prefill (default: off)",
    )
    s.add_argument(
        "--native-verify-hc-gdn-fused",
        choices=["off", "on"],
        default=None,
        help="Opt-in fused HC+GDN multi-row verification (default: off)",
    )
    s.add_argument(
        "--native-dflash2-defer-context",
        choices=["off", "on"],
        default=None,
        help="Deferred DFlash2 context mode (default: on for dflash2)",
    )
    s.add_argument(
        "--native-dflash2-verify-rows",
        choices=["all", "auto"],
        default=None,
        help="Opt-in DFlash2 adaptive verify row budget (default: all)",
    )
    s.add_argument(
        "--drafter",
        choices=["mtp", "dflash2"],
        default="mtp",
        help="Speculative drafter mode (default: mtp)",
    )
    s.add_argument(
        "--dflash2-dir",
        type=Path,
        default=None,
        help="Opt-in DFlash2 safetensors checkpoint directory",
    )
    s.add_argument(
        "--concurrent-dispatch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Metal concurrent compute dispatch (default: on)",
    )
    s.add_argument(
        "--q8-gemv-rg4",
        choices=["off", "on", "wide"],
        default="off",
        help="Retired Q8 GEMV experiment (default: off)",
    )
    s.add_argument(
        "--q8-three-msg",
        choices=["off", "on", "wide"],
        default="off",
        help="Retired multi-SG Q8 experiment (default: off)",
    )
    s.add_argument(
        "--prefill-multiseg",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Retired multi-segment prefill (default: off)",
    )
    s.add_argument(
        "--prefill-swizzle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Swizzled QSA prefill projection (default: on)",
    )
    s.add_argument(
        "--qsa-prefill-fused",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fused QSA prefill operator (default: on)",
    )
    s.add_argument(
        "--hc-consumer-fused",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fused HC consumer operator (default: on)",
    )
    s.add_argument(
        "--ple-overlap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Chunk-ahead PLE prefill overlap (default: on)",
    )
    s.add_argument(
        "--shared-prefix-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Host-RAM shared system-prefix cache (default: on)",
    )
    s.add_argument(
        "--parked-lanes-mib",
        type=int,
        default=None,
        help="Host RAM for conversations parked on a switch (0..16384, default: 4096 when shared prefix cache is enabled)",
    )
    s.add_argument(
        "--mtp-lookup",
        choices=["off", "bypass", "hybrid"],
        default=None,
        help="MTP prompt-lookup mode (default: hybrid for mtp)",
    )
    s.add_argument(
        "--mtp-lookup-match",
        type=int,
        choices=range(2, 17),
        metavar="2..16",
        default=None,
        help="Opt-in MTP prompt-lookup match length (2..16)",
    )
    s.add_argument(
        "--dflash2-lookup",
        choices=["off", "bypass", "hybrid"],
        default=None,
        help="DFlash2 prompt-lookup mode (default: hybrid for dflash2)",
    )
    s.add_argument(
        "--dflash2-lookup-match",
        type=int,
        choices=range(2, 17),
        metavar="2..16",
        default=None,
        help="Opt-in DFlash2 prompt-lookup match length (2..16)",
    )
    s.add_argument(
        "--tool-enum-validation",
        choices=["strict", "executor"],
        default=None,
        help="OpenAI tool enum schema policy (default: strict)",
    )
    s.add_argument(
        "--tool-arg-validation",
        choices=["strict", "executor"],
        default=None,
        help="OpenAI tool argument schema policy (default: strict)",
    )
    s.add_argument(
        "--private-prefix-trace",
        action="store_true",
        default=False,
        help="Opt-in private prefix reuse trace telemetry",
    )
    s.add_argument(
        "--telemetry-stdout",
        action="store_true",
        default=False,
        help="Opt-in structured JSONL telemetry on stdout",
    )
    s.add_argument(
        "--checkpoint-dir",
        "--checkpoint-ssd-dir",
        dest="checkpoint_dir",
        type=Path,
        default=None,
        help="Custom SSD checkpoint storage directory (default: .tokspur/state/checkpoints; needs a capacity above 0)",
    )
    s.add_argument(
        "--checkpoint-cap-gib",
        "--checkpoint-ssd-budget-gib",
        dest="checkpoint_cap_gib",
        type=float,
        default=None,
        help="SSD checkpoint capacity budget in GiB (0.0..32.0, default: 0.0, opt in with 8.0)",
    )
    s.add_argument(
        "--checkpoint-floor-gib",
        "--checkpoint-ssd-floor-gib",
        dest="checkpoint_floor_gib",
        type=float,
        default=None,
        help="APFS free space floor in GiB (default: 20.0; needs a capacity above 0)",
    )
    s.add_argument(
        "--checkpoint-policy",
        choices=["recompute-cost", "lru"],
        default=None,
        help="Eviction priority policy (default: recompute-cost; needs a capacity above 0)",
    )
    s.add_argument(
        "--overlap-prefill",
        action="store_true",
        default=False,
        help="Enable 16-token overlap background prefill with boundary protection",
    )
    args = parser.parse_args()
    try:
        if args.command == "pull":
            if args.dir_pos is not None and args.dir_flag is not None:
                raise ValueError("pull accepts a single model directory")
            pull(
                args.dir_pos or args.dir_flag,
                dflash2=args.dflash2,
                dflash2_dir=args.dflash2_dir,
                port=args.port,
            )
        elif args.command == "adopt":
            with state_lock():
                if args.check_only:
                    if (
                        args.dir_pos is not None
                        or args.target is not None
                        or args.draft is not None
                        or args.dflash2_dir is not None
                    ):
                        raise ValueError(
                            "--check-only cannot be combined with directory arguments"
                        )
                    data = read_json(ROOT / ".tokspur/adoption.json")
                    check_adoption(data)
                    print(
                        "Existing weights verified in place. No model files copied or downloaded."
                    )
                elif args.dir_pos is not None:
                    if args.target is not None or args.draft is not None:
                        raise ValueError(
                            "adopt <dir> cannot be combined with --target or --draft"
                        )
                    adopt(args.dir_pos, args.dir_pos, args.dflash2_dir)
                else:
                    if not args.target:
                        raise ValueError(
                            "Pass a model directory (tokspur adopt <dir>) or --target/--draft, or --check-only"
                        )
                    adopt(
                        args.target,
                        args.draft or args.target,
                        args.dflash2_dir,
                    )
        elif args.command == "configure":
            with state_lock():
                configure(args.port)
        elif args.command == "serve":
            if args.model_dir_pos is not None and args.model_dir_flag is not None:
                raise ValueError("serve accepts a single model directory")
            serve(
                allow_raw_gguf=args.raw_gguf,
                pack_v3=args.pack_v3,
                derived_pack=args.derived_pack,
                gdn_prefill_only=args.gdn_prefill_only,
                derived_pack_trust_mode=args.derived_pack_trust_mode,
                output_packet=args.output_packet,
                first_output_packet=args.first_output_packet,
                prefill_dense=args.prefill_dense,
                prefill_grouped=args.prefill_grouped,
                known_token_capacity=args.known_token_capacity,
                context_length=args.context_length,
                mtp_draft_slice=args.mtp_draft_slice,
                verify_nax=args.verify_nax,
                verify_nax_m8=args.verify_nax_m8,
                speed_preset=args.speed_preset,
                mtp_depth=args.mtp_depth,
                dflash2_precision=args.dflash2_precision,
                dflash2_max_drafts=args.dflash2_max_drafts,
                dflash2_vocab_slice=args.dflash2_vocab_slice,
                dflash2_vocab_gather=args.dflash2_vocab_gather,
                native_host_logit_views=args.native_host_logit_views,
                native_logit_topk=args.native_logit_topk,
                native_ple_async_gather=args.native_ple_async_gather,
                ple_ram_resident=args.ple_ram_resident,
                native_ple_ram_resident=args.native_ple_ram_resident,
                native_mtp_fast_round=args.native_mtp_fast_round,
                native_prefill_router_fast=args.native_prefill_router_fast,
                native_verify_hc_gdn_fused=args.native_verify_hc_gdn_fused,
                native_dflash2_defer_context=args.native_dflash2_defer_context,
                native_dflash2_verify_rows=args.native_dflash2_verify_rows,
                native_mtp_draft_slice=args.native_mtp_draft_slice,
                native_verify_nax=args.native_verify_nax,
                native_verify_nax_m8=args.native_verify_nax_m8,
                drafter=args.drafter,
                dflash2_dir=args.dflash2_dir,
                concurrent_dispatch=args.concurrent_dispatch,
                q8_gemv_rg4=args.q8_gemv_rg4,
                q8_three_msg=args.q8_three_msg,
                prefill_multiseg=args.prefill_multiseg,
                prefill_swizzle=args.prefill_swizzle,
                qsa_prefill_fused=args.qsa_prefill_fused,
                hc_consumer_fused=args.hc_consumer_fused,
                ple_overlap=args.ple_overlap,
                shared_prefix_cache=args.shared_prefix_cache,
                parked_lanes_mib=args.parked_lanes_mib,
                mtp_lookup=args.mtp_lookup,
                mtp_lookup_match=args.mtp_lookup_match,
                dflash2_lookup=args.dflash2_lookup,
                dflash2_lookup_match=args.dflash2_lookup_match,
                tool_enum_validation=args.tool_enum_validation,
                tool_arg_validation=args.tool_arg_validation,
                private_prefix_trace=args.private_prefix_trace,
                telemetry_stdout=args.telemetry_stdout,
                checkpoint_dir=args.checkpoint_dir,
                checkpoint_cap_gib=args.checkpoint_cap_gib,
                checkpoint_floor_gib=args.checkpoint_floor_gib,
                checkpoint_policy=args.checkpoint_policy,
                overlap_prefill=args.overlap_prefill,
                model_dir=args.model_dir_pos or args.model_dir_flag,
                port=args.port,
            )
        elif args.command == "verify-build":
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/bootstrap.py"), "--verify-only"],
                check=True,
            )
        elif args.command == "plan":
            print(json.dumps(profile(), indent=2))
        else:
            print(
                json.dumps(
                    {
                        "version": __version__,
                        "cli_implementation": "python",
                        "layout": "checkout",
                        "native_built": (
                            ROOT / "build/owned-native/tokspur-native"
                        ).is_file(),
                        "server_built": (
                            (
                                ROOT / "build/rust-fast-math/release/tokspur-server"
                            ).is_file()
                            or (
                                ROOT / "build/rust-target/release/tokspur-server"
                            ).is_file()
                            or (ROOT / "rust/target/release/tokspur-server").is_file()
                        ),
                        "weights_adopted": (ROOT / ".tokspur/adoption.json").is_file(),
                        "configured": (ROOT / ".tokspur/configured.json").is_file(),
                        "default_port_available": port_available(profile()["port"]),
                        "other_inference_workers": inference_processes(),
                    },
                    indent=2,
                )
            )
    except (
        OSError,
        ValueError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as error:
        parser.exit(1, f"tokspur: {error}\n")
    except KeyError as error:
        parser.exit(
            1, f"tokspur: missing configuration field {error}; re-run adopt/configure\n"
        )
    except ModuleNotFoundError as error:
        parser.exit(1, f"tokspur: {error}; run scripts/bootstrap.py first\n")


if __name__ == "__main__":
    main()
