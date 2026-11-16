# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path

import pytest

from tokspur import local

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(local, "ROOT", tmp_path)
    return tmp_path


def test_private_state_permissions_and_no_symlinks(sandbox):
    state = local.private_dir(sandbox / ".tokspur/state")
    assert state.stat().st_mode & 0o777 == 0o700
    (state / "link").symlink_to(sandbox, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        local.private_dir(state / "link/elsewhere")
    with pytest.raises(ValueError):
        local.private_dir(sandbox / "outside")
    with pytest.raises(ValueError):
        local.private_dir(sandbox / ".tokspur/state/../../outside")


def test_atomic_private_json_refuses_link(sandbox):
    target = sandbox / ".tokspur/state/test.json"
    local.write_json(target, {"key": "private"})
    assert json.loads(target.read_text()) == {"key": "private"}
    assert target.stat().st_mode & 0o777 == 0o600
    link = target.parent / "link.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        local.write_json(link, {})
    assert json.loads(target.read_text()) == {"key": "private"}


def test_hash_verified_in_place(tmp_path):
    asset = tmp_path / "weights"
    asset.write_bytes(b"test weight fixture")
    spec = {
        "size": asset.stat().st_size,
        "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
    }
    result = local.verify_asset(asset, spec)
    assert result["sha256"] == spec["sha256"]
    assert result["inode"] == asset.stat().st_ino
    with pytest.raises(ValueError, match="Checksum"):
        local.verify_asset(asset, {**spec, "sha256": "0" * 64})
    with pytest.raises(ValueError, match="size"):
        local.verify_asset(asset, {**spec, "size": 0})


@pytest.mark.skipif(not local.IS_DARWIN, reason="requires the Apple volume API")
def test_volume_identity_matches_distinct_files_on_the_same_volume(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    a, b = local.identity(first), local.identity(second)
    assert a["inode"] != b["inode"]
    assert len(a["volume_uuid"]) == 32
    assert a["volume_uuid"] == b["volume_uuid"]


def test_adoption_remount_keeps_volume_and_file_checks(tmp_path, monkeypatch):
    monkeypatch.setattr(local, "_volume_uuid", lambda fd: "ab" * 16)
    data = {"schema": 1, "verification": "full-sha256"}
    profile = {}
    for group in ("target", "draft"):
        directory = tmp_path / group
        directory.mkdir()
        path = directory / "fixture"
        path.write_bytes(group.encode())
        spec = {
            "name": "fixture",
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        profile[group] = {"files": [spec]}
        data[group] = {
            "directory": str(directory),
            "assets": [local.verify_asset(path, spec)],
        }
    monkeypatch.setattr(local, "read_json", lambda path: profile)
    local.check_adoption(data)
    original = json.loads(json.dumps(data))
    asset = data["target"]["assets"][0]
    asset["device"] += 1
    local.check_adoption(data)
    for key in ("inode", "size", "mtime_ns", "ctime_ns"):
        asset[key] += 1
        with pytest.raises(ValueError, match="Asset changed"):
            local.check_adoption(data)
        asset[key] -= 1
    for uuid in (None, 0, "bad", "z" * 32, "0" * 32):
        asset["volume_uuid"] = uuid
        with pytest.raises(ValueError, match="volume UUID|Asset changed"):
            local.check_adoption(data)
    asset["volume_uuid"] = "ab" * 16
    for device in (-1, True, 0x80000000):
        asset["device"] = device
        with pytest.raises(ValueError, match="invalid recorded asset device"):
            local.check_adoption(data)
    legacy = original["target"]["assets"][0]
    del legacy["volume_uuid"]
    local.check_adoption(original)
    legacy["device"] += 1
    with pytest.raises(ValueError, match="Asset changed"):
        local.check_adoption(original)


def test_port_guard():
    with socket.socket() as occupied:
        occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        assert not local.port_available(occupied.getsockname()[1])


def test_port_guard_sees_wildcard_listener():
    # macOS lets an exact 127.0.0.1 bind succeed beside a 0.0.0.0 listener, so
    # the bind probe alone would report this port as free.
    with socket.socket() as occupied:
        occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        occupied.bind(("0.0.0.0", 0))
        occupied.listen()
        assert not local.port_available(occupied.getsockname()[1])


def test_port_guard_allows_reuse_after_connection_closes():
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
            connection, _ = listener.accept()
            connection.close()  # Server-side TIME_WAIT must not look occupied.
            assert client.recv(1) == b""
    assert local.port_available(port)


def test_inference_processes_include_owned_native_worker(monkeypatch):
    import psutil

    class Fake:
        def __init__(self, pid, name, cmdline):
            self.pid = pid
            self.info = {"name": name, "cmdline": cmdline}

    fakes = [
        Fake(10, None, ["/x/build/owned-native/tokspur-native", "target", "--policy"]),
        Fake(11, "omlx-server", None),  # cmdline unreadable; the name still counts
        Fake(12, "bash", ["bash", "-c", "echo tokspur-native"]),
        Fake(13, "Python", ["python", "-m", "pytest"]),
        Fake(14, "branch8-target", None),
        Fake(15, None, ["/x/private/branch8-target", "target"]),
        Fake(16, "stage8-target", None),
        Fake(17, None, ["/x/private/stage8-target", "target"]),
    ]
    monkeypatch.setattr(psutil, "process_iter", lambda attrs=None: iter(fakes))
    assert {entry["pid"] for entry in local.inference_processes()} == {
        10,
        11,
        14,
        15,
        16,
        17,
    }


def test_inference_processes_match_the_executable_not_its_operands(monkeypatch):
    from types import SimpleNamespace

    import psutil

    cases = [
        # A worker name as an operand of another program is not a worker.
        (220, "tail", ["tail", "-f", "logs/llama-server"], False),
        (221, "vim", ["/usr/bin/vim", "splash"], False),
        (222, "less", ["less", "/x/build/owned-native/tokspur-native"], False),
        (223, "cat", ["cat", "mlx_lm.server"], False),
        (224, "grep", ["grep", "-r", "ollama", "runner"], False),
        # An interpreter runs the script or module named by a later word.
        (225, "python3.12", ["/venv/bin/python3.12", "/venv/bin/omlx", "serve"], True),
        (
            226,
            "Python",
            ["/Frameworks/Python", "/venv/bin/mlx_lm.server", "--model"],
            True,
        ),
        (227, "bash", ["bash", "/opt/bin/llama-server", "--port"], True),
        (228, "python3", ["python3", "train.py", "--epochs"], False),
        (229, "llama-server", ["/opt/bin/llama-server", "--port"], True),
        (230, "python3", ["python3", "-m", "mlx_lm.server"], True),
        (231, "ollama", ["ollama", "runner"], True),
        (232, "ollama", ["ollama", "list"], False),
    ]
    fakes = [
        SimpleNamespace(pid=pid, info={"name": name, "cmdline": argv})
        for pid, name, argv, _expected in cases
    ]
    monkeypatch.setattr(psutil, "process_iter", lambda attrs=None: iter(fakes))
    assert {row["pid"] for row in local.inference_processes()} == {
        pid for pid, _name, _argv, expected in cases if expected
    }


def test_scalar_identity_full_argv_unreadable_name_and_exact_darwin_alias(monkeypatch):
    from types import SimpleNamespace

    import psutil

    full = local.SCALAR_WORKER_NAME
    short = full[: local.DARWIN_MAXCOMLEN]
    cases = [
        (201, None, ["/private/candidate/" + full, "target"]),
        (202, full, None),
        (203, short, None),  # Another user's argv is unreadable.
        (204, short, []),
        (205, short, ["/usr/bin/unrelated"]),
        (206, short + "x", None),
        (207, short[:-1], None),
        (208, "unrelated", ["bash", "-c", "echo " + full]),
        (209, "unrelated", None),
        (210, full, ["/private/candidate/" + full]),  # self
    ]
    fakes = [
        SimpleNamespace(pid=pid, info={"name": name, "cmdline": argv})
        for pid, name, argv in cases
    ]
    monkeypatch.setattr(local, "IS_DARWIN", True)
    monkeypatch.setattr(local.os, "getpid", lambda: 210)
    monkeypatch.setattr(psutil, "process_iter", lambda attrs=None: iter(fakes))
    assert {row["pid"] for row in local.inference_processes()} == {201, 202, 203, 204}
    monkeypatch.setattr(local, "IS_DARWIN", False)
    assert {row["pid"] for row in local.inference_processes()} == {201, 202}


def test_scalar_identity_retains_permission_and_disappearance_handling(monkeypatch):
    import psutil

    class Unreadable:
        pid = 301

        @property
        def info(self):
            raise psutil.AccessDenied(self.pid)

    class Gone(Unreadable):
        @property
        def info(self):
            raise psutil.NoSuchProcess(self.pid)

    monkeypatch.setattr(
        psutil, "process_iter", lambda attrs=None: iter([Unreadable(), Gone()])
    )
    assert local.inference_processes() == []


def test_child_environment_is_an_allowlist(sandbox, monkeypatch):
    for key in (
        "HF_TOKEN",
        "HF_API_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "PYTHONPATH",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "AWS_SECRET_ACCESS_KEY",
        "GITHUB_TOKEN",
        "HTTPS_PROXY",
        "SSH_AUTH_SOCK",
        "TMPDIR",
        "LANGUAGE",
    ):
        monkeypatch.setenv(key, "do-not-inherit")
    inherited = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/Users/fixture",
        "USER": "fixture",
        "LOGNAME": "fixture",
        "SHELL": "/bin/zsh",
        "LANG": "en_GB.UTF-8",
        "LC_CTYPE": "en_GB.UTF-8",
        "TZ": "Europe/London",
        "TERM": "xterm-256color",
        "TOKSPUR_STATE_ROOT": "/fixture/state",
    }
    for key, value in inherited.items():
        monkeypatch.setenv(key, value)
    env = local.isolated_environment()
    assert "do-not-inherit" not in env.values()
    state = sandbox / ".tokspur/state"
    assert env == {
        **inherited,
        "HF_HOME": str(state / "hf-home"),
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        "TMPDIR": str(state / "tmp"),
        "NO_PROXY": "localhost,127.0.0.1,::1",
    }


def test_state_lock_is_exclusive_and_released(sandbox):
    with local.state_lock():
        with pytest.raises(ValueError, match="owns this state"):
            with local.state_lock():
                pytest.fail("Two operations acquired the same state")
    with local.state_lock():
        pass


def test_state_lock_refuses_symlink(sandbox):
    state = local.private_dir(sandbox / ".tokspur/state")
    outside = sandbox / "outside"
    outside.write_text("unchanged")
    (state / "serve.lock").symlink_to(outside)
    with pytest.raises(OSError):
        with local.state_lock():
            pass
    assert outside.read_text() == "unchanged"


def test_profile_has_explicit_conservative_contract():
    p = local.profile()
    assert (p["context_tokens"], p["kv_format"], p["state_format"]) == (
        65536,
        "bf16",
        "fp32",
    )
    assert p["verify_rows"] == "all" and not p["persistent_cache"]
    assert p["resident_experts"] and p["queue_size"] == 1
    assert p["native_memory_gib"] < p["api_memory_ceiling_gib"] < 128


def test_adoption_variants_and_derived_pack_sealing(sandbox, tmp_path, monkeypatch):
    from tokspur import cli

    monkeypatch.setattr(cli, "ROOT", sandbox)

    def make_spec(path: Path, role: str | None = None) -> dict:
        entry = {
            "name": path.name,
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        if role is not None:
            entry["role"] = role
        return entry

    primary_dir = tmp_path / "primary"
    alt4_dir = tmp_path / "alt4"
    pack_dir = tmp_path / "pack"
    draft_dir = tmp_path / "draft"
    for d in (primary_dir, alt4_dir, pack_dir, draft_dir):
        d.mkdir()

    primary_specs = []
    alt4_specs = []
    for i in range(4):
        p = primary_dir / f"primary-{i}.gguf"
        p.write_bytes(f"primary-{i}".encode())
        primary_specs.append(make_spec(p))
        a = alt4_dir / f"own-{i}.gguf"
        a.write_bytes(f"own-{i}".encode())
        alt4_specs.append(make_spec(a))

    base_gguf = pack_dir / "Qwen3.8-Flash-Next-base.gguf"
    base_gguf.write_bytes(b"cpu-base-gguf")
    pack_gguf = pack_dir / "Qwen3.8-Flash-Next-Q4_K.tokspur.gguf"
    pack_gguf.write_bytes(b"prebuilt-pack-v3")
    pack_gguf.chmod(0o644)
    pack_specs = [make_spec(base_gguf), make_spec(pack_gguf, role="derived_pack")]

    draft_file = draft_dir / "mtp.gguf"
    draft_file.write_bytes(b"draft-mtp")
    draft_spec = make_spec(draft_file)

    profile = {
        "schema": 1,
        "target": {
            "files": primary_specs,
            "variants": [
                {"files": alt4_specs},
                {"files": pack_specs},
            ],
        },
        "draft": {"files": [draft_spec]},
    }
    profiles_dir = sandbox / "profiles"
    profiles_dir.mkdir()
    (profiles_dir / "flash-next-assets.json").write_text(json.dumps(profile))

    # Hard-linked derived pack is refused before sealing or adoption.
    extra_link = tmp_path / "extra-pack.link"
    extra_link.hardlink_to(pack_gguf)
    with pytest.raises(ValueError, match="hard links"):
        cli.adopt(pack_dir, draft_dir)
    extra_link.unlink()

    # Writable derived pack is verified and sealed 0444 during adoption.
    cli.adopt(pack_dir, draft_dir)
    receipt = local.read_json(sandbox / ".tokspur/adoption.json")
    assert pack_gguf.stat().st_mode & 0o777 == 0o444
    assert receipt["target"]["assets"][1]["role"] == "derived_pack"
    local.check_adoption(receipt)

    # Unsealing after adoption is caught by check_adoption.
    pack_gguf.chmod(0o644)
    with pytest.raises(ValueError, match="Asset changed"):
        local.check_adoption(receipt)
    pack_gguf.chmod(0o444)

    # Alternative 4-file variant adopts cleanly without colliding with primary 4-file variant.
    cli.adopt(alt4_dir, draft_dir)
    alt_receipt = local.read_json(sandbox / ".tokspur/adoption.json")
    local.check_adoption(alt_receipt)
    tampered = json.loads(json.dumps(alt_receipt))
    tampered["target"]["assets"][0]["sha256"] = primary_specs[0]["sha256"]
    with pytest.raises(
        ValueError, match="Asset manifest differs from the pinned profile"
    ):
        local.check_adoption(tampered)

    # Ambiguous directory matching multiple variants is refused.
    both_dir = tmp_path / "both"
    both_dir.mkdir()
    for src in list(primary_dir.iterdir()) + list(pack_dir.iterdir()):
        dst = both_dir / src.name
        dst.write_bytes(src.read_bytes())
    with pytest.raises(ValueError, match="Ambiguous target asset variants"):
        cli.adopt(both_dir, draft_dir)

    # Single-path adoption + optional dflash2_draft verification.
    unified_dir = tmp_path / "unified"
    dflash2_dir = tmp_path / "dflash2"
    (unified_dir / "MTP").mkdir(parents=True)
    dflash2_dir.mkdir()
    for name, src in (
        ("Qwen3.8-Flash-Next-base.gguf", base_gguf),
        ("Qwen3.8-Flash-Next-Q4_K.tokspur.gguf", pack_gguf),
    ):
        dst = unified_dir / name
        dst.write_bytes(src.read_bytes())
    uni_mtp = unified_dir / "MTP/mtp.gguf"
    uni_mtp.write_bytes(draft_file.read_bytes())
    df_cfg = dflash2_dir / "config.json"
    df_cfg.write_bytes(b'{"architectures":["DFlash2"]}')
    profile["draft"]["files"] = [
        {
            "name": "MTP/mtp.gguf",
            "size": uni_mtp.stat().st_size,
            "sha256": hashlib.sha256(uni_mtp.read_bytes()).hexdigest(),
        }
    ]
    profile["dflash2_draft"] = {"files": [make_spec(df_cfg)]}
    (profiles_dir / "flash-next-assets.json").write_text(json.dumps(profile))
    cli.adopt(unified_dir, dflash2_dir=dflash2_dir)
    uni_receipt = local.read_json(sandbox / ".tokspur/adoption.json")
    assert "dflash2_draft" in uni_receipt
    local.check_adoption(uni_receipt)


def test_shipped_profile_has_every_file_set_check_adoption_reads():
    expected = json.loads((ROOT / "profiles/flash-next-assets.json").read_text())
    sizes = [
        len(files)
        for files in local.candidate_file_specs(
            expected, "draft", include_dflash2_draft=True
        )
    ]
    # The MTP sidecar, then the DFlash2 checkpoint a receipt may name instead.
    assert sizes == [1, 2]
    target = local.candidate_file_specs(expected, "target")
    assert [len(files) for files in target] == [2]
