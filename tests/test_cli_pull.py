# SPDX-License-Identifier: Apache-2.0
"""Model-free checks of the Python reference `pull`; curl is replaced by a stand-in."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO = "fixture-org/Fixture-Pull-GGUF"
REVISION = "0123456789abcdef0123456789abcdef01234567"
DFLASH2_REPO = "fixture-org/Fixture-DFlash2"
FILES = {
    "target.gguf": b"synthetic-pull-target\n",
    "MTP/draft.gguf": b"synthetic-pull-draft\n",
    "config.json": b'{"architectures":["DFlash2"]}\n',
    "model.safetensors": b"synthetic-pull-dflash2\n",
}


def group(*names: str, repo: str = REPO) -> dict:
    return {
        "repo": repo,
        "revision": REVISION,
        "files": [
            {
                "name": name,
                "size": len(FILES[name]),
                "sha256": hashlib.sha256(FILES[name]).hexdigest(),
            }
            for name in names
        ],
    }


class Checkout:
    """A temporary checkout whose curl stand-in serves `FILES` and records each command."""

    def __init__(self, tmp_path: Path, monkeypatch):
        from tokspur import cli, local

        self.cli = cli
        self.local = local
        self.root = tmp_path
        self.commands: list[list[str]] = []
        policy = local.profile()
        monkeypatch.setattr(cli, "ROOT", tmp_path)
        monkeypatch.setattr(local, "ROOT", tmp_path)
        monkeypatch.setattr(cli, "profile", lambda: policy)
        monkeypatch.setattr(cli, "port_available", lambda port: True)
        monkeypatch.setattr(cli.subprocess, "run", self.curl)
        self.pin({"target": group("target.gguf"), "draft": group("MTP/draft.gguf")})

    def pin(self, profile: dict) -> None:
        path = self.root / "profiles/flash-next-assets.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema": 1, **profile}))

    @property
    def managed(self) -> Path:
        return self.root / ".tokspur/models/Fixture-Pull-GGUF"

    def curl(self, cmd, check, env):
        self.commands.append(cmd)
        name = cmd[-1].split(f"/resolve/{REVISION}/", 1)[1]
        output = Path(cmd[cmd.index("--output") + 1])
        # The stand-in never writes outside the temporary checkout.
        assert output.resolve().is_relative_to(self.root), output
        resume = "--continue-at" in cmd
        with output.open("ab" if resume else "wb") as stream:
            stream.write(FILES[name][output.stat().st_size if resume else 0 :])

    def adoption(self) -> dict:
        return json.loads((self.root / ".tokspur/adoption.json").read_text())


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    return Checkout(tmp_path, monkeypatch)


def test_pull_downloads_verifies_and_configures(checkout, capsys):
    checkout.cli.pull()
    assert (checkout.managed / "target.gguf").read_bytes() == FILES["target.gguf"]
    assert (checkout.managed / "MTP/draft.gguf").read_bytes() == FILES["MTP/draft.gguf"]
    assert not list(checkout.managed.rglob("*.part"))
    adoption = checkout.adoption()
    assert adoption["target"]["directory"] == str(checkout.managed.resolve())
    assert (checkout.root / ".tokspur/configured.json").is_file()
    assert "Ready." in capsys.readouterr().out


def test_pull_keeps_the_callers_curlrc_out_of_the_download(checkout):
    checkout.cli.pull()
    assert len(checkout.commands) == 2
    for cmd in checkout.commands:
        # curl honours `-q` only in first position.
        assert cmd[:2] == ["/usr/bin/curl", "-q"]


def test_pull_caps_each_transfer_at_the_pinned_size(checkout, monkeypatch):
    size = len(FILES["target.gguf"])
    # The floor is below these fixtures here, so the pins themselves are the caps.
    monkeypatch.setattr(checkout.cli, "_MIN_TRANSFER_CAP", 1)
    checkout.managed.mkdir(parents=True)
    (checkout.managed / "target.gguf.part").write_bytes(FILES["target.gguf"][:5])
    checkout.cli.pull()
    resumed, fresh = checkout.commands
    # On a resume the cap is what the partial file still lacks.
    assert resumed[resumed.index("--continue-at") + 1] == "-"
    assert resumed[resumed.index("--max-filesize") + 1] == str(size - 5)
    assert "--continue-at" not in fresh
    assert fresh[fresh.index("--max-filesize") + 1] == str(len(FILES["MTP/draft.gguf"]))
    assert (checkout.managed / "target.gguf").read_bytes() == FILES["target.gguf"]


def test_pull_never_caps_a_transfer_below_a_redirect_body(checkout):
    # Before 8.11.0 curl counts the cap against the body of a redirect it
    # follows, so a cap of the few bytes these files need would fail there.
    checkout.managed.mkdir(parents=True)
    (checkout.managed / "target.gguf.part").write_bytes(FILES["target.gguf"][:5])
    checkout.cli.pull()
    assert checkout.cli._MIN_TRANSFER_CAP == 64 << 10
    for cmd in checkout.commands:
        assert cmd[cmd.index("--max-filesize") + 1] == str(64 << 10)
    assert (checkout.managed / "target.gguf").read_bytes() == FILES["target.gguf"]


def test_pull_holds_a_file_to_its_pin_under_the_transfer_floor(checkout, monkeypatch):
    # The floor lets curl accept more than a small pin; the size check does not.
    def oversized(cmd, check, env):
        checkout.commands.append(cmd)
        Path(cmd[cmd.index("--output") + 1]).write_bytes(b"x" * 4096)

    monkeypatch.setattr(checkout.cli.subprocess, "run", oversized)
    with pytest.raises(ValueError, match="Unexpected size"):
        checkout.cli.pull()
    assert not list(checkout.managed.rglob("*"))
    assert not (checkout.root / ".tokspur/adoption.json").exists()


def test_plain_pull_keeps_the_recorded_dflash2_checkpoint(checkout):
    checkout.pin(
        {
            "target": group("target.gguf"),
            "draft": group("MTP/draft.gguf"),
            "dflash2_draft": group(
                "config.json", "model.safetensors", repo=DFLASH2_REPO
            ),
        }
    )
    checkout.cli.pull(dflash2=True)
    recorded = checkout.adoption()["dflash2_draft"]
    assert recorded["directory"] == str(
        (checkout.root / ".tokspur/models/Fixture-DFlash2").resolve()
    )

    checkout.cli.pull()
    assert checkout.adoption()["dflash2_draft"] == recorded

    # A checkpoint that no longer checks out is not carried over.
    weights = Path(recorded["directory"]) / "model.safetensors"
    weights.write_bytes(b"synthetic-pull-forged!\n")
    checkout.cli.pull()
    assert "dflash2_draft" not in checkout.adoption()


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("../escape.gguf", "must not traverse directories"),
        ("MTP/../../escape.gguf", "must not traverse directories"),
        ("MTP//draft.gguf", "must not traverse directories"),
        ("/tmp/escape.gguf", "must be a relative path"),
        ("bad name.gguf", "invalid asset filename component"),
        ("", "must not be empty"),
    ],
)
def test_pull_refuses_unsafe_asset_names(checkout, name, message):
    draft = group("MTP/draft.gguf")
    draft["files"][0]["name"] = name
    checkout.pin({"target": group("target.gguf"), "draft": draft})
    with pytest.raises(ValueError, match=message):
        checkout.cli.pull()
    assert len(checkout.commands) == 1  # the target only
    assert not (checkout.root / ".tokspur/adoption.json").exists()
    assert not list(checkout.root.rglob("escape.gguf*"))


def test_pull_refuses_a_symlinked_download_directory(checkout, tmp_path_factory):
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    checkout.managed.mkdir(parents=True)
    (checkout.managed / "MTP").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(
        ValueError, match="download directory must not traverse symlinks"
    ):
        checkout.cli.pull()
    assert not list(elsewhere.iterdir())
    assert not (checkout.root / ".tokspur/adoption.json").exists()

    # The managed directory itself may not be a symlink either.
    (checkout.managed / "MTP").unlink()
    (checkout.managed / "target.gguf").unlink()
    checkout.managed.rmdir()
    checkout.managed.symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(
        ValueError, match="model destination directory must not traverse symlinks"
    ):
        checkout.cli.pull()
    assert not list(elsewhere.iterdir())


def test_pull_refuses_a_symlinked_explicit_directory(checkout, tmp_path_factory):
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    link = checkout.root / "models-link"
    link.symlink_to(elsewhere, target_is_directory=True)
    for directory in (link, link / "below"):
        with pytest.raises(
            ValueError, match="model destination directory must not traverse symlinks"
        ):
            checkout.cli.pull(directory)
    assert not checkout.commands
    assert not list(elsewhere.iterdir())

    # The same holds for an explicit DFlash2 directory, checked when its group is pulled.
    checkout.pin(
        {
            "target": group("target.gguf"),
            "draft": group("MTP/draft.gguf"),
            "dflash2_draft": group(
                "config.json", "model.safetensors", repo=DFLASH2_REPO
            ),
        }
    )
    with pytest.raises(
        ValueError, match="model destination directory must not traverse symlinks"
    ):
        checkout.cli.pull(dflash2_dir=link)
    assert len(checkout.commands) == 2  # the target and the draft only
    assert not list(elsewhere.iterdir())
    assert not (checkout.root / ".tokspur/adoption.json").exists()

    # A plain explicit directory is still used as given.
    plain = checkout.root / "models-plain"
    checkout.cli.pull(plain, dflash2_dir=checkout.root / "dflash2-plain")
    assert (plain / "target.gguf").read_bytes() == FILES["target.gguf"]
    assert checkout.adoption()["dflash2_draft"]["directory"] == str(
        checkout.root / "dflash2-plain"
    )


@pytest.mark.parametrize(
    "kind", ["symlink", "dangling symlink", "hard link", "directory"]
)
def test_pull_refuses_a_partial_download_that_is_not_a_plain_file(
    checkout, tmp_path_factory, kind
):
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    victim = elsewhere / "victim"
    victim.write_bytes(FILES["target.gguf"][:5])
    checkout.managed.mkdir(parents=True)
    part = checkout.managed / "target.gguf.part"
    message = "must be a regular file"
    if kind == "symlink":
        part.symlink_to(victim)
    elif kind == "dangling symlink":
        part.symlink_to(elsewhere / "absent")
    elif kind == "hard link":
        part.hardlink_to(victim)
        message = "must not have multiple hard links"
    else:
        part.mkdir()
    with pytest.raises(ValueError, match=message):
        checkout.cli.pull()
    # curl was never pointed at it, and nothing beyond the checkout changed.
    assert not checkout.commands
    assert victim.read_bytes() == FILES["target.gguf"][:5]
    assert sorted(path.name for path in elsewhere.iterdir()) == ["victim"]
    assert not (checkout.root / ".tokspur/adoption.json").exists()


@pytest.mark.parametrize(
    "repo",
    [
        "owner",
        "owner/leaf/extra",
        "owner/..",
        "../leaf",
        "owner/le af",
        "/leaf",
        "owner/",
    ],
)
def test_repository_identifiers_are_two_plain_components(checkout, repo):
    with pytest.raises(ValueError, match="invalid repository identifier"):
        checkout.cli._repo_leaf_name(repo)
    assert checkout.cli._repo_leaf_name("owner/Leaf-1.0_x") == "Leaf-1.0_x"
