# SPDX-License-Identifier: Apache-2.0
"""Source records must detect omissions and changed local or upstream bytes."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import re
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def checker():
    spec = importlib.util.spec_from_file_location(
        "source_check", ROOT / "scripts/check_sources.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def reviewed_tree(tmp_path, checker):
    manifest = json.loads((ROOT / checker.MANIFEST).read_text())
    paths = [checker.MANIFEST]
    paths.extend(row["path"] for row in manifest["native_files"])
    paths.extend(row["path"] for row in manifest.get("rust_files", []))
    paths.extend(row["path"] for row in manifest.get("drafter_files", []))
    paths.extend(row["licence_file"] for row in manifest["projects"])
    paths.extend([manifest["model_licence"]["file"], "profiles/flash-next-assets.json"])
    paths.extend(
        [manifest["rust_standard_library"]["file"], "rust/rust-toolchain.toml"]
    )
    for name in paths:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    for entry in manifest["native_files"]:
        entry["status"] = "recorded-adaptation"
    manifest["native_files"][0]["status"] = "unresolved"
    manifest["unresolved"] = [
        {
            "id": "fixture-source-gap",
            "files": [manifest["native_files"][0]["path"]],
            "reason": "Synthetic missing import record",
            "resolution": "Review the synthetic source",
        }
    ]
    manifest["model_licence"]["review_status"] = "owner-review-required"
    save(tmp_path, checker, manifest)
    return tmp_path, manifest


def save(root, checker, manifest):
    (root / checker.MANIFEST).write_text(json.dumps(manifest))


def test_ordinary_source_checks_stay_offline(checker):
    def unexpected_network(*_args, **_kwargs):
        pytest.fail("ordinary source checks must stay offline")

    result = checker.validate(ROOT, opener=unexpected_network)
    assert result["upstream_files_fetched"] == 0


def test_publication_refuses_source_gaps_and_pending_model_review(
    checker, reviewed_tree, monkeypatch, capsys
):
    root, manifest = reviewed_tree
    monkeypatch.setattr(checker, "ROOT", root)
    monkeypatch.setattr("sys.argv", ["check_sources.py", "--publication"])
    assert checker.main() == 2
    assert not json.loads(capsys.readouterr().out)["publication_ready"]
    manifest["unresolved"] = []
    manifest["native_files"][0]["status"] = "recorded-adaptation"
    save(root, checker, manifest)
    assert checker.main() == 2
    capsys.readouterr()
    manifest["model_licence"]["review_status"] = "confirmed"
    save(root, checker, manifest)
    assert checker.main() == 0
    assert json.loads(capsys.readouterr().out)["publication_ready"]


def test_a_changed_native_file_requires_another_review(checker, reviewed_tree):
    root, manifest = reviewed_tree
    path = root / manifest["native_files"][0]["path"]
    path.write_bytes(path.read_bytes() + b"\n// Changed implementation.\n")
    with pytest.raises(ValueError, match="review has drifted"):
        checker.validate(root)


def test_a_changed_rust_file_requires_another_review(checker, reviewed_tree):
    root, manifest = reviewed_tree
    path = root / manifest["rust_files"][0]["path"]
    path.write_bytes(path.read_bytes() + b"\n// Changed implementation.\n")
    with pytest.raises(ValueError, match="Rust source review has drifted"):
        checker.validate(root)


def test_omitted_credited_rust_source_is_refused(checker, reviewed_tree):
    root, manifest = reviewed_tree
    manifest["rust_files"] = manifest["rust_files"][1:]
    save(root, checker, manifest)
    with pytest.raises(ValueError, match="credited Rust source is unrecorded"):
        checker.validate(root)


def test_rust_records_must_stay_under_the_rust_tree(checker, reviewed_tree):
    root, manifest = reviewed_tree
    manifest["rust_files"][0]["path"] = manifest["native_files"][1]["path"]
    save(root, checker, manifest)
    with pytest.raises(ValueError, match="Rust source path required"):
        checker.validate(root)


def test_a_changed_drafter_file_requires_another_review(checker, reviewed_tree):
    root, manifest = reviewed_tree
    path = root / manifest["drafter_files"][0]["path"]
    path.write_bytes(path.read_bytes() + b"\n# Changed implementation.\n")
    with pytest.raises(ValueError, match="drafter source review has drifted"):
        checker.validate(root)


def test_omitted_credited_drafter_source_is_refused(checker, reviewed_tree):
    root, manifest = reviewed_tree
    manifest["drafter_files"] = manifest["drafter_files"][1:]
    save(root, checker, manifest)
    with pytest.raises(ValueError, match="credited drafter source is unrecorded"):
        checker.validate(root)


def test_drafter_records_must_stay_under_the_drafter_tree(checker, reviewed_tree):
    root, manifest = reviewed_tree
    manifest["drafter_files"][0]["path"] = manifest["native_files"][1]["path"]
    save(root, checker, manifest)
    with pytest.raises(ValueError, match="drafter source path required"):
        checker.validate(root)


def test_runtime_records_cannot_cite_a_source_only_project(checker, reviewed_tree):
    root, manifest = reviewed_tree
    manifest["native_files"][1]["sources"].append("DFlash")
    save(root, checker, manifest)
    with pytest.raises(ValueError, match="runtime source cites a source-only project"):
        checker.validate(root)


def test_omitted_credited_kernel_is_refused(checker, reviewed_tree):
    root, manifest = reviewed_tree
    manifest["native_files"] = [
        row
        for row in manifest["native_files"]
        if row["path"] != "native/kernels/raw_decode.metal"
    ]
    save(root, checker, manifest)
    with pytest.raises(ValueError, match="credited native source is unrecorded"):
        checker.validate(root)


def test_a_removed_gap_does_not_resolve_its_source(checker, reviewed_tree):
    root, manifest = reviewed_tree
    manifest["unresolved"].pop(0)
    save(root, checker, manifest)
    with pytest.raises(ValueError, match="files and gaps disagree"):
        checker.validate(root)


def test_runtime_only_review_does_not_grant_model_rights(checker, reviewed_tree):
    root, manifest = reviewed_tree
    manifest["unresolved"] = []
    manifest["native_files"][0]["status"] = "recorded-adaptation"
    manifest["model_licence"]["review_status"] = "runtime-only"
    save(root, checker, manifest)
    assert checker.validate(root)["publication_ready"]
    manifest["model_licence"]["distribution_scope"][
        "commercial_model_authorisation"
    ] = True
    save(root, checker, manifest)
    with pytest.raises(ValueError, match="must grant no model rights"):
        checker.validate(root)


def test_retired_shader_file_cannot_return(checker, reviewed_tree):
    root, manifest = reviewed_tree
    name = manifest["retired_sources"][0]["removed_files"][0]
    (root / name).write_text("// An old shader returned.\n")
    with pytest.raises(ValueError, match="removed source has returned"):
        checker.validate(root)


def test_retired_shader_cannot_move_to_another_file(checker, reviewed_tree):
    root, manifest = reviewed_tree
    symbol = manifest["retired_shader_symbols"][0]
    (root / "native/kernels/returned.metal").write_text(
        f"kernel void {symbol}() {{}}\n"
    )
    with pytest.raises(ValueError, match="removed shader implementation has returned"):
        checker.validate(root)


def test_bundled_licence_changes_are_refused(checker, reviewed_tree):
    root, manifest = reviewed_tree
    (root / manifest["projects"][0]["licence_file"]).write_text("Different notice\n")
    with pytest.raises(ValueError, match="bundled licence differs"):
        checker.validate(root)


@pytest.mark.parametrize("revision", ["6050b467", "main", "v0.7.1"])
def test_source_pins_must_be_full_commits(checker, reviewed_tree, revision):
    root, manifest = reviewed_tree
    manifest["projects"][0]["revision"] = revision
    save(root, checker, manifest)
    with pytest.raises(ValueError, match="full source commit"):
        checker.validate(root)


@pytest.mark.parametrize("path", ["../LICENSE", "/etc/passwd", "x/../LICENSE"])
def test_manifest_paths_cannot_leave_the_source_tree(checker, reviewed_tree, path):
    root, manifest = reviewed_tree
    manifest["projects"][0]["licence_file"] = path
    save(root, checker, manifest)
    with pytest.raises(ValueError, match="relative and canonical"):
        checker.validate(root)


def test_an_explicit_fetch_checks_bytes_without_executing_them(checker, reviewed_tree):
    root, manifest = reviewed_tree
    data = b"Pinned public source fixture\n"
    count = 0
    for project in manifest["projects"]:
        for source in project["files"]:
            source.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            count += 1
    manifest["model_licence"]["sha256"] = hashlib.sha256(data).hexdigest()
    (root / manifest["model_licence"]["file"]).write_bytes(data)
    save(root, checker, manifest)
    urls = []

    def source(url, *, timeout):
        assert timeout == 30
        assert url.startswith(
            ("https://raw.githubusercontent.com/", "https://huggingface.co/")
        )
        urls.append(url)
        return io.BytesIO(data)

    assert (
        checker.validate(root, fetch=True, opener=source)["upstream_files_fetched"]
        == count
    )
    assert len(urls) == count + 1
    with pytest.raises(ValueError, match="upstream bytes differ"):
        checker.validate(
            root, fetch=True, opener=lambda *_args, **_kwargs: io.BytesIO(b"changed")
        )


def test_changed_model_assets_require_another_licence_review(checker, reviewed_tree):
    root, _manifest = reviewed_tree
    (root / "profiles/flash-next-assets.json").write_text(
        '{"schema":1,"changed":true}\n'
    )
    with pytest.raises(ValueError, match="without a licence review"):
        checker.validate(root)


def test_changed_rust_toolchain_requires_updated_library_notices(
    checker, reviewed_tree
):
    root, _manifest = reviewed_tree
    (root / "rust/rust-toolchain.toml").write_text('[toolchain]\nchannel = "0.0.0"\n')
    with pytest.raises(ValueError, match="without reviewing library notices"):
        checker.validate(root)


def test_changed_rust_library_notices_are_refused(checker, reviewed_tree):
    root, manifest = reviewed_tree
    (root / manifest["rust_standard_library"]["file"]).write_text("Changed notices\n")
    with pytest.raises(ValueError, match="Rust library notice differs"):
        checker.validate(root)


def test_notice_the_manifest_and_the_release_licences_agree(checker):
    notice = (ROOT / "NOTICE").read_text()
    manifest = json.loads((ROOT / checker.MANIFEST).read_text())
    spec = importlib.util.spec_from_file_location(
        "release_licences", ROOT / "scripts/release.py"
    )
    release = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(release)
    shipped = set(release.LICENCES.values())
    adapted = {
        name
        for key in ("native_files", "rust_files", "drafter_files")
        for row in manifest.get(key, [])
        for name in row["sources"]
    }
    source_only = set()
    for project in manifest["projects"]:
        name = project["id"]
        distribution = project.get("distribution", "runtime")
        assert project["revision"] in notice, name
        assert Path(project["licence_file"]).parent.as_posix() in notice, name
        assert (project["licence_file"] in shipped) == (distribution == "runtime"), name
        if distribution == "source-only":
            source_only.add(name)
        # A pin is a recorded source exactly when a recorded file adapts it.
        assert (project["pin_kind"] == "recorded-source") == (name in adapted), name
    assert source_only == {"DFlash"}
    assert manifest["model_licence"]["file"] in shipped
    assert manifest["rust_standard_library"]["file"] in shipped

    recorded = {project["revision"] for project in manifest["projects"]}
    unrecorded = [
        line
        for line in notice.splitlines()
        if set(re.findall(r"\b[0-9a-f]{40}\b", line)) - recorded
    ]
    assert not unrecorded
    assert "except DFlash" not in notice
    assert (ROOT / "third_party/dflash/LICENSE").is_file()
    assert not any("dflash" in source for source in shipped)

    m5_holder = "Copyright (c) 2026 m5-max-kernels contributors"
    assert m5_holder in notice
    assert m5_holder in (ROOT / "native/third_party/m5-max-kernels/LICENSE").read_text()
    assert m5_holder in (ROOT / "native/kernels/dense_q8_staged_f32.metal").read_text()
