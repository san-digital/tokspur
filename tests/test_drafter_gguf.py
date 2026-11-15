# SPDX-License-Identifier: Apache-2.0
"""The minimal GGUF reader must find tensors behind skipped metadata and decode Q8_0 rows exactly."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    from drafter import gguf
finally:
    sys.path.remove(str(ROOT))


def text(value: str) -> bytes:
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded


def write_gguf(
    path: Path, q8: np.ndarray, scales: np.ndarray, plain: np.ndarray
) -> None:
    rows, columns = q8.shape
    blocks = np.zeros((rows, columns // 32, 34), dtype=np.uint8)
    blocks[:, :, :2] = (
        scales.astype("<f2").view(np.uint8).reshape(rows, columns // 32, 2)
    )
    blocks[:, :, 2:] = q8.reshape(rows, columns // 32, 32).view(np.uint8)
    first, second = blocks.tobytes(), plain.astype("<f4").tobytes()
    second_offset = -(-len(first) // 32) * 32
    body = b"GGUF" + struct.pack("<IQQ", 3, 2, 3)
    body += text("general.alignment") + struct.pack("<II", 4, 32)
    # A string array, like the vocabulary, has to be skipped element by element.
    body += (
        text("tokenizer.tokens")
        + struct.pack("<IIQ", 9, 8, 3)
        + text("a")
        + text("bc")
        + text("")
    )
    body += text("general.name") + struct.pack("<I", 8) + text("tiny")
    body += text("rows.weight") + struct.pack("<IQQIQ", 2, columns, rows, 8, 0)
    body += text("plain.weight") + struct.pack(
        "<IQQIQ", 2, plain.shape[1], plain.shape[0], 0, second_offset
    )
    body += b"\0" * (-len(body) % 32)
    path.write_bytes(body + first + b"\0" * (second_offset - len(first)) + second)


def test_reader_finds_tensors_and_decodes_q8_rows(tmp_path):
    rng = np.random.default_rng(0)
    q8 = rng.integers(-127, 128, (3, 64), dtype=np.int8)
    scales = rng.uniform(0.001, 0.02, (3, 2)).astype(np.float16)
    plain = rng.standard_normal((2, 4)).astype(np.float32)
    path = tmp_path / "tiny.gguf"
    write_gguf(path, q8, scales, plain)

    metadata, tensors = gguf.header(path)
    assert metadata["general.name"] == "tiny" and "tokenizer.tokens" not in metadata
    assert (
        tensors["rows.weight"].shape == (3, 64)
        and tensors["rows.weight"].type == "Q8_0"
    )
    wanted = q8.astype(np.float32) * np.repeat(scales.astype(np.float32), 32, axis=1)
    assert np.array_equal(gguf.rows(path, tensors["rows.weight"]), wanted)
    assert np.array_equal(
        gguf.rows(path, tensors["rows.weight"], np.array([2, 0])), wanted[[2, 0]]
    )
    assert np.array_equal(gguf.rows(path, tensors["plain.weight"]), plain)


def test_reader_refuses_other_files(tmp_path):
    path = tmp_path / "not.gguf"
    path.write_bytes(b"GGML" + b"\0" * 32)
    with pytest.raises(ValueError, match="not a GGUF"):
        gguf.header(path)
