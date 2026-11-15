# SPDX-License-Identifier: Apache-2.0
"""Minimal GGUF reader for the two target tensors the drafter borrows: the token embedding and the output matrix.

Only the header and the requested rows are read. Supported element types are the ones those tensors use in the adopted assets: F32, F16, BF16 and Q8_0.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = b"GGUF"
SCALARS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}
STRING, ARRAY = 8, 9
TYPES = {0: "F32", 1: "F16", 8: "Q8_0", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 30: "BF16"}
Q8_BLOCK = 34


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple[int, ...]
    type: str
    offset: int


class Reader:
    def __init__(self, handle):
        self.handle = handle

    def scalar(self, code: str):
        size = struct.calcsize(code)
        return struct.unpack(code, self.handle.read(size))[0]

    def string(self) -> str:
        return self.handle.read(self.scalar("<Q")).decode("utf-8", errors="replace")

    def value(self, kind: int, keep: bool):
        if kind in SCALARS:
            return self.scalar(SCALARS[kind])
        if kind == STRING:
            return self.string()
        if kind != ARRAY:
            raise ValueError(f"unknown GGUF value type {kind}")
        element, count = self.scalar("<I"), self.scalar("<Q")
        if element in SCALARS:
            size = struct.calcsize(SCALARS[element])
            if keep and count <= 64:
                return [self.scalar(SCALARS[element]) for _ in range(count)]
            self.handle.seek(size * count, 1)
            return None
        # String arrays such as the vocabulary are skipped element by element.
        for _ in range(count):
            self.value(element, False)
        return None


def header(path: Path) -> tuple[dict, dict[str, Tensor]]:
    """Metadata (short values only) and tensor directory of a GGUF file, with absolute data offsets."""
    with path.open("rb") as handle:
        if handle.read(4) != MAGIC:
            raise ValueError(f"{path} is not a GGUF file")
        reader = Reader(handle)
        version, tensor_count, pair_count = (
            reader.scalar("<I"),
            reader.scalar("<Q"),
            reader.scalar("<Q"),
        )
        if version not in (2, 3):
            raise ValueError(f"unsupported GGUF version {version}")
        metadata = {}
        for _ in range(pair_count):
            key = reader.string()
            value = reader.value(reader.scalar("<I"), True)
            if value is not None and (not isinstance(value, str) or len(value) <= 256):
                metadata[key] = value
        infos = []
        for _ in range(tensor_count):
            name = reader.string()
            dims = [reader.scalar("<Q") for _ in range(reader.scalar("<I"))]
            kind, offset = reader.scalar("<I"), reader.scalar("<Q")
            infos.append(
                (name, tuple(reversed(dims)), TYPES.get(kind, f"type{kind}"), offset)
            )
        alignment = int(metadata.get("general.alignment", 32))
        start = -(-handle.tell() // alignment) * alignment
    return metadata, {
        name: Tensor(name, shape, kind, start + offset)
        for name, shape, kind, offset in infos
    }


def rows(path: Path, tensor: Tensor, indices: np.ndarray | None = None) -> np.ndarray:
    """Rows of a [rows, columns] tensor as float32. With `indices`, only those rows are read."""
    if len(tensor.shape) != 2:
        raise ValueError(f"{tensor.name} is not a matrix")
    count, columns = tensor.shape
    wanted = (
        np.arange(count) if indices is None else np.asarray(indices, dtype=np.int64)
    )
    if tensor.type == "Q8_0":
        if columns % 32:
            raise ValueError("Q8_0 rows need a multiple of 32 columns")
        row_bytes = columns // 32 * Q8_BLOCK
        raw = np.memmap(
            path,
            dtype=np.uint8,
            mode="r",
            offset=tensor.offset,
            shape=(count, row_bytes),
        )
        blocks = np.ascontiguousarray(raw[wanted]).reshape(
            len(wanted), columns // 32, Q8_BLOCK
        )
        scales = blocks[:, :, :2].copy().view("<f2").astype(np.float32)
        return (blocks[:, :, 2:].view(np.int8).astype(np.float32) * scales).reshape(
            len(wanted), columns
        )
    plain = {"F32": "<f4", "F16": "<f2", "BF16": "<u2"}.get(tensor.type)
    if plain is None:
        raise ValueError(
            f"{tensor.name} uses {tensor.type}, which this reader does not decode"
        )
    raw = np.memmap(
        path, dtype=plain, mode="r", offset=tensor.offset, shape=(count, columns)
    )
    picked = np.ascontiguousarray(raw[wanted])
    if tensor.type == "BF16":
        return (picked.astype(np.uint32) << np.uint32(16)).view(np.float32)
    return picked.astype(np.float32)
