# SPDX-License-Identifier: Apache-2.0
"""Read the tap dump the worker writes with DFLASH2_TAP_DUMP=1 (native/include/tokspur/dflash2_tap_dump.hpp).

A dump is a sequence of records, each holding consecutive token ids and the target taps for them. A record that starts at position zero begins a new conversation. A record that starts before the end of the current one replaces everything from that position on, because the worker rewound.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC = 0x50544B54
VERSION = 1
HALF = 1
HEADER = struct.Struct("<8I")
SOURCES = ("prefill", "step", "verification")


@dataclass(frozen=True)
class Record:
    ordinal: int
    source: str
    first_position: int
    rows: int
    width: int
    token_offset: int
    tap_offset: int


@dataclass
class Conversation:
    """One conversation: its tokens and where each run of tap rows sits in the dump file."""

    width: int
    tokens: np.ndarray
    runs: list[tuple[int, int, int]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.tokens)

    def taps(
        self, payload: np.ndarray, start: int = 0, stop: int | None = None
    ) -> np.ndarray:
        """Tap rows [start, stop) as float16 [rows, width], read from the memory-mapped dump."""
        stop = len(self) if stop is None else stop
        parts = []
        for first, rows, offset in self.runs:
            low, high = max(start, first), min(stop, first + rows)
            if low < high:
                begin = offset + (low - first) * self.width
                parts.append(payload[begin : begin + (high - low) * self.width])
        found = np.concatenate(parts) if parts else np.empty(0, dtype=np.float16)
        if found.size != (stop - start) * self.width:
            raise ValueError("conversation is missing tap rows in the requested range")
        return found.reshape(stop - start, self.width)


def write_record(
    handle,
    source: int,
    first_position: int,
    tokens: np.ndarray,
    taps: np.ndarray,
    ordinal: int,
) -> None:
    """Write one record in the worker's format. Used by tests and by tools that re-pack dumps."""
    tokens = np.ascontiguousarray(tokens, dtype="<u4")
    taps = np.ascontiguousarray(taps, dtype="<f2")
    if taps.ndim != 2 or taps.shape[0] != tokens.size or not tokens.size:
        raise ValueError("taps must be [rows, width] with one row per token")
    handle.write(
        HEADER.pack(
            MAGIC,
            VERSION | (source << 16),
            first_position,
            tokens.size,
            taps.shape[1],
            HALF,
            ordinal,
            0,
        )
    )
    handle.write(tokens.tobytes())
    handle.write(taps.tobytes())


def records(path: Path) -> list[Record]:
    """Every record header in a dump. A truncated final record, as left by a stopped worker, is ignored."""
    found = []
    size = path.stat().st_size
    with path.open("rb") as handle:
        offset = 0
        while offset + HEADER.size <= size:
            handle.seek(offset)
            magic, version_source, first, rows, width, element, ordinal, zero = (
                HEADER.unpack(handle.read(HEADER.size))
            )
            if (
                magic != MAGIC
                or version_source & 0xFFFF != VERSION
                or element != HALF
                or zero
                or not rows
                or not width
            ):
                raise ValueError(
                    f"record at byte {offset} is not a version {VERSION} tap record"
                )
            source = version_source >> 16
            if source >= len(SOURCES) or ordinal != len(found):
                raise ValueError(
                    f"record at byte {offset} has an unknown source or is out of order"
                )
            token_offset = offset + HEADER.size
            tap_offset = token_offset + 4 * rows
            end = tap_offset + 2 * rows * width
            if end > size:
                break
            found.append(
                Record(
                    ordinal,
                    SOURCES[source],
                    first,
                    rows,
                    width,
                    token_offset,
                    tap_offset,
                )
            )
            offset = end
    return found


def conversations(path: Path) -> tuple[np.ndarray, list[Conversation]]:
    """The memory-mapped dump as float16 words and the conversations it holds, in order."""
    # A stopped worker can leave the file cut at an odd byte; map whole half-float words only.
    payload = np.memmap(path, dtype="<f2", mode="r", shape=(path.stat().st_size // 2,))
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    found: list[Conversation] = []
    current: Conversation | None = None
    for record in records(path):
        tokens = np.frombuffer(
            raw, dtype="<u4", count=record.rows, offset=record.token_offset
        )
        if (
            current is None
            or record.first_position == 0
            or record.width != current.width
        ):
            if record.first_position != 0:
                raise ValueError(
                    f"record {record.ordinal} starts a conversation at position {record.first_position}"
                )
            current = Conversation(record.width, np.empty(0, dtype=np.uint32))
            found.append(current)
        if record.first_position > len(current):
            raise ValueError(
                f"record {record.ordinal} leaves a gap after position {len(current)}"
            )
        if record.first_position < len(current):
            # The worker rewound: drop the rows from this position on before appending.
            keep = record.first_position
            current.tokens = current.tokens[:keep]
            current.runs = [
                (first, min(rows, keep - first), offset)
                for first, rows, offset in current.runs
                if first < keep
            ]
        current.tokens = np.concatenate([current.tokens, tokens])
        # Tap offsets are kept in float16 words, the unit of the memory map.
        current.runs.append(
            (record.first_position, record.rows, record.tap_offset // 2)
        )
    return payload, found
