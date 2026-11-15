# SPDX-License-Identifier: Apache-2.0
"""Draft vocabulary gather presets matching native/include/tokspur/speculative_controls.hpp."""

from __future__ import annotations

import json
import struct
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from drafter import gguf

QWEN_ORDINARY_VOCAB = 248_044
QWEN_CONTROL_END = 248_076
PRESETS = frozenset({"control32k", "cjk48k", "cjk64k", "cjk80k"})


def _reverse_byte_alphabet() -> list[int]:
    rev = [-1] * 324
    extension = 256
    for byte in range(256):
        visible = (33 <= byte <= 126) or (161 <= byte <= 172) or (byte >= 174)
        cp = byte if visible else extension
        if not visible:
            extension += 1
        rev[cp] = byte
    return rev


_REV_ALPHABET = _reverse_byte_alphabet()


def _decode_gpt2_byte_spelling(spelling: bytes) -> bytes | None:
    out = bytearray()
    i = 0
    n = len(spelling)
    while i < n:
        cp = spelling[i]
        i += 1
        if cp >= 128:
            if cp < 0xC2 or cp > 0xC5 or i == n:
                return None
            cont = spelling[i]
            i += 1
            if (cont & 0xC0) != 0x80:
                return None
            cp = ((cp & 31) << 6) | (cont & 63)
        if cp >= len(_REV_ALPHABET) or _REV_ALPHABET[cp] < 0:
            return None
        out.append(_REV_ALPHABET[cp])
    return bytes(out)


def _inspect_utf8_cjk(raw: bytes) -> tuple[bool, int, bool, bool]:
    cp_count = 0
    has_kana_or_punct = False
    has_han = False
    i = 0
    n = len(raw)
    while i < n:
        b0 = raw[i]
        if b0 < 0x80:
            cp = b0
            width = 1
        elif 0xC2 <= b0 <= 0xDF:
            if i + 2 > n:
                return False, 0, False, False
            b1 = raw[i + 1]
            if (b1 & 0xC0) != 0x80:
                return False, 0, False, False
            cp = ((b0 & 0x1F) << 6) | (b1 & 0x3F)
            width = 2
        elif 0xE0 <= b0 <= 0xEF:
            if i + 3 > n:
                return False, 0, False, False
            b1, b2 = raw[i + 1], raw[i + 2]
            if (b1 & 0xC0) != 0x80 or (b2 & 0xC0) != 0x80:
                return False, 0, False, False
            if (b0 == 0xE0 and b1 < 0xA0) or (b0 == 0xED and b1 > 0x9F):
                return False, 0, False, False
            cp = ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F)
            width = 3
        elif 0xF0 <= b0 <= 0xF4:
            if i + 4 > n:
                return False, 0, False, False
            b1, b2, b3 = raw[i + 1], raw[i + 2], raw[i + 3]
            if (b1 & 0xC0) != 0x80 or (b2 & 0xC0) != 0x80 or (b3 & 0xC0) != 0x80:
                return False, 0, False, False
            if (b0 == 0xF0 and b1 < 0x90) or (b0 == 0xF4 and b1 > 0x8F):
                return False, 0, False, False
            cp = (
                ((b0 & 0x07) << 18)
                | ((b1 & 0x3F) << 12)
                | ((b2 & 0x3F) << 6)
                | (b3 & 0x3F)
            )
            width = 4
        else:
            return False, 0, False, False
        cp_count += 1
        if (
            (0x3000 <= cp <= 0x30FF)
            or (0x31F0 <= cp <= 0x31FF)
            or (0xFF00 <= cp <= 0xFFEF)
        ):
            has_kana_or_punct = True
        if (
            (0x3400 <= cp <= 0x4DBF)
            or (0x4E00 <= cp <= 0x9FFF)
            or (0xF900 <= cp <= 0xFAFF)
        ):
            has_han = True
        i += width
    return True, cp_count, has_kana_or_punct, has_han


def parse_vocab_gather(value: str, vocabulary: int) -> np.ndarray:
    """Parse ascending comma-separated token IDs and inclusive ranges into int32 IDs."""
    refusal = "vocab_gather must be ascending token IDs or ranges such as 0-32734,248044-248076, a multiple of 32 in [256, vocabulary)"
    if not value or len(value) > 65_536 or vocabulary <= 0:
        raise ValueError(refusal)
    ids: list[int] = []
    at = 0
    n = len(value)

    def number() -> int:
        nonlocal at
        begin = at
        parsed = 0
        while at < n and "0" <= value[at] <= "9":
            parsed = parsed * 10 + (ord(value[at]) - 48)
            at += 1
            if parsed >= vocabulary:
                raise ValueError(refusal)
        if at == begin or (value[begin] == "0" and at - begin > 1):
            raise ValueError(refusal)
        return parsed

    while True:
        first = number()
        last = first
        if at < n and value[at] == "-":
            at += 1
            last = number()
        if last < first or (ids and first <= ids[-1]):
            raise ValueError(refusal)
        ids.extend(range(first, last + 1))
        if at == n:
            break
        if value[at] != ",":
            raise ValueError(refusal)
        at += 1

    if len(ids) < 256 or len(ids) >= vocabulary or len(ids) % 32 != 0:
        raise ValueError(refusal)
    return np.asarray(ids, dtype=np.int32)


def resolve_vocab_gather(
    value: str, vocabulary: int, tokens: Sequence[str | bytes] = ()
) -> np.ndarray:
    """Resolve a named preset or explicit range string into strictly ascending int32 token IDs."""
    if value in PRESETS:
        if vocabulary <= QWEN_CONTROL_END:
            raise ValueError(
                "named vocab_gather preset requires a Qwen3.8 vocabulary (> 248076)"
            )
        if value == "control32k":
            return parse_vocab_gather("0-32734,248044-248076", vocabulary)
    else:
        return parse_vocab_gather(value, vocabulary)

    if value == "cjk64k":
        total_rows, base_prefix = 65_536, 28_672
    elif value == "cjk80k":
        total_rows, base_prefix = 81_920, 32_768
    else:
        total_rows, base_prefix = 49_152, 24_576

    if len(tokens) <= QWEN_CONTROL_END or vocabulary <= total_rows:
        raise ValueError(
            "CJK vocab_gather preset requires vocabulary token strings covering Qwen3.8"
        )

    selected = bytearray(vocabulary)
    chosen = 0

    def add_id(tok_id: int) -> None:
        nonlocal chosen
        if not selected[tok_id]:
            selected[tok_id] = 1
            chosen += 1

    for tok_id in range(base_prefix):
        add_id(tok_id)
    for tok_id in range(QWEN_ORDINARY_VOCAB, QWEN_CONTROL_END + 1):
        add_id(tok_id)

    partial_utf8: list[int] = []
    kana_punct: list[int] = []
    cjk_kana: list[int] = []
    single_han: list[int] = []
    jp_ext_han: list[int] = []
    multi_han_2: list[int] = []
    multi_han_rest: list[int] = []

    for tok_id in range(QWEN_ORDINARY_VOCAB):
        item = tokens[tok_id]
        raw_sp = (
            item
            if isinstance(item, bytes)
            else item.encode("utf-8", errors="surrogateescape")
        )
        raw = _decode_gpt2_byte_spelling(raw_sp)
        if raw is None or not raw:
            continue
        ok_utf8, cp_count, has_kp, has_han = _inspect_utf8_cjk(raw)
        if not ok_utf8:
            partial_utf8.append(tok_id)
        elif has_kp and not has_han:
            kana_punct.append(tok_id)
        elif has_kp and has_han:
            cjk_kana.append(tok_id)
        elif has_han and cp_count == 1:
            single_han.append(tok_id)
        elif has_han and tok_id >= 148_000:
            jp_ext_han.append(tok_id)
        elif has_han and cp_count == 2:
            multi_han_2.append(tok_id)
        elif has_han:
            multi_han_rest.append(tok_id)

    for bucket in (
        partial_utf8,
        kana_punct,
        cjk_kana,
        single_han,
        jp_ext_han,
        multi_han_2,
        multi_han_rest,
    ):
        for tok_id in bucket:
            if chosen >= total_rows:
                break
            add_id(tok_id)
        if chosen >= total_rows:
            break

    for tok_id in range(base_prefix, QWEN_ORDINARY_VOCAB):
        if chosen >= total_rows:
            break
        add_id(tok_id)

    if chosen != total_rows:
        raise ValueError(
            "CJK vocab_gather preset failed to populate exact multiple-of-32 row count"
        )

    return np.flatnonzero(np.frombuffer(selected, dtype=np.uint8)).astype(np.int32)


def read_vocabulary_tokens(path: Path) -> list[str]:
    """Read token spellings ordered by ID from either a GGUF file holding tokenizer.ggml.tokens or a Hugging Face tokenizer.json."""
    with path.open("rb") as handle:
        prefix = handle.read(4)
    if prefix == gguf.MAGIC:
        return _read_gguf_tokens(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    vocab = data.get("model", {}).get("vocab")
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError(f"{path} is neither a GGUF file nor a tokenizer.json")
    added = data.get("added_tokens") or []
    max_id = max(
        max(int(v) for v in vocab.values()),
        max((int(a["id"]) for a in added), default=-1),
    )
    out = [""] * (max_id + 1)
    for spelling, idx in vocab.items():
        out[int(idx)] = spelling
    for item in added:
        out[int(item["id"])] = str(item["content"])
    return out


def _read_gguf_tokens(path: Path) -> list[str]:
    with path.open("rb") as handle:
        if handle.read(4) != gguf.MAGIC:
            raise ValueError(f"{path} is not a GGUF file")
        version, _, pair_count = struct.unpack("<IQQ", handle.read(20))
        if version not in (2, 3):
            raise ValueError(f"unsupported GGUF version {version}")

        def read_str() -> str:
            length = struct.unpack("<Q", handle.read(8))[0]
            return handle.read(length).decode("utf-8", errors="surrogateescape")

        def skip(kind: int) -> None:
            if kind in gguf.SCALARS:
                handle.seek(struct.calcsize(gguf.SCALARS[kind]), 1)
            elif kind == gguf.STRING:
                length = struct.unpack("<Q", handle.read(8))[0]
                handle.seek(length, 1)
            elif kind == gguf.ARRAY:
                elem, count = struct.unpack("<IQ", handle.read(12))
                if elem in gguf.SCALARS:
                    handle.seek(struct.calcsize(gguf.SCALARS[elem]) * count, 1)
                else:
                    for _ in range(count):
                        skip(elem)
            else:
                raise ValueError(f"unknown GGUF metadata type {kind}")

        for _ in range(pair_count):
            key = read_str()
            kind = struct.unpack("<I", handle.read(4))[0]
            if key == "tokenizer.ggml.tokens":
                if kind != gguf.ARRAY:
                    raise ValueError("tokenizer.ggml.tokens is not an array")
                elem, count = struct.unpack("<IQ", handle.read(12))
                if elem != gguf.STRING:
                    raise ValueError("tokenizer.ggml.tokens is not a string array")
                return [read_str() for _ in range(count)]
            skip(kind)
    raise ValueError(f"{path} does not hold tokenizer.ggml.tokens")
