# SPDX-License-Identifier: Apache-2.0
"""Check a DFlash2 drafter checkpoint against what the native loader accepts.

The loader in native/src/dflash2_drafter.cpp pins the drafter geometry to the compiled Metal kernels. This script states that contract once, checks a checkpoint directory against it without reading tensor payloads, and can re-derive the tensor table from the loader source so the two cannot drift apart.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOADER_SOURCE = "native/src/dflash2_drafter.cpp"

HIDDEN = 2560
INTERMEDIATE = 7680
LAYERS = 5
QUERY_HEADS = 32
KEY_VALUE_HEADS = 8
HEAD_DIM = 128
WINDOW = 2048
VOCABULARY = 248320
TARGET_LAYERS = 48
TARGET_LAYER_IDS = (5, 15, 25, 35, 45)
BLOCK = 8
CONV_KERNEL = 2
CONV_GROUP = 16
SELECTOR_RANK = 256
SELECTOR_TOP_K = 16
DTYPE = "BF16"
MAX_CONFIG_BYTES = 1 << 20
MAX_HEADER_BYTES = 16 << 20

TARGET_CONCAT = len(TARGET_LAYER_IDS) * HIDDEN
CONV_DYNAMIC = 2 * CONV_KERNEL * (HIDDEN // CONV_GROUP)
QUERY_WIDTH = QUERY_HEADS * HEAD_DIM
KEY_VALUE_WIDTH = KEY_VALUE_HEADS * HEAD_DIM

GLOBAL_TENSORS = {
    "fc.weight": (HIDDEN, TARGET_CONCAT),
    "hidden_norm.weight": (HIDDEN,),
    "norm.weight": (HIDDEN,),
    "candidate_selector.hidden_projection.weight": (SELECTOR_RANK, HIDDEN),
    "candidate_selector.predecessor_codebook": (VOCABULARY, SELECTOR_RANK),
    "candidate_selector.successor_codebook": (VOCABULARY, SELECTOR_RANK),
}

LAYER_TENSORS = {
    "input_layernorm.weight": (HIDDEN,),
    "attention_conv.base_kernel": (2, CONV_KERNEL, HIDDEN),
    "attention_conv.kernel_projection.weight": (CONV_DYNAMIC, HIDDEN),
    "self_attn.q_proj.weight": (QUERY_WIDTH, HIDDEN),
    "self_attn.k_proj.weight": (KEY_VALUE_WIDTH, HIDDEN),
    "self_attn.v_proj.weight": (KEY_VALUE_WIDTH, HIDDEN),
    "self_attn.q_norm.weight": (HEAD_DIM,),
    "self_attn.k_norm.weight": (HEAD_DIM,),
    "self_attn.o_proj.weight": (HIDDEN, QUERY_WIDTH),
    "post_attention_layernorm.weight": (HIDDEN,),
    "mlp_conv.base_kernel": (2, CONV_KERNEL, HIDDEN),
    "mlp_conv.kernel_projection.weight": (CONV_DYNAMIC, HIDDEN),
    "mlp.gate_proj.weight": (INTERMEDIATE, HIDDEN),
    "mlp.up_proj.weight": (INTERMEDIATE, HIDDEN),
    "mlp.down_proj.weight": (HIDDEN, INTERMEDIATE),
}

# Names the loader source uses for dimensions, with the values the geometry check fixes them to.
LOADER_SYMBOLS = {
    "hidden_size_": HIDDEN,
    "intermediate_size_": INTERMEDIATE,
    "head_dim_": HEAD_DIM,
    "vocab_size_": VOCABULARY,
    "selector_rank_": SELECTOR_RANK,
    "target_concat": TARGET_CONCAT,
}

REQUIRE_CALL = re.compile(
    r'require_bf16\(\s*(pfx\s*\+\s*)?"([^"]+)"\s*,\s*\{([^}]*)\}\s*\)'
)


def tensors() -> dict[str, tuple[int, ...]]:
    """Every tensor the loader requires, by full name."""
    table = dict(GLOBAL_TENSORS)
    for layer in range(LAYERS):
        for name, shape in LAYER_TENSORS.items():
            table[f"layers.{layer}.{name}"] = shape
    return table


def parameters() -> int:
    return sum(math.prod(shape) for shape in tensors().values())


def loader_tensors(source: str) -> dict[str, tuple[int, ...]]:
    """Re-derive the tensor table from the require_bf16 calls in the loader source."""
    table: dict[str, tuple[int, ...]] = {}
    for per_layer, name, dims in REQUIRE_CALL.findall(source):
        shape = []
        for token in (part.strip() for part in dims.split(",")):
            if token in LOADER_SYMBOLS:
                shape.append(LOADER_SYMBOLS[token])
            elif token.isdigit():
                shape.append(int(token))
            else:
                raise ValueError(f"unknown dimension {token!r} for {name}")
        names = (
            [f"layers.{layer}.{name}" for layer in range(LAYERS)]
            if per_layer
            else [name]
        )
        for full in names:
            if full in table:
                raise ValueError(f"loader requires {full} twice")
            table[full] = tuple(shape)
    return table


def is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def is_positive_number(value: object) -> bool:
    number = isinstance(value, (int, float)) and not isinstance(value, bool)
    return number and math.isfinite(value) and value > 0


def check_config(config: object) -> list[str]:
    """Problems that make the loader reject config.json, in the loader's own order."""
    if not isinstance(config, dict):
        return ["config.json is not a JSON object"]
    problems = []

    def expect(
        key: str, wanted: object, holder: dict = config, label: str = ""
    ) -> None:
        found = holder.get(key, None)
        # Compare as JSON text so that 2560.0 or 0 cannot stand in for 2560 or false: the loader reads integers and booleans strictly.
        if key not in holder or json.dumps(found) != json.dumps(wanted):
            problems.append(f"{label}{key} must be {wanted!r}, found {found!r}")

    expect("architectures", ["DFlash2DraftModel"])
    expect("is_causal", False)
    expect("attention_bias", False)
    expect("hidden_act", "silu")
    expect("hidden_size", HIDDEN)
    expect("intermediate_size", INTERMEDIATE)
    expect("num_hidden_layers", LAYERS)
    expect("num_attention_heads", QUERY_HEADS)
    expect("num_key_value_heads", KEY_VALUE_HEADS)
    expect("head_dim", HEAD_DIM)
    expect("sliding_window", WINDOW)
    expect("vocab_size", VOCABULARY)
    if not is_positive_number(config.get("rms_norm_eps")):
        problems.append("rms_norm_eps must be a finite number above zero")
    rope = config.get("rope_parameters")
    if not isinstance(rope, dict) or not is_positive_number(rope.get("rope_theta")):
        problems.append("rope_parameters.rope_theta must be a finite number above zero")

    draft = config.get("dflash_config")
    if not isinstance(draft, dict):
        problems.append("dflash_config must be a JSON object")
        return problems
    for key, wanted in (
        ("block_size", BLOCK),
        ("conv_group_size", CONV_GROUP),
        ("conv_kernel_size", CONV_KERNEL),
        ("selector_rank", SELECTOR_RANK),
        ("selector_top_k", SELECTOR_TOP_K),
        ("target_layer_ids", list(TARGET_LAYER_IDS)),
    ):
        expect(key, wanted, draft, "dflash_config.")
    mask = draft.get("mask_token_id")
    if not is_count(mask) or mask >= VOCABULARY:
        problems.append(
            "dflash_config.mask_token_id must be a token id below the vocabulary size"
        )
    return problems


def check_header(header: object, payload_bytes: int) -> list[str]:
    """Problems that make the loader reject the safetensors header."""
    if not isinstance(header, dict):
        return ["safetensors header is not a JSON object"]
    problems = []
    entries = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        # The loader parses every entry before it looks any tensor up, so a malformed extra entry also fails the load.
        offsets = meta.get("data_offsets") if isinstance(meta, dict) else None
        shape = meta.get("shape") if isinstance(meta, dict) else None
        well_formed = (
            isinstance(meta, dict)
            and isinstance(meta.get("dtype"), str)
            and isinstance(shape, list)
            and all(is_count(dim) for dim in shape)
            and isinstance(offsets, list)
            and len(offsets) == 2
            and all(is_count(offset) for offset in offsets)
        )
        if not well_formed:
            problems.append(f"{name}: entry needs dtype, shape and two data_offsets")
            continue
        begin, end = offsets
        if not begin <= end <= payload_bytes or begin % 2:
            problems.append(
                f"{name}: span {begin}..{end} is out of bounds or starts on an odd byte"
            )
            continue
        entries[name] = (meta["dtype"], tuple(shape), end - begin)

    for name, shape in tensors().items():
        if name not in entries:
            problems.append(f"{name}: missing")
            continue
        dtype, found, size = entries[name]
        if dtype != DTYPE or found != shape:
            problems.append(
                f"{name}: needs {DTYPE} {list(shape)}, found {dtype} {list(found)}"
            )
        elif size != 2 * math.prod(shape):
            problems.append(f"{name}: holds {size} bytes, needs {2 * math.prod(shape)}")
    return problems


def read_header(path: Path) -> tuple[dict, int]:
    """The header and the number of payload bytes after it. Tensor payloads are not read."""
    size = path.stat().st_size
    if size <= 16:
        raise ValueError("safetensors file is too small")
    with path.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        if not 2 < length <= MAX_HEADER_BYTES or 8 + length > size:
            raise ValueError("safetensors header length is out of bounds")
        header = json.loads(handle.read(length))
    return header, size - 8 - length


def regular_file(path: Path) -> Path:
    # The loader resolves the path first and then refuses anything that is not a regular file.
    resolved = Path(os.path.realpath(path))
    if not resolved.is_file():
        raise ValueError(f"{path.name} is missing or not a regular file")
    return resolved


def check_directory(directory: Path) -> dict:
    """Check config.json and model.safetensors in a checkpoint directory."""
    problems: list[str] = []
    extra: list[str] = []
    try:
        config_path = regular_file(directory / "config.json")
        if not 0 < config_path.stat().st_size <= MAX_CONFIG_BYTES:
            raise ValueError("config.json is empty or above 1 MiB")
        problems += check_config(json.loads(config_path.read_bytes()))
    except (OSError, ValueError) as error:
        problems.append(f"config.json: {error}")
    try:
        header, payload_bytes = read_header(
            regular_file(directory / "model.safetensors")
        )
        problems += check_header(header, payload_bytes)
        if isinstance(header, dict):
            known = tensors()
            extra = sorted(set(header) - set(known) - {"__metadata__"})
    except (OSError, ValueError, struct.error) as error:
        problems.append(f"model.safetensors: {error}")
    return {
        "directory": str(directory),
        "accepted": not problems,
        "problems": problems,
        "extra_tensors": extra,
        "required_tensors": len(tensors()),
        "required_parameters": parameters(),
    }


def check_loader(root: Path) -> list[str]:
    """Differences between this script's tensor table and the loader source."""
    found = loader_tensors((root / LOADER_SOURCE).read_text())
    wanted = tensors()
    problems = [
        f"loader no longer requires {name}" for name in sorted(set(wanted) - set(found))
    ]
    problems += [
        f"loader requires {name}, which the table lacks"
        for name in sorted(set(found) - set(wanted))
    ]
    for name in sorted(set(wanted) & set(found)):
        if wanted[name] != found[name]:
            problems.append(
                f"{name}: table has {list(wanted[name])}, loader has {list(found[name])}"
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="check a checkpoint directory")
    check.add_argument("directory", type=Path)
    commands.add_parser(
        "loader", help="compare the tensor table with the loader source"
    )
    commands.add_parser("table", help="print the required tensors")
    args = parser.parse_args(argv)

    if args.command == "check":
        report = check_directory(args.directory)
        print(json.dumps(report, indent=2))
        return 0 if report["accepted"] else 1
    if args.command == "loader":
        problems = check_loader(ROOT)
        print(json.dumps({"matches": not problems, "problems": problems}, indent=2))
        return 0 if not problems else 1
    for name, shape in tensors().items():
        print(f"{name}\t{DTYPE}\t{list(shape)}")
    print(f"{len(tensors())} tensors, {parameters()} parameters", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
