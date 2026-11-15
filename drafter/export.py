# SPDX-License-Identifier: Apache-2.0
"""Write and read drafter checkpoints in the exact form the native loader accepts."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np

from drafter import contract


def to_bf16(values: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16 bits, to nearest with ties to even, the rounding the Metal kernels use."""
    values = np.ascontiguousarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("tensor holds a value that is not finite")
    bits = values.view(np.uint32)
    rounded = (
        (bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1)))
        >> np.uint32(16)
    ).astype(np.uint16)
    if ((rounded & 0x7F80) == 0x7F80).any():
        raise ValueError("tensor holds a value too large for bfloat16")
    return rounded


def from_bf16(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def config_json(
    mask_token_id: int = 248077, rms_norm_eps: float = 1e-6, rope_theta: int = 10000
) -> dict:
    """config.json for the fixed geometry. The keys beyond the loader's are the ones z-lab/dflash reads, so the checkpoint also loads there."""
    return {
        "architectures": ["DFlash2DraftModel"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "dflash_config": {
            "block_size": contract.BLOCK,
            "conv_group_size": contract.CONV_GROUP,
            "conv_kernel_size": contract.CONV_KERNEL,
            "mask_token_id": mask_token_id,
            "selector_rank": contract.SELECTOR_RANK,
            "selector_top_k": contract.SELECTOR_TOP_K,
            "target_layer_ids": list(contract.TARGET_LAYER_IDS),
        },
        "dtype": "bfloat16",
        "head_dim": contract.HEAD_DIM,
        "hidden_act": "silu",
        "hidden_size": contract.HIDDEN,
        "intermediate_size": contract.INTERMEDIATE,
        "is_causal": False,
        "layer_types": ["sliding_attention"] * contract.LAYERS,
        "max_position_embeddings": 262144,
        "model_type": "qwen3",
        "num_attention_heads": contract.QUERY_HEADS,
        "num_hidden_layers": contract.LAYERS,
        "num_key_value_heads": contract.KEY_VALUE_HEADS,
        "num_target_layers": contract.TARGET_LAYERS,
        "rms_norm_eps": rms_norm_eps,
        "rope_parameters": {"rope_theta": rope_theta, "rope_type": "default"},
        "sliding_window": contract.WINDOW,
        "use_sliding_window": True,
        "vocab_size": contract.VOCABULARY,
    }


def write_checkpoint(
    directory: Path, tensors: dict[str, np.ndarray], config: dict
) -> dict:
    """Write config.json and model.safetensors into a new directory and return the contract report. Nothing existing is overwritten."""
    wanted = contract.tensors()
    problems = [f"{name}: missing" for name in wanted if name not in tensors]
    problems += [
        f"{name}: not part of the checkpoint" for name in tensors if name not in wanted
    ]
    problems += [
        f"{name}: needs {list(shape)}, found {list(tensors[name].shape)}"
        for name, shape in wanted.items()
        if name in tensors and tuple(tensors[name].shape) != shape
    ]
    problems += contract.check_config(config)
    if problems:
        raise ValueError("; ".join(problems))

    header, offset = {}, 0
    for name, shape in wanted.items():
        size = 2 * math.prod(shape)
        header[name] = {
            "dtype": contract.DTYPE,
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)

    directory.mkdir(parents=True)
    with (directory / "model.safetensors").open("xb") as handle:
        handle.write(struct.pack("<Q", len(encoded)) + encoded)
        for name in wanted:
            handle.write(to_bf16(tensors[name]).tobytes())
    with (directory / "config.json").open("x") as handle:
        handle.write(json.dumps(config, indent=2) + "\n")

    report = contract.check_directory(directory)
    if not report["accepted"]:
        raise ValueError(
            "written checkpoint fails its own contract: "
            + "; ".join(report["problems"])
        )
    return report


def read_checkpoint(directory: Path) -> dict[str, np.ndarray]:
    """Every required tensor of a checkpoint as float32. This reads the tensor payload."""
    report = contract.check_directory(directory)
    if not report["accepted"]:
        raise ValueError("; ".join(report["problems"]))
    path = contract.regular_file(directory / "model.safetensors")
    header, payload_bytes = contract.read_header(path)
    payload = np.memmap(
        path, dtype=np.uint16, mode="r", offset=path.stat().st_size - payload_bytes
    )
    tensors = {}
    for name, shape in contract.tensors().items():
        begin, end = header[name]["data_offsets"]
        tensors[name] = from_bf16(np.asarray(payload[begin // 2 : end // 2])).reshape(
            shape
        )
    return tensors


def model_tensors(model, target=None) -> dict[str, np.ndarray]:
    """The checkpoint tensors of a drafter/model.py DraftModel. Projected selector tables are written out in full, which needs the target rows."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    flat = dict(tree_flatten(model.parameters()))
    tensors = {
        name: np.array(flat[name].astype(mx.float32))
        for name in contract.tensors()
        if name in flat
    }
    if not model.config.selector_tables:
        if target is None:
            raise ValueError(
                "projected selector tables need the target rows to be written out"
            )
        selector = "candidate_selector."
        for table, source in (
            ("predecessor", target.embedding),
            ("successor", target.head),
        ):
            projection = flat[f"{selector}{table}_projection.weight"].astype(mx.float32)
            tensors[f"{selector}{table}_codebook"] = np.array(
                source.astype(mx.float32) @ projection.T
            )
    elif model.config.selector_delta_projections:
        if target is None:
            raise ValueError(
                "selector delta projections need the target rows to be folded at export"
            )
        selector = "candidate_selector."
        for table, source in (
            ("predecessor", target.embedding),
            ("successor", target.head),
        ):
            delta = flat[f"{selector}delta_{table}_projection.weight"].astype(
                mx.float32
            )
            folded = (
                flat[f"{selector}{table}_codebook"].astype(mx.float32)
                + source.astype(mx.float32) @ delta.T
            )
            tensors[f"{selector}{table}_codebook"] = np.array(folded)
    return tensors


def load_tensors(model, tensors: dict[str, np.ndarray]) -> None:
    """Load checkpoint tensors into a DraftModel that keeps full selector tables."""
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    if not model.config.selector_tables:
        raise ValueError(
            "a checkpoint holds full selector tables; build the model with selector_tables=True"
        )
    model.update(
        tree_unflatten([(name, mx.array(value)) for name, value in tensors.items()])
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser(
        "init", help="write a randomly initialised checkpoint into a new directory"
    )
    init.add_argument("directory", type=Path)
    init.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    import mlx.core as mx

    from drafter.model import DraftModel

    # Writing a checkpoint is host work; staying off the GPU keeps it from disturbing a timed run.
    mx.set_default_device(mx.cpu)
    mx.random.seed(args.seed)
    report = write_checkpoint(
        args.directory, model_tensors(DraftModel()), config_json()
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
