# SPDX-License-Identifier: Apache-2.0
"""The drafter checkpoint contract must match the native loader and reject what it rejects."""

from __future__ import annotations

import copy
import importlib
import json
import math
import os
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def contract():
    # The drafter tooling is not part of the installed tokspur package, so it is imported from the checkout root.
    sys.path.insert(0, str(ROOT))
    try:
        return importlib.import_module("drafter.contract")
    finally:
        sys.path.remove(str(ROOT))


def valid_config(contract) -> dict:
    return {
        "architectures": ["DFlash2DraftModel"],
        "attention_bias": False,
        "is_causal": False,
        "hidden_act": "silu",
        "hidden_size": contract.HIDDEN,
        "intermediate_size": contract.INTERMEDIATE,
        "num_hidden_layers": contract.LAYERS,
        "num_attention_heads": contract.QUERY_HEADS,
        "num_key_value_heads": contract.KEY_VALUE_HEADS,
        "head_dim": contract.HEAD_DIM,
        "sliding_window": contract.WINDOW,
        "vocab_size": contract.VOCABULARY,
        "rms_norm_eps": 1e-06,
        "rope_parameters": {"rope_theta": 10000, "rope_type": "default"},
        "dflash_config": {
            "block_size": 8,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "mask_token_id": 248077,
            "selector_rank": 256,
            "selector_top_k": 16,
            "target_layer_ids": [5, 15, 25, 35, 45],
        },
    }


def valid_header(contract) -> tuple[dict, int]:
    header, offset = {}, 0
    for name, shape in contract.tensors().items():
        size = 2 * math.prod(shape)
        header[name] = {
            "dtype": "BF16",
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    return header, offset


def write_checkpoint(directory: Path, config: dict, header: dict, payload: int) -> None:
    (directory / "config.json").write_text(json.dumps(config))
    encoded = json.dumps(header).encode()
    with (directory / "model.safetensors").open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)) + encoded)
        # A sparse file stands in for the payload: only names, shapes and spans are checked.
        handle.truncate(8 + len(encoded) + payload)


def test_table_matches_the_published_checkpoint_size(contract):
    assert len(contract.tensors()) == 81
    assert contract.parameters() == 603_065_600


def test_table_matches_the_native_loader(contract):
    assert contract.check_loader(ROOT) == []


def test_loader_drift_is_reported(contract):
    source = (ROOT / contract.LOADER_SOURCE).read_text()
    changed = source.replace(
        'require_bf16("norm.weight", {hidden_size_})',
        'require_bf16("final_norm.weight", {hidden_size_})',
    )
    assert changed != source
    found = contract.loader_tensors(changed)
    assert "final_norm.weight" in found and "norm.weight" not in found


def test_valid_checkpoint_is_accepted(contract, tmp_path):
    header, payload = valid_header(contract)
    header["__metadata__"] = {"format": "pt"}
    write_checkpoint(tmp_path, valid_config(contract), header, payload)
    report = contract.check_directory(tmp_path)
    assert report["accepted"], report["problems"]
    assert report["extra_tensors"] == []


@pytest.mark.parametrize(
    "path, value",
    [
        (("architectures",), ["DFlashDraftModel"]),
        (("is_causal",), True),
        (("is_causal",), 0),
        (("hidden_act",), "gelu"),
        (("hidden_size",), 2560.0),
        (("num_hidden_layers",), 8),
        (("rms_norm_eps",), 0),
        (("rope_parameters", "rope_theta"), -1),
        (("dflash_config", "block_size"), 16),
        (("dflash_config", "mask_token_id"), 248320),
        (("dflash_config", "target_layer_ids"), [1, 12, 23, 34, 45]),
    ],
)
def test_config_the_loader_refuses(contract, path, value):
    config = valid_config(contract)
    holder = config
    for key in path[:-1]:
        holder = holder[key]
    holder[path[-1]] = value
    assert contract.check_config(config)


def test_missing_config_section_is_reported(contract):
    config = valid_config(contract)
    del config["dflash_config"]
    assert contract.check_config(config) == ["dflash_config must be a JSON object"]


def test_header_the_loader_refuses(contract):
    header, payload = valid_header(contract)
    assert contract.check_header(header, payload) == []

    def problems(change) -> list[str]:
        altered = copy.deepcopy(header)
        change(altered)
        return contract.check_header(altered, payload)

    name = "layers.3.mlp.down_proj.weight"
    assert problems(lambda h: h.pop(name)) == [f"{name}: missing"]
    assert problems(lambda h: h[name].update(dtype="F16"))
    assert problems(lambda h: h[name].update(shape=[7680, 2560]))
    assert problems(lambda h: h[name].update(data_offsets=[1, 3]))
    assert problems(lambda h: h[name].update(data_offsets=[0, payload + 2]))
    # A stray tensor is ignored when well formed and fatal when not, because the loader parses every entry.
    assert not problems(
        lambda h: h.update(extra={"dtype": "F32", "shape": [1], "data_offsets": [0, 4]})
    )
    assert problems(lambda h: h.update(extra={"dtype": "F32", "shape": [1]}))
    assert contract.check_header(header, payload - 2)


def test_published_checkpoint_when_available(contract):
    directory = os.environ.get("TOKSPUR_DFLASH2_DIR")
    if not directory:
        pytest.skip("set TOKSPUR_DFLASH2_DIR to check a real checkpoint")
    report = contract.check_directory(Path(directory))
    assert report["accepted"], report["problems"]
