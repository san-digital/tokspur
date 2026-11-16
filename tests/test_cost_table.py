# SPDX-License-Identifier: Apache-2.0
"""The cost-table profiles must record the controller costs the worker compiles."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "native/include/tokspur/mtp_anchor_state.hpp"
PROFILES = ("profiles/cost-table.json", "profiles/cost-table-m5-max-128.json")


def compiled_costs() -> dict:
    """The StageACostTable constants, read from the header text."""
    text = HEADER.read_text()
    table = text[text.index("struct StageACostTable") :]

    def scalar(name: str) -> str:
        found = re.search(rf"static constexpr \w+ {name} = ([0-9.]+);", table)
        assert found, name
        return found.group(1)

    def rounds(prefix: str) -> dict:
        return {f"k{k}": int(scalar(f"{prefix}t_round_k{k}_us")) for k in (1, 2, 3)}

    lookup = table[table.index("lookup_t_ord_us") :]
    lookup = lookup[: lookup.index("static double expected_tokens")]
    nax, ordinary = lookup.split("if (verify_nax) {", 1)[1].split("}", 1)

    def steps(body: str) -> dict:
        bounded = re.findall(r"context_length <= (\d+)\) return (\d+);", body)
        last = re.findall(r"^\s*return (\d+);", body, flags=re.MULTILINE)
        assert len(bounded) == 3 and len(last) == 1, body
        costs = {bound: int(cost) for bound, cost in bounded}
        costs["4096"] = int(last[0])
        return costs

    lut = re.search(r"f_cal_lut = \{([^}]*)\}", table)
    assert lut
    return {
        "parameters": {
            "alpha_0": float(scalar("alpha_0")),
            "beta": float(scalar("beta")),
            "eta_exit": float(scalar("eta_exit")),
            "eta_enter": float(scalar("eta_enter")),
            "probe_interval": int(scalar("probe_interval")),
        },
        "t_ord_us": steps(ordinary),
        "t_round_us": rounds(""),
        "verify_nax": {"t_ord_us": steps(nax), "t_round_us": rounds("verify_nax_")},
        "f_cal": {"lut": [float(value) for value in lut.group(1).split(",")]},
    }


@pytest.mark.parametrize("name", PROFILES)
def test_profile_records_the_compiled_costs(name):
    profile = json.loads((ROOT / name).read_text())
    recorded = {
        key: value for key, value in profile.items() if key not in ("schema", "sha256")
    }
    assert recorded == compiled_costs()


@pytest.mark.parametrize("name", PROFILES)
def test_profile_hash_covers_its_own_content(name):
    profile = json.loads((ROOT / name).read_text())
    digest = profile.pop("sha256")
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode()).hexdigest() == digest
