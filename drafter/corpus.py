# SPDX-License-Identifier: Apache-2.0
"""Multilingual and <think> trace diagnostics and capture manifests for DFlash2 self-distillation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from drafter import contract
from drafter.data import Trace, TraceSet

ASCII_CODE_CEILING = 32_768
SHINJITAI_COMPOUND_FLOOR = 148_000
CONTROL_FLOOR = 248_044
CONTROL_CEILING = 248_077
THINK_OPEN_ID = 248_068
THINK_CLOSE_ID = 248_069
IM_END_ID = 248_046


def _prompt_opens_think(prompt: np.ndarray, think_open: int, think_close: int) -> bool:
    """Return True when the rendered chat prompt leaves an open <think> block for generation to complete."""
    inside = False
    for tok in prompt.tolist():
        if tok == think_open:
            inside = True
        elif tok == think_close:
            inside = False
    return inside


def _split_think_regions(
    generated: np.ndarray,
    think_open: int,
    think_close: int,
    *,
    initial_inside: bool = False,
) -> tuple[bool, np.ndarray, np.ndarray]:
    """Return whether <think>...</think> delimiters are balanced alongside boolean masks for reasoning and post-think prose."""
    think_mask = np.zeros(len(generated), dtype=bool)
    prose_mask = np.zeros(len(generated), dtype=bool)
    inside = initial_inside
    saw_close = False
    balanced = True
    for idx, tok in enumerate(generated.tolist()):
        if tok == think_open:
            if inside or saw_close:
                balanced = False
            inside = True
        elif tok == think_close:
            if not inside:
                balanced = False
            inside = False
            saw_close = True
        elif inside:
            think_mask[idx] = True
        else:
            prose_mask[idx] = True
    if inside:
        balanced = False
    if saw_close and not np.any(prose_mask):
        balanced = False
    return balanced, think_mask, prose_mask


def trace_band_counts(
    trace: Trace,
    *,
    cjk_floor: int = ASCII_CODE_CEILING,
    compound_floor: int = SHINJITAI_COMPOUND_FLOOR,
    control_floor: int = CONTROL_FLOOR,
    control_ceiling: int = CONTROL_CEILING,
    think_open: int = THINK_OPEN_ID,
    think_close: int = THINK_CLOSE_ID,
) -> dict:
    """Count token-ID bands and reasoning vs prose splits across the generated portion of one trace."""
    prompt = np.asarray(trace.tokens[: trace.prompt_tokens], dtype=np.int64)
    generated = np.asarray(trace.tokens[trace.prompt_tokens :], dtype=np.int64)
    prompt_opened_think = _prompt_opens_think(prompt, think_open, think_close)
    balanced, think_mask, prose_mask = _split_think_regions(
        generated,
        think_open,
        think_close,
        initial_inside=prompt_opened_think,
    )
    cjk_mask = (generated >= cjk_floor) & (generated < control_floor)
    compound_mask = (generated >= compound_floor) & (generated < control_floor)
    control_mask = (generated >= control_floor) & (generated < control_ceiling)
    ascii_mask = generated < cjk_floor
    prose_count = int(np.sum(prose_mask))
    prose_cjk = int(np.sum(prose_mask & cjk_mask))
    think_count = int(np.sum(think_mask))
    think_ascii = int(np.sum(think_mask & ascii_mask))
    return {
        "prompt_tokens": int(trace.prompt_tokens),
        "generated_tokens": int(len(generated)),
        "ascii_code_tokens": int(np.sum(ascii_mask)),
        "cjk_multilingual_tokens": int(np.sum(cjk_mask)),
        "shinjitai_compound_tokens": int(np.sum(compound_mask)),
        "control_tokens": int(np.sum(control_mask)),
        "think_open_count": int(np.sum(generated == think_open)),
        "think_close_count": int(np.sum(generated == think_close)),
        "think_span_tokens": think_count,
        "think_ascii_tokens": think_ascii,
        "post_think_tokens": prose_count,
        "post_think_cjk_tokens": prose_cjk,
        "has_think": bool(
            prompt_opened_think
            or np.any((generated == think_open) | (generated == think_close))
        ),
        "balanced_think": balanced,
        "max_token_id": int(generated.max()) if len(generated) else -1,
        "min_token_id": int(generated.min()) if len(generated) else 0,
    }


def token_band_report(
    traces: TraceSet,
    *,
    cjk_floor: int = ASCII_CODE_CEILING,
    compound_floor: int = SHINJITAI_COMPOUND_FLOOR,
    control_floor: int = CONTROL_FLOOR,
    control_ceiling: int = CONTROL_CEILING,
    think_open: int = THINK_OPEN_ID,
    think_close: int = THINK_CLOSE_ID,
) -> dict:
    """Summarise token-ID bands and <think> transition coverage across a TraceSet."""
    rows = [
        trace_band_counts(
            trace,
            cjk_floor=cjk_floor,
            compound_floor=compound_floor,
            control_floor=control_floor,
            control_ceiling=control_ceiling,
            think_open=think_open,
            think_close=think_close,
        )
        for trace in traces.traces
    ]
    generated = sum(r["generated_tokens"] for r in rows)
    cjk = sum(r["cjk_multilingual_tokens"] for r in rows)
    compounds = sum(r["shinjitai_compound_tokens"] for r in rows)
    think_span = sum(r["think_span_tokens"] for r in rows)
    think_ascii = sum(r["think_ascii_tokens"] for r in rows)
    prose = sum(r["post_think_tokens"] for r in rows)
    prose_cjk = sum(r["post_think_cjk_tokens"] for r in rows)
    denom = max(generated, 1)
    return {
        "conversations": len(rows),
        "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "generated_tokens": generated,
        "ascii_code_tokens": sum(r["ascii_code_tokens"] for r in rows),
        "cjk_multilingual_tokens": cjk,
        "shinjitai_compound_tokens": compounds,
        "control_tokens": sum(r["control_tokens"] for r in rows),
        "cjk_fraction": cjk / denom,
        "shinjitai_compound_fraction": compounds / denom,
        "think_open_count": sum(r["think_open_count"] for r in rows),
        "think_close_count": sum(r["think_close_count"] for r in rows),
        "think_conversations": sum(1 for r in rows if r["has_think"]),
        "balanced_think_conversations": sum(
            1 for r in rows if r["has_think"] and r["balanced_think"]
        ),
        "think_span_tokens": think_span,
        "think_ascii_fraction": think_ascii / max(think_span, 1),
        "post_think_tokens": prose,
        "post_think_cjk_fraction": prose_cjk / max(prose, 1),
    }


def validate_multilingual_think_corpus(
    traces: TraceSet,
    *,
    vocabulary: int = contract.VOCABULARY,
    cjk_floor: int = ASCII_CODE_CEILING,
    control_floor: int = CONTROL_FLOOR,
    think_open: int = THINK_OPEN_ID,
    think_close: int = THINK_CLOSE_ID,
    min_cjk_fraction: float = 0.05,
    require_balanced_think: bool = True,
) -> list[str]:
    """Return problems that make a self-distillation dump unsuitable for multilingual + <think> DFlash2 training."""
    if not len(traces):
        return ["corpus holds zero conversations"]
    problems: list[str] = []
    for index, trace in enumerate(traces.traces):
        all_tokens = np.asarray(trace.tokens, dtype=np.int64)
        if (
            len(all_tokens) == 0
            or int(all_tokens.min()) < 0
            or int(all_tokens.max()) >= vocabulary
        ):
            problems.append(
                f"conversation {index} holds token IDs outside [0, {vocabulary})"
            )
        if len(trace) - trace.prompt_tokens < 2:
            problems.append(f"conversation {index} has fewer than 2 generated tokens")
            continue
        counts = trace_band_counts(
            trace,
            cjk_floor=cjk_floor,
            compound_floor=min(SHINJITAI_COMPOUND_FLOOR, control_floor),
            control_floor=control_floor,
            control_ceiling=min(CONTROL_CEILING, vocabulary),
            think_open=think_open,
            think_close=think_close,
        )
        if (
            require_balanced_think
            and counts["has_think"]
            and not counts["balanced_think"]
        ):
            problems.append(
                f"conversation {index} has unbalanced <think>...</think> delimiters or empty post-think prose"
            )
    summary = token_band_report(
        traces,
        cjk_floor=cjk_floor,
        compound_floor=min(SHINJITAI_COMPOUND_FLOOR, control_floor),
        control_floor=control_floor,
        control_ceiling=min(CONTROL_CEILING, vocabulary),
        think_open=think_open,
        think_close=think_close,
    )
    if summary["generated_tokens"] > 0 and summary["cjk_fraction"] < min_cjk_fraction:
        problems.append(
            f"generated CJK/multilingual fraction {summary['cjk_fraction']:.4f} is below required {min_cjk_fraction:.4f}"
        )
    return problems


def capture_manifest(specs: list[dict], *, seed: int = 0) -> list[dict]:
    """Build a deterministic tap-capture request manifest for TOKSPUR_DFLASH2_TAP_DUMP_DIR replay."""
    if not specs:
        raise ValueError(
            "capture manifest requires at least one conversation specification"
        )
    seen: set[str] = set()
    normalised: list[dict] = []
    for index, item in enumerate(specs):
        conv_id = str(item.get("id", "")).strip()
        messages = item.get("messages")
        if not conv_id or conv_id in seen:
            raise ValueError(f"conversation {index} must have a unique non-empty id")
        if not isinstance(messages, list) or not messages:
            raise ValueError(
                f"conversation {conv_id!r} must provide a non-empty messages list"
            )
        seen.add(conv_id)
        normalised.append(
            {
                "id": conv_id,
                "language": str(item.get("language", "en")),
                "domain": str(item.get("domain", "general")),
                "thinking": bool(item.get("thinking", True)),
                "max_tokens": int(item.get("max_tokens", 1792)),
                "temperature": float(item.get("temperature", 0.6)),
                "messages": messages,
            }
        )
    order = np.random.default_rng(seed).permutation(len(normalised)).tolist()
    return [normalised[i] for i in order]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dumps",
        type=Path,
        nargs="+",
        required=True,
        help="tap dump files or directories holding taps.bin",
    )
    parser.add_argument("--min-cjk-fraction", type=float, default=0.05)
    parser.add_argument("--allow-unbalanced-think", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if validation reports any problem",
    )
    args = parser.parse_args(argv)

    traces = TraceSet.load(args.dumps)
    report = token_band_report(traces)
    problems = validate_multilingual_think_corpus(
        traces,
        min_cjk_fraction=args.min_cjk_fraction,
        require_balanced_think=not args.allow_unbalanced_think,
    )
    payload = {"summary": report, "problems": problems, "valid": not problems}
    print(json.dumps(payload, indent=2))
    return 1 if (args.strict and problems) else 0


if __name__ == "__main__":
    raise SystemExit(main())
