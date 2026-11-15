# SPDX-License-Identifier: Apache-2.0
"""Measure training speed of the full-size drafter on a tap dump.

This trains for a fixed time on the conversations of one dump, in a few batch shapes, and reports step time, supervised positions per second and the arithmetic rate that implies. It answers how much data this machine can train on. It is not a training run, since a dump of a few conversations is only enough to see the loss fall.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import optimizers
from mlx.utils import tree_flatten

from drafter import gguf, taps, train
from drafter.model import DraftConfig, DraftModel, TargetRows


def multiply_accumulates(
    config: DraftConfig, context: int, anchors: int, head_rows: int
) -> float:
    """Multiply-accumulates of one training step, counting three passes over trainable weights and two over the frozen output matrix."""
    hidden, dim = config.hidden, config.head_dim
    context_row = (
        config.taps * hidden * hidden
        + config.layers * 2 * config.key_value_heads * dim * hidden
    )
    conv = 2 * 2 * config.conv_kernel * (hidden // config.conv_group) * hidden
    attention = (
        config.query_heads + 2 * config.key_value_heads
    ) * dim * hidden + hidden * config.query_heads * dim
    scores = 2 * config.query_heads * dim * (context + config.block)
    block_row = config.layers * (
        conv + attention + 3 * config.intermediate * hidden + scores
    )
    trainable = context * context_row + anchors * config.block * block_row
    frozen = anchors * (config.block - 1) * head_rows * hidden
    return 3.0 * trainable + 2.0 * frozen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument(
        "--shard",
        type=Path,
        required=True,
        help="the GGUF shard holding token_embd.weight and output.weight",
    )
    parser.add_argument(
        "--shapes",
        default="256:0,512:0,512:32768",
        help="anchors:head_rows per phase, 0 meaning the full vocabulary",
    )
    parser.add_argument(
        "--minutes",
        type=float,
        default=9.0,
        help="total training time, split evenly over the phases",
    )
    parser.add_argument(
        "--context-limit",
        type=int,
        default=0,
        help="use only the last N positions of each conversation",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=30,
        help="steps over which the learning rate rises from zero",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cpu", action="store_true", help="run on the CPU, for a smoke test only"
    )
    parser.add_argument(
        "--out", type=Path, help="write the report here; the file must not exist"
    )
    args = parser.parse_args(argv)

    if args.cpu:
        mx.set_default_device(mx.cpu)
    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    config = DraftConfig(selector_tables=False)

    started = time.perf_counter()
    payload, found = taps.conversations(args.dump)
    conversations = []
    for conversation in found:
        first = (
            max(0, len(conversation) - args.context_limit) if args.context_limit else 0
        )
        conversations.append(
            (
                conversation.tokens[first:],
                np.array(conversation.taps(payload, first)),
                first,
            )
        )
    _, tensors = gguf.header(args.shard)
    target = TargetRows(
        mx.array(gguf.rows(args.shard, tensors["token_embd.weight"])),
        mx.array(gguf.rows(args.shard, tensors["output.weight"])),
    )
    model = DraftModel(config)
    optimizer = optimizers.AdamW(
        learning_rate=args.learning_rate, weight_decay=0.0, bias_correction=True
    )
    mx.eval(model.parameters(), target.embedding, target.head)
    trainable = sum(
        value.size for _, value in tree_flatten(model.trainable_parameters())
    )
    report = {
        "device": str(mx.default_device()),
        "mlx": mx.__version__,
        "trainable_parameters": int(trainable),
        "conversations": [int(len(tokens)) for tokens, _, _ in conversations],
        "load_seconds": round(time.perf_counter() - started, 1),
        "phases": [],
    }

    shapes = [
        tuple(int(part) for part in shape.split(":"))
        for shape in args.shapes.split(",")
    ]
    taken = 0
    for anchors, head_rows in shapes:
        rows = head_rows or config.vocabulary
        step = train.make_step(model, target, optimizer, head_rows=head_rows or None)
        deadline = time.perf_counter() + 60.0 * args.minutes / len(shapes)
        steps, failure = [], None
        while time.perf_counter() < deadline:
            tokens, tap_rows, first = conversations[
                int(rng.integers(len(conversations)))
            ]
            picked = rng.choice(
                np.arange(1, len(tokens) - 1),
                size=min(anchors, len(tokens) - 2),
                replace=False,
            )
            batch = train.build_batch(
                tokens, tap_rows, picked, config.block, first_position=first
            )
            # Without a warm-up the first updates move the zero-initialised convolution and selector parts too far and the loss rises.
            optimizer.learning_rate = args.learning_rate * min(
                1.0, (taken + 1) / max(1, args.warmup_steps)
            )
            taken += 1
            begin = time.perf_counter()
            try:
                total, metrics = step(batch)
                mx.eval(model.parameters(), optimizer.state, total, metrics)
            except Exception as error:  # a shape that does not fit must not lose the phases already measured
                failure = f"{type(error).__name__}: {error}"
                break
            steps.append(
                {
                    "seconds": time.perf_counter() - begin,
                    "context": int(len(tokens)),
                    "anchors": int(len(picked)),
                    "loss": float(total.item()),
                    "unary_loss": float(metrics["unary_loss"].item()),
                    "top1": float(metrics["top1"].item()),
                    "top_k_recall": float(metrics["top_k_recall"].item()),
                    "positions": int(metrics["positions"].item()),
                }
            )
        if failure or not steps:
            report["phases"].append(
                {
                    "anchors": anchors,
                    "head_rows": rows,
                    "steps": len(steps),
                    "error": failure or "no step finished",
                }
            )
            print(json.dumps(report["phases"][-1]), flush=True)
            continue
        # The first two steps of a shape include compilation and allocation.
        timed = steps[2:] or steps
        seconds = sum(s["seconds"] for s in timed)
        work = sum(
            multiply_accumulates(config, s["context"], s["anchors"], rows)
            for s in timed
        )
        report["phases"].append(
            {
                "anchors": anchors,
                "head_rows": rows,
                "steps": len(steps),
                "timed_steps": len(timed),
                "seconds_per_step": seconds / len(timed),
                "positions_per_second": sum(s["positions"] for s in timed) / seconds,
                "context_tokens_per_second": sum(s["context"] for s in timed) / seconds,
                "effective_tflops": 2.0 * work / seconds / 1e12,
                "first_loss": steps[0]["loss"],
                "last_loss": float(np.mean([s["loss"] for s in steps[-5:]])),
                "last_top1": float(np.mean([s["top1"] for s in steps[-5:]])),
                "last_top_k_recall": float(
                    np.mean([s["top_k_recall"] for s in steps[-5:]])
                ),
                "peak_memory_gib": mx.get_peak_memory() / 2**30,
                "steps_detail": steps,
            }
        )
        print(
            json.dumps(
                {k: v for k, v in report["phases"][-1].items() if k != "steps_detail"}
            ),
            flush=True,
        )

    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        with args.out.open("x") as handle:
            handle.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
