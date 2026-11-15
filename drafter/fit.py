# SPDX-License-Identifier: Apache-2.0
"""Train the drafter on traced conversations and export a checkpoint the worker loads.

  python -m drafter.fit --dumps DUMP [DUMP ...] --shard TARGET.gguf --out RUN_DIRECTORY

A run directory holds the log, the latest training state and, at the end, the exported checkpoint. A run can be stopped and continued with --resume: the order of steps is fixed by the seed, so a continued run takes exactly the steps the interrupted one would have taken.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import optimizers
from mlx.utils import tree_flatten, tree_unflatten

from drafter import export, gguf, train, vocab
from drafter.data import TraceSet
from drafter.model import DraftConfig, DraftModel, TargetRows


@dataclass
class FitOptions:
    epochs: int = 6
    anchors_per_sequence: int = 512
    # More than 256 anchors in one step was several times slower per position on the M5 Max (bench-001).
    anchors_per_step: int = 256
    learning_rate: float = 3e-4
    warmup_steps: int = 200
    gamma: float = 4.0
    selector_alpha: float = 1.0
    eal_weight: float = 0.0
    cjk_weight: float = 1.0
    think_boundary_weight: float = 1.0
    cjk_floor: int = 32_768
    control_floor: int = 248_044
    think_tokens: tuple[int, ...] = (248_068, 248_069)
    head_rows: int | None = None
    head_gather: str | None = None
    selector_delta_projections: bool = False
    freeze_selector_tables: bool = False
    checkpoint_every: int = 500
    log_every: int = 20
    validation_steps: int = 8
    max_steps: int | None = None
    seed: int = 0


def learning_rate_at(step: int, total: int, options: FitOptions) -> float:
    """Linear warm-up, then a cosine fall to zero at the last step. Without the warm-up the first updates overshoot and the loss rises."""
    if step < options.warmup_steps:
        return options.learning_rate * (step + 1) / options.warmup_steps
    progress = (step - options.warmup_steps) / max(1, total - options.warmup_steps)
    return options.learning_rate * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def objective(
    options: FitOptions, head_spec: int | train.HeadGather | None = None
) -> dict:
    return {
        "gamma": options.gamma,
        "selector_alpha": options.selector_alpha,
        "eal_weight": options.eal_weight,
        "head_rows": options.head_rows if head_spec is None else head_spec,
    }


def evaluate(
    model: DraftModel,
    target: TargetRows,
    traces: TraceSet,
    options: FitOptions,
    head_spec: int | train.HeadGather | None = None,
) -> dict:
    """Mean loss and metrics on held-out conversations, on the same anchors every time it is called."""
    config = model.config
    plan = traces.plan(
        np.random.default_rng(options.seed + 10_000),
        options.anchors_per_sequence,
        options.anchors_per_step,
    )
    sums: dict[str, float] = {}
    anchors_seen = 0
    for index, anchors in plan[: options.validation_steps]:
        total, metrics = train.losses(
            model,
            target,
            traces.batch(index, anchors, config.block, config.window),
            **objective(options, head_spec),
        )
        mx.eval(total, metrics)
        values = {
            "loss": total.item(),
            **{
                name: value.item()
                for name, value in metrics.items()
                if name != "positions"
            },
        }
        for name, value in values.items():
            sums[name] = sums.get(name, 0.0) + value * len(anchors)
        anchors_seen += len(anchors)
    return (
        {name: value / anchors_seen for name, value in sums.items()}
        | {"anchors": anchors_seen}
        if anchors_seen
        else {}
    )


def save_state(out: Path, model: DraftModel, optimizer, progress: dict) -> None:
    """Write the latest training state. Each file is written beside its final name and moved into place, so a stop mid-write leaves the previous state intact."""
    state = out / "state"
    state.mkdir(exist_ok=True)
    weights, moments = (
        state / "weights.next.safetensors",
        state / "optimizer.next.safetensors",
    )
    model.save_weights(str(weights))
    mx.save_safetensors(str(moments), dict(tree_flatten(optimizer.state)))
    (state / "progress.next.json").write_text(json.dumps(progress, indent=2) + "\n")
    for name in ("weights", "optimizer"):
        os.replace(state / f"{name}.next.safetensors", state / f"{name}.safetensors")
    os.replace(state / "progress.next.json", state / "progress.json")


def load_state(out: Path, model: DraftModel, optimizer) -> dict:
    state = out / "state"
    progress = json.loads((state / "progress.json").read_text())
    model.load_weights(str(state / "weights.safetensors"))
    optimizer.state = tree_unflatten(
        list(mx.load(str(state / "optimizer.safetensors")).items())
    )
    mx.eval(model.parameters(), optimizer.state)
    return progress


def fit(
    model: DraftModel,
    target: TargetRows,
    training: TraceSet,
    validation: TraceSet,
    options: FitOptions,
    out: Path,
    resume: bool = False,
    pause=None,
    head_spec: int | train.HeadGather | None = None,
) -> dict:
    """Train `model` in place. `pause` is called after every step; a long run uses it to give the machine to other jobs now and then."""
    config = model.config
    out.mkdir(parents=True, exist_ok=resume)
    optimizer = optimizers.AdamW(
        learning_rate=options.learning_rate, weight_decay=0.0, bias_correction=True
    )
    step_once = train.make_step(
        model, target, optimizer, **objective(options, head_spec)
    )
    # The plan of every epoch depends only on the seed, which is what lets a resumed run continue where the first one stopped.
    plans = [
        training.plan(
            np.random.default_rng(options.seed + epoch),
            options.anchors_per_sequence,
            options.anchors_per_step,
            block=config.block,
            cjk_weight=options.cjk_weight,
            think_boundary_weight=options.think_boundary_weight,
            cjk_floor=options.cjk_floor,
            control_floor=options.control_floor,
            think_tokens=options.think_tokens,
        )
        for epoch in range(options.epochs)
    ]
    steps = [(epoch, item) for epoch, plan in enumerate(plans) for item in plan]
    total = min(len(steps), options.max_steps) if options.max_steps else len(steps)
    if not total:
        raise ValueError("the training set has no anchors")
    done = load_state(out, model, optimizer)["step"] if resume else 0
    summary = {"steps": total, "resumed_from": done, "log": [], "validation": None}
    recent: list[dict] = []
    with (out / "log.jsonl").open("a") as log:

        def write(row: dict) -> None:
            log.write(json.dumps(row) + "\n")
            log.flush()

        for step in range(done, total):
            epoch, (index, anchors) = steps[step]
            optimizer.learning_rate = learning_rate_at(step, total, options)
            began = time.perf_counter()
            loss, metrics = step_once(
                training.batch(index, anchors, config.block, config.window)
            )
            mx.eval(model.parameters(), optimizer.state, loss, metrics)
            recent.append(
                {
                    "loss": loss.item(),
                    "seconds": time.perf_counter() - began,
                    **{name: value.item() for name, value in metrics.items()},
                }
            )
            last = step + 1 == total
            if (step + 1) % options.log_every == 0 or last:
                row = {
                    name: float(np.mean([r[name] for r in recent]))
                    for name in recent[0]
                    if name != "positions"
                }
                row |= {
                    "event": "train",
                    "step": step + 1,
                    "epoch": epoch,
                    "learning_rate": learning_rate_at(step, total, options),
                    "positions": int(sum(r["positions"] for r in recent)),
                }
                write(row)
                summary["log"].append(row)
                recent = []
            if (step + 1) % options.checkpoint_every == 0 or last:
                held = (
                    evaluate(model, target, validation, options, head_spec)
                    if len(validation)
                    else {}
                )
                if held:
                    summary["validation"] = held
                    write({"event": "validation", "step": step + 1, **held})
                save_state(
                    out,
                    model,
                    optimizer,
                    {"step": step + 1, "total": total, "validation": held},
                )
            if pause is not None:
                pause()
    return summary


class LockSlices:
    """Hold the serve lock while training, and give it up for a moment every so often so that other sessions' jobs are not shut out for hours."""

    def __init__(self, path: Path, seconds: float):
        self.handle = path.open("a+")
        self.seconds = seconds
        self.acquire()

    def acquire(self) -> None:
        while True:
            try:
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(5)
        self.until = time.monotonic() + self.seconds

    def __call__(self) -> None:
        if time.monotonic() < self.until:
            return
        # Return cached buffers first: another job may need the memory while this one waits.
        mx.clear_cache()
        fcntl.flock(self.handle, fcntl.LOCK_UN)
        # Waiting jobs poll every few seconds; stay out long enough for one of them to take the lock.
        time.sleep(12)
        self.acquire()


def build_initial_model(
    init_checkpoint: Path | None = None,
    *,
    resume: bool = False,
    selector_delta_projections: bool = False,
    freeze_selector_tables: bool = False,
) -> DraftModel:
    """Create a fresh projected-selector DraftModel or warm-start from an existing DFlash2 checkpoint directory."""
    if init_checkpoint is None:
        if selector_delta_projections or freeze_selector_tables:
            raise ValueError(
                "selector_delta_projections and freeze_selector_tables require --init-checkpoint"
            )
        return DraftModel(DraftConfig(selector_tables=False))
    model = DraftModel(
        DraftConfig(
            selector_tables=True,
            selector_delta_projections=selector_delta_projections,
        )
    )
    if not resume:
        export.load_tensors(model, export.read_checkpoint(init_checkpoint))
        mx.eval(model.parameters())
    if freeze_selector_tables:
        model.candidate_selector.freeze(
            keys=["predecessor_codebook", "successor_codebook"], recurse=False
        )
    return model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dumps",
        type=Path,
        nargs="+",
        required=True,
        help="tap dump files or directories holding taps.bin",
    )
    parser.add_argument(
        "--shard",
        type=Path,
        required=True,
        help="the GGUF shard holding token_embd.weight and output.weight",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="run directory; must not exist unless --resume is given",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="existing DFlash2 checkpoint directory to initialise weights from before training",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument(
        "--head-rows",
        type=int,
        default=0,
        help="restrict the logits to the first N token ids; 0 uses the full vocabulary",
    )
    parser.add_argument(
        "--head-gather",
        help="restrict the logits to a named vocab_gather preset (control32k, cjk48k, cjk64k, cjk80k) or ascending range list",
    )
    parser.add_argument(
        "--vocab-source",
        type=Path,
        help="GGUF file with tokenizer.ggml.tokens or tokenizer.json for named --head-gather presets",
    )
    parser.add_argument(
        "--selector-delta-projections",
        action="store_true",
        help="add zero-initialised semantic delta projections onto target.embedding and target.head for warm-started selector tables",
    )
    parser.add_argument(
        "--freeze-selector-tables",
        action="store_true",
        help="freeze predecessor_codebook and successor_codebook when warm-starting",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="skip writing the loadable checkpoint at the end",
    )
    parser.add_argument("--lock", type=Path, help="serve lock to hold while training")
    parser.add_argument(
        "--yield-seconds",
        type=float,
        default=900.0,
        help="how long to hold the lock before letting other jobs in",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="run on the CPU, for a smoke test only"
    )
    defaults = FitOptions()
    for name in (
        "epochs",
        "anchors_per_sequence",
        "anchors_per_step",
        "warmup_steps",
        "checkpoint_every",
        "log_every",
        "validation_steps",
        "seed",
    ):
        parser.add_argument(
            f"--{name.replace('_', '-')}", type=int, default=getattr(defaults, name)
        )
    for name in (
        "learning_rate",
        "gamma",
        "selector_alpha",
        "eal_weight",
        "cjk_weight",
        "think_boundary_weight",
    ):
        parser.add_argument(
            f"--{name.replace('_', '-')}", type=float, default=getattr(defaults, name)
        )
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args(argv)
    if args.head_rows and args.head_gather:
        raise ValueError("cannot combine --head-rows and --head-gather")

    if args.cpu:
        mx.set_default_device(mx.cpu)
    options = FitOptions(
        **{
            name: getattr(args, name)
            for name in asdict(defaults)
            if name
            not in (
                "head_rows",
                "head_gather",
                "selector_delta_projections",
                "freeze_selector_tables",
                "cjk_floor",
                "control_floor",
                "think_tokens",
            )
        },
        head_rows=args.head_rows or None,
        head_gather=args.head_gather or None,
        selector_delta_projections=bool(args.selector_delta_projections),
        freeze_selector_tables=bool(args.freeze_selector_tables),
    )
    mx.random.seed(options.seed)
    training, validation = TraceSet.load(args.dumps).split(
        args.validation_fraction, options.seed
    )
    _, tensors = gguf.header(args.shard)
    target = TargetRows(
        mx.array(gguf.rows(args.shard, tensors["token_embd.weight"])),
        mx.array(gguf.rows(args.shard, tensors["output.weight"])),
    )
    head_spec: int | train.HeadGather | None = options.head_rows
    if options.head_gather:
        tok_list = (
            vocab.read_vocabulary_tokens(args.vocab_source or args.shard)
            if options.head_gather in vocab.PRESETS
            and options.head_gather != "control32k"
            else ()
        )
        gather_ids = vocab.resolve_vocab_gather(
            options.head_gather, int(target.head.shape[0]), tok_list
        )
        head_spec = train.build_head_gather(target.head, gather_ids)
    model = build_initial_model(
        args.init_checkpoint,
        resume=args.resume,
        selector_delta_projections=options.selector_delta_projections,
        freeze_selector_tables=options.freeze_selector_tables,
    )
    pause = LockSlices(args.lock, args.yield_seconds) if args.lock else None

    described = {
        "options": asdict(options),
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "vocab_source": str(args.vocab_source) if args.vocab_source else None,
        "dumps": [str(path) for path in args.dumps],
        "shard": str(args.shard),
        "training": {"conversations": len(training), "tokens": training.tokens()},
        "validation": {"conversations": len(validation), "tokens": validation.tokens()},
    }
    summary = fit(
        model,
        target,
        training,
        validation,
        options,
        args.out,
        resume=args.resume,
        pause=pause,
        head_spec=head_spec,
    )
    (args.out / "run.json").write_text(
        json.dumps(
            described
            | {
                "steps": summary["steps"],
                "resumed_from": summary["resumed_from"],
                "validation_at_end": summary["validation"],
            },
            indent=2,
        )
        + "\n"
    )
    if not args.no_export:
        report = export.write_checkpoint(
            args.out / "checkpoint",
            export.model_tensors(model, target),
            export.config_json(),
        )
        print(
            json.dumps(
                {"exported": report["directory"], "accepted": report["accepted"]}
            )
        )
    print(
        json.dumps(
            {
                "steps": summary["steps"],
                "last": summary["log"][-1] if summary["log"] else None,
                "validation": summary["validation"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
