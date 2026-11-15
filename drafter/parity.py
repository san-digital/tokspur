# SPDX-License-Identifier: Apache-2.0
"""Compare this package's model with the worker, round by round, from a tap dump.

A dump written during generation holds one verification record per round. Its first position is the round's base and its row count is the anchor plus the drafts the target accepted. Running the same checkpoint through drafter/model.py on the dumped taps gives the drafts this code would have proposed, and so the number the target would have accepted. If the model code computes what the Metal kernels compute, the two numbers agree in every round.

The comparison holds for greedy requests served with all rows verified (`--native-dflash2-verify-rows all`), where a round always proposes as many drafts as the budget allows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from drafter import export, gguf, taps, vocab
from drafter.model import DraftConfig, DraftModel, TargetRows


def load_target(shard: Path, head_rows: int = 0) -> TargetRows:
    """The target's input embedding and either all or the first `head_rows` rows of its output matrix, decoded from the adopted GGUF shard."""
    _, tensors = gguf.header(shard)
    embedding = gguf.rows(shard, tensors["token_embd.weight"])
    indices = np.arange(head_rows) if head_rows > 0 else None
    head = gguf.rows(shard, tensors["output.weight"], indices)
    return TargetRows(mx.array(embedding), mx.array(head))


def load_model(checkpoint: Path, emulate_bf16: bool) -> DraftModel:
    model = DraftModel(DraftConfig())
    export.load_tensors(model, export.read_checkpoint(checkpoint))
    for layer in model.layers:
        layer.self_attn.emulate_bf16 = emulate_bf16
    mx.eval(model.parameters())
    return model


def proposals(
    model: DraftModel,
    target: TargetRows,
    tokens: np.ndarray,
    tap_rows: np.ndarray,
    bases: np.ndarray,
    chunk: int = 32,
    gather_ids: mx.array | None = None,
) -> np.ndarray:
    """The greedy drafts [bases, block - 1] for anchors at the given positions of one conversation."""
    positions = mx.arange(len(tokens))
    context = model.context(mx.array(tap_rows.astype(np.float32)), positions)
    mx.eval(context)
    top = model.config.selector_top_k
    head_mat = target.head[gather_ids] if gather_ids is not None else target.head
    edge_target = target if model.config.selector_delta_projections else None
    drafted = []
    for start in range(0, len(bases), chunk):
        anchors = mx.array(bases[start : start + chunk])
        anchor_tokens = mx.array(tokens[bases[start : start + chunk]].astype(np.int64))
        hidden = model.hidden(
            model.block_inputs(anchor_tokens, target), anchors, context, positions
        )[:, 1:]
        logits = hidden @ head_mat.T
        cand_rows = mx.argpartition(-logits, top - 1, axis=-1)[..., :top]
        unary = mx.take_along_axis(logits, cand_rows, axis=-1)
        candidates = gather_ids[cand_rows] if gather_ids is not None else cand_rows
        previous, path = anchor_tokens, []
        for position in range(hidden.shape[1]):
            scores = unary[:, position] + model.candidate_selector.edges(
                hidden[:, position], previous, candidates[:, position], edge_target
            )
            choice = mx.argmax(scores, axis=-1)
            previous = mx.take_along_axis(
                candidates[:, position], choice[:, None], axis=-1
            )[:, 0]
            path.append(previous)
        drafted.append(np.array(mx.stack(path, axis=1)))
    return (
        np.concatenate(drafted)
        if drafted
        else np.empty((0, model.config.block - 1), dtype=np.int64)
    )


def conversation_rounds(records: list[taps.Record]) -> list[list[taps.Record]]:
    grouped: list[list[taps.Record]] = []
    for record in records:
        if record.first_position == 0 or not grouped:
            grouped.append([])
        grouped[-1].append(record)
    return grouped


def compare(
    dump: Path,
    model: DraftModel,
    target: TargetRows,
    budgets: list[int],
    stops: set[int],
    max_drafts: int,
    gather_ids: mx.array | None = None,
) -> dict:
    payload, found = taps.conversations(dump)
    grouped = conversation_rounds(taps.records(dump))
    if len(found) != len(grouped) or len(found) != len(budgets):
        raise ValueError(
            f"dump holds {len(found)} conversations but {len(budgets)} budgets were given"
        )
    report = {
        "conversations": [],
        "rounds": 0,
        "agree": 0,
        "accepted_by_worker": 0,
        "accepted_by_model": 0,
        "disagreements": [],
    }
    for index, (conversation, rounds, budget) in enumerate(
        zip(found, grouped, budgets)
    ):
        tokens = conversation.tokens
        prompt = 0
        for record in rounds:
            if record.source != "prefill" or record.first_position != prompt:
                break
            prompt += record.rows
        verified = [r for r in rounds if r.source == "verification"]
        forward = all(
            a.first_position + a.rows <= b.first_position
            for a, b in zip(verified, verified[1:])
        )
        if not forward:
            raise ValueError(
                f"conversation {index} rewinds during generation; the comparison assumes it does not"
            )
        bases = np.array([r.first_position for r in verified], dtype=np.int64)
        drafted = proposals(
            model,
            target,
            tokens,
            conversation.taps(payload),
            bases,
            gather_ids=gather_ids,
        )
        agree = accepted_worker = accepted_model = 0
        for record, drafts in zip(verified, drafted):
            base, worker = record.first_position, record.rows - 1
            width = min(max_drafts, budget - (base - prompt) - 1)
            ours = 0
            # Count the leading drafts that equal the tokens the target went on to commit. The token after the last accepted draft is the next round's anchor, so it is in the dump unless generation ended there.
            while (
                ours < width
                and base + 1 + ours < len(tokens)
                and drafts[ours] == tokens[base + 1 + ours]
            ):
                ours += 1
                if int(tokens[base + ours]) in stops:
                    break
            agree += ours == worker
            accepted_worker += worker
            accepted_model += ours
            if ours != worker and len(report["disagreements"]) < 40:
                report["disagreements"].append(
                    {
                        "conversation": index,
                        "base": int(base),
                        "width": int(width),
                        "worker_accepted": int(worker),
                        "model_accepted": int(ours),
                        "model_drafts": [int(t) for t in drafts[:width]],
                        "committed_after_anchor": [
                            int(t) for t in tokens[base + 1 : base + 1 + width]
                        ],
                    }
                )
        report["conversations"].append(
            {
                "tokens": int(len(tokens)),
                "prompt_tokens": int(prompt),
                "generated_tokens": int(len(tokens) - prompt),
                "budget": int(budget),
                "rounds": len(verified),
                "single_steps": sum(r.source == "step" for r in rounds),
                "agree": int(agree),
                "accepted_by_worker": int(accepted_worker),
                "accepted_by_model": int(accepted_model),
            }
        )
        report["rounds"] += len(verified)
        report["agree"] += int(agree)
        report["accepted_by_worker"] += int(accepted_worker)
        report["accepted_by_model"] += int(accepted_model)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dump", type=Path, required=True, help="taps.bin written by the worker"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="the drafter directory the worker loaded",
    )
    parser.add_argument(
        "--shard",
        type=Path,
        required=True,
        help="the GGUF shard holding token_embd.weight and output.weight",
    )
    parser.add_argument(
        "--budgets",
        required=True,
        help="max_tokens of each request in the dump, comma separated",
    )
    parser.add_argument("--stops", default="", help="stop token ids, comma separated")
    parser.add_argument(
        "--head-rows",
        type=int,
        default=32768,
        help="the draft vocabulary slice the worker used; ignored when --head-gather is given",
    )
    parser.add_argument(
        "--head-gather",
        help="named vocab_gather preset (control32k, cjk48k, cjk64k, cjk80k) or ascending range list",
    )
    parser.add_argument(
        "--vocab-source",
        type=Path,
        help="GGUF file with tokenizer.ggml.tokens or tokenizer.json for named --head-gather presets",
    )
    parser.add_argument("--max-drafts", type=int, default=7)
    parser.add_argument(
        "--exact-attention",
        action="store_true",
        help="skip the bfloat16 rounding the runtime applies to queries, keys and values",
    )
    parser.add_argument(
        "--gpu", action="store_true", help="compute on the GPU instead of the CPU"
    )
    parser.add_argument(
        "--out", type=Path, help="write the report here; the file must not exist"
    )
    args = parser.parse_args(argv)

    if not args.gpu:
        mx.set_default_device(mx.cpu)
    model = load_model(args.checkpoint, not args.exact_attention)
    gather_ids = None
    if args.head_gather:
        target = load_target(args.shard, 0)
        tok_list = (
            vocab.read_vocabulary_tokens(args.vocab_source or args.shard)
            if args.head_gather in vocab.PRESETS and args.head_gather != "control32k"
            else ()
        )
        gather_ids = mx.array(
            vocab.resolve_vocab_gather(
                args.head_gather, int(target.head.shape[0]), tok_list
            )
        )
    else:
        target = load_target(args.shard, args.head_rows)
    report = compare(
        args.dump,
        model,
        target,
        [int(value) for value in args.budgets.split(",")],
        {int(value) for value in args.stops.split(",") if value},
        args.max_drafts,
        gather_ids=gather_ids,
    )
    report["settings"] = {
        "dump": str(args.dump),
        "checkpoint": str(args.checkpoint),
        "head_rows": None if args.head_gather else args.head_rows,
        "head_gather": args.head_gather,
        "max_drafts": args.max_drafts,
        "bf16_attention": not args.exact_attention,
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        with args.out.open("x") as handle:
            handle.write(text)
    summary = {
        key: report[key]
        for key in ("rounds", "agree", "accepted_by_worker", "accepted_by_model")
    }
    print(json.dumps(summary | {"disagreements_listed": len(report["disagreements"])}))
    return 0 if report["rounds"] and report["agree"] == report["rounds"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
