# SPDX-License-Identifier: Apache-2.0
"""Training objective for the DFlash2 drafter: predict the target's next tokens from its own hidden states.

For each anchor token in a traced conversation the drafter sees the target's taps for every earlier position and must predict the tokens that follow the anchor. The plain logits are trained with a position-weighted cross-entropy (DFlash, arXiv 2602.06036). The selector is trained with a cross-entropy over the top candidates with the true previous token as predecessor. Upstream incoai has not published its own selector objective, and this form is the one the public re-implementations use.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from mlx import nn

from drafter.model import DraftModel, TargetRows

MASKED_LOG = -1e9


@dataclass
class Batch:
    """Anchor blocks that share one conversation's context."""

    taps: mx.array
    context_positions: mx.array
    anchors: mx.array
    anchor_tokens: mx.array
    labels: mx.array
    valid: mx.array


def build_batch(
    tokens: np.ndarray,
    taps: np.ndarray,
    anchors: np.ndarray,
    block: int,
    first_position: int = 0,
) -> Batch:
    """A batch for anchors given as indices into `tokens`. `taps[i]` is the target's state for `tokens[i]`, which sits at absolute position first_position + i."""
    tokens = np.asarray(tokens, dtype=np.int64)
    anchors = np.asarray(anchors, dtype=np.int64)
    if (
        len(tokens) != len(taps)
        or anchors.min() < 0
        or anchors.max() >= len(tokens) - 1
    ):
        raise ValueError("every anchor needs a token after it inside the conversation")
    following = anchors[:, None] + np.arange(1, block)[None]
    valid = following < len(tokens)
    labels = tokens[np.minimum(following, len(tokens) - 1)]
    return Batch(
        taps=mx.array(np.asarray(taps, dtype=np.float32)),
        context_positions=mx.array(first_position + np.arange(len(tokens))),
        anchors=mx.array(first_position + anchors),
        anchor_tokens=mx.array(tokens[anchors]),
        labels=mx.array(labels),
        valid=mx.array(valid),
    )


@dataclass(frozen=True)
class HeadGather:
    """Non-contiguous draft vocabulary rows matching native DFlash2 vocab_gather."""

    ids: mx.array
    row_of: mx.array
    head: mx.array


def build_head_gather(
    target_head: mx.array, token_ids: np.ndarray | mx.array
) -> HeadGather:
    """Materialise gathered output-head rows and a full-vocabulary-to-row lookup table."""
    ids_np = np.asarray(token_ids, dtype=np.int32)
    vocab_size = int(target_head.shape[0])
    if (
        ids_np.ndim != 1
        or ids_np.size == 0
        or int(ids_np[0]) < 0
        or int(ids_np[-1]) >= vocab_size
        or not np.all(ids_np[1:] > ids_np[:-1])
    ):
        raise ValueError(
            "HeadGather token_ids must be strictly ascending inside [0, vocabulary)"
        )
    row_of_np = np.full(vocab_size, -1, dtype=np.int32)
    row_of_np[ids_np] = np.arange(ids_np.size, dtype=np.int32)
    ids_mx = mx.array(ids_np)
    row_of_mx = mx.array(row_of_np)
    head_mx = target_head[ids_mx].astype(mx.float32)
    mx.eval(ids_mx, row_of_mx, head_mx)
    return HeadGather(ids=ids_mx, row_of=row_of_mx, head=head_mx)


def losses(
    model: DraftModel,
    target: TargetRows,
    batch: Batch,
    gamma: float = 4.0,
    selector_alpha: float = 1.0,
    head_rows: int | HeadGather | None = None,
    eal_weight: float = 0.0,
) -> tuple[mx.array, dict[str, mx.array]]:
    """Total loss and metrics for one batch.

    `head_rows` restricts the logits either to the first N token ids (`int`) or to an arbitrary gathered subset (`HeadGather`), matching `--dflash2-vocab-slice` and `--dflash2-vocab-gather`; a label outside the active draft set cannot be drafted and is left out of the loss.

    `eal_weight` adds the expected accepted length of arXiv 2609.24150 for greedy verification: the probability that every draft up to position k is the true token, summed over k, is maximised directly.
    """
    config = model.config
    context = model.context(batch.taps, batch.context_positions)
    blocks = model.block_inputs(batch.anchor_tokens, target)
    hidden = model.hidden(blocks, batch.anchors, context, batch.context_positions)[
        :, 1:
    ]
    if isinstance(head_rows, HeadGather):
        logits = hidden @ head_rows.head.T
        mapped = head_rows.row_of[batch.labels]
        valid = batch.valid & (mapped >= 0)
        row_labels = mx.where(valid, mapped, 0)
        prev_labels = mx.where(batch.valid, batch.labels, 0)
    else:
        head = target.head if head_rows is None else target.head[:head_rows]
        logits = hidden @ head.astype(mx.float32).T
        rows = logits.shape[-1]
        valid = batch.valid & (batch.labels < rows)
        row_labels = mx.where(valid, batch.labels, 0)
        prev_labels = row_labels

    positions = hidden.shape[1]
    weights = mx.exp(-mx.arange(positions).astype(mx.float32) / gamma)[None] * valid
    total_weight = mx.maximum(mx.sum(weights), 1.0)
    unary = (
        mx.sum(nn.losses.cross_entropy(logits, row_labels, reduction="none") * weights)
        / total_weight
    )

    # The selector only ever chooses among the top candidates, so it is trained on that set. When the true token is not in it, the weakest candidate gives up its place so that the choice stays defined.
    frozen = mx.stop_gradient(logits)
    top = config.selector_top_k
    cand_rows = mx.argpartition(-frozen, top - 1, axis=-1)[..., :top]
    present = mx.any(cand_rows == row_labels[..., None], axis=-1)
    weakest = mx.argmin(mx.take_along_axis(frozen, cand_rows, axis=-1), axis=-1)
    swap = (mx.arange(top)[None, None] == weakest[..., None]) & ~present[..., None]
    cand_rows = mx.where(swap, row_labels[..., None], cand_rows)
    candidates = (
        head_rows.ids[cand_rows] if isinstance(head_rows, HeadGather) else cand_rows
    )
    previous = mx.concatenate(
        [batch.anchor_tokens[:, None], prev_labels[:, :-1]], axis=1
    )
    scores = mx.take_along_axis(
        logits, cand_rows, axis=-1
    ) + model.candidate_selector.edges(hidden, previous, candidates, target)
    choice = mx.argmax(cand_rows == row_labels[..., None], axis=-1)
    selector = (
        mx.sum(nn.losses.cross_entropy(scores, choice, reduction="none") * weights)
        / total_weight
    )

    # A draft is accepted only while every earlier draft was right, which is exactly when teacher forcing the previous token is correct. A true token outside the runtime's candidates, or past the end, ends the accepted run.
    reachable = present & valid
    picked = mx.take_along_axis(
        nn.log_softmax(scores, axis=-1), choice[..., None], axis=-1
    )
    expected = mx.exp(
        mx.cumsum(mx.where(reachable, picked[..., 0], MASKED_LOG), axis=1)
    )
    hit = (mx.argmax(mx.stop_gradient(scores), axis=-1) == choice) & reachable
    anchors = mx.maximum(mx.sum(valid[:, 0]), 1)
    expected_accepted = mx.sum(expected) / anchors

    count = mx.maximum(mx.sum(valid), 1)
    metrics = {
        "expected_accepted": expected_accepted,
        "accepted": mx.sum(mx.cumprod(hit.astype(mx.float32), axis=1)) / anchors,
        "unary_loss": unary,
        "selector_loss": selector,
        "top1": mx.sum((mx.argmax(frozen, axis=-1) == row_labels) & valid) / count,
        "top_k_recall": mx.sum(present & valid) / count,
        "selector_top1": mx.sum(
            (mx.argmax(mx.stop_gradient(scores), axis=-1) == choice) & valid
        )
        / count,
        "positions": mx.sum(valid),
    }
    return unary + selector_alpha * selector - eal_weight * expected_accepted, metrics


def make_step(
    model: DraftModel, target: TargetRows, optimizer, clip: float = 1.0, **objective
):
    """A function that takes one optimiser step on a batch and returns the loss and metrics."""
    from mlx import optimizers

    def objective_of(batch: Batch):
        return losses(model, target, batch, **objective)

    value_and_grad = nn.value_and_grad(model, objective_of)

    def step(batch: Batch):
        (total, metrics), gradients = value_and_grad(batch)
        gradients, _ = optimizers.clip_grad_norm(gradients, clip)
        optimizer.update(model, gradients)
        return total, metrics

    return step
