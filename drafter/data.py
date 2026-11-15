# SPDX-License-Identifier: Apache-2.0
"""Training data for the drafter: traced conversations read from tap dumps.

A trace is one conversation's tokens with the target's taps for every position. Anchors are drawn from the part the target generated, because a drafter is only ever asked to continue the target's own output. Each step takes anchors that sit close together, so that only the context window they can see has to be loaded and scored.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from drafter import contract, taps, train


@dataclass
class Trace:
    tokens: np.ndarray
    prompt_tokens: int
    conversation: taps.Conversation
    payload: np.ndarray

    def __len__(self) -> int:
        return len(self.tokens)

    def anchors(self) -> np.ndarray:
        """Positions that may serve as an anchor: generated tokens with at least one token after them."""
        return np.arange(max(self.prompt_tokens, 1), len(self.tokens) - 1)

    def anchor_weights(
        self,
        block: int = contract.BLOCK,
        *,
        cjk_weight: float = 1.0,
        think_boundary_weight: float = 1.0,
        cjk_floor: int = 32_768,
        control_floor: int = 248_044,
        think_tokens: tuple[int, ...] = (248_068, 248_069),
    ) -> np.ndarray:
        """Sampling weights for eligible anchors, up-weighting high-ID CJK label spans and <think>/</think> transition blocks."""
        if not (
            math.isfinite(cjk_weight)
            and cjk_weight > 0.0
            and math.isfinite(think_boundary_weight)
            and think_boundary_weight > 0.0
        ):
            raise ValueError(
                "cjk_weight and think_boundary_weight must be finite numbers above zero"
            )
        eligible = self.anchors()
        weights = np.ones(len(eligible), dtype=np.float64)
        if not len(eligible) or (cjk_weight == 1.0 and think_boundary_weight == 1.0):
            return weights
        tokens = np.asarray(self.tokens, dtype=np.int64)
        offsets = eligible[:, None] + np.arange(block)[None, :]
        valid = offsets < len(tokens)
        window = tokens[np.minimum(offsets, len(tokens) - 1)]
        if cjk_weight != 1.0 and block > 1:
            cjk_labels = (
                valid[:, 1:]
                & (window[:, 1:] >= cjk_floor)
                & (window[:, 1:] < control_floor)
            )
            weights = np.where(
                np.any(cjk_labels, axis=1), weights * cjk_weight, weights
            )
        if think_boundary_weight != 1.0 and think_tokens:
            boundary = valid & np.isin(window, np.asarray(think_tokens, dtype=np.int64))
            weights = np.where(
                np.any(boundary, axis=1), weights * think_boundary_weight, weights
            )
        return weights


def load_dump(path: Path) -> list[Trace]:
    """The traces of one dump. A file conversations.json beside it may give each conversation's prompt length; without it the leading prefill records count as the prompt."""
    path = path / "taps.bin" if path.is_dir() else path
    payload, conversations = taps.conversations(path)
    groups: list[list[taps.Record]] = []
    for record in taps.records(path):
        if record.first_position == 0 or not groups:
            groups.append([])
        groups[-1].append(record)
    sidecar = path.with_name("conversations.json")
    declared = json.loads(sidecar.read_text()) if sidecar.exists() else None
    if declared is not None and len(declared) != len(conversations):
        raise ValueError(
            f"{sidecar} describes {len(declared)} conversations, the dump holds {len(conversations)}"
        )
    traces = []
    for index, (conversation, records) in enumerate(zip(conversations, groups)):
        if declared is not None:
            prompt = int(declared[index]["prompt_tokens"])
        else:
            prompt = 0
            for record in records:
                if record.source != "prefill" or record.first_position != prompt:
                    break
                prompt += record.rows
            # A dump made by replaying whole conversations is all prefill; without a sidecar every position is then eligible.
            prompt = 0 if prompt == len(conversation) else prompt
        if not 0 <= prompt <= len(conversation):
            raise ValueError(
                f"conversation {index} of {path} has an impossible prompt length {prompt}"
            )
        traces.append(Trace(conversation.tokens, prompt, conversation, payload))
    return traces


@dataclass
class TraceSet:
    traces: list[Trace]

    @classmethod
    def load(cls, dumps: list[Path]) -> "TraceSet":
        return cls([trace for dump in dumps for trace in load_dump(dump)])

    def __len__(self) -> int:
        return len(self.traces)

    def tokens(self) -> int:
        return sum(len(trace) for trace in self.traces)

    def split(
        self, validation_fraction: float, seed: int
    ) -> tuple["TraceSet", "TraceSet"]:
        """Training and validation sets, split by whole conversation."""
        order = np.random.default_rng(seed).permutation(len(self.traces))
        held = int(round(validation_fraction * len(self.traces)))
        held = min(
            max(held, 1 if validation_fraction > 0 and len(self.traces) > 1 else 0),
            len(self.traces) - 1,
        )
        validation = sorted(order[:held].tolist())
        training = sorted(order[held:].tolist())
        return TraceSet([self.traces[i] for i in training]), TraceSet(
            [self.traces[i] for i in validation]
        )

    def plan(
        self,
        rng: np.random.Generator,
        anchors_per_sequence: int,
        anchors_per_step: int,
        *,
        block: int = contract.BLOCK,
        cjk_weight: float = 1.0,
        think_boundary_weight: float = 1.0,
        cjk_floor: int = 32_768,
        control_floor: int = 248_044,
        think_tokens: tuple[int, ...] = (248_068, 248_069),
    ) -> list[tuple[int, np.ndarray]]:
        """One epoch as a shuffled list of (trace index, anchor positions). Anchors of a step are neighbours in position."""
        if not (
            math.isfinite(cjk_weight)
            and cjk_weight > 0.0
            and math.isfinite(think_boundary_weight)
            and think_boundary_weight > 0.0
        ):
            raise ValueError(
                "cjk_weight and think_boundary_weight must be finite numbers above zero"
            )
        steps = []
        stratified = cjk_weight != 1.0 or think_boundary_weight != 1.0
        for index, trace in enumerate(self.traces):
            eligible = trace.anchors()
            if not len(eligible):
                continue
            take = min(anchors_per_sequence, len(eligible))
            probabilities = None
            if stratified and take < len(eligible):
                weights = trace.anchor_weights(
                    block,
                    cjk_weight=cjk_weight,
                    think_boundary_weight=think_boundary_weight,
                    cjk_floor=cjk_floor,
                    control_floor=control_floor,
                    think_tokens=think_tokens,
                )
                probabilities = weights / weights.sum()
            chosen = np.sort(
                rng.choice(
                    eligible,
                    size=take,
                    replace=False,
                    p=probabilities,
                )
            )
            steps += [
                (index, chosen[start : start + anchors_per_step])
                for start in range(0, len(chosen), anchors_per_step)
            ]
        return [steps[i] for i in rng.permutation(len(steps))]

    def batch(
        self, index: int, anchors: np.ndarray, block: int, window: int
    ) -> train.Batch:
        """The batch for one step: only the context these anchors can see, and the tokens after the last anchor that serve as labels."""
        trace = self.traces[index]
        low = max(0, int(anchors.min()) - (window - 1))
        high = min(len(trace), int(anchors.max()) + block)
        return train.build_batch(
            trace.tokens[low:high],
            trace.conversation.taps(trace.payload, low, high),
            anchors - low,
            block,
            first_position=low,
        )
