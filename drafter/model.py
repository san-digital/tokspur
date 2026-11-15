# SPDX-License-Identifier: Apache-2.0
# The layer structure, the grouped dynamic convolution and the candidate selector follow dflash/model.py and dflash/model_mlx.py in z-lab/dflash (MIT, see third_party/dflash/LICENSE). The batched training forward, the absolute-position handling and the projected selector tables are Tokspur's.
"""Trainable MLX model of the DFlash2 drafter.

Parameter names and shapes are those the native loader requires (drafter/contract.py), so a trained model exports to a checkpoint that `tokspur serve --drafter dflash2` loads unchanged. One forward serves many anchor blocks that share one context, which is how training amortises a traced conversation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mlx.core as mx
from mlx import nn

from drafter import contract

MASKED = -1e9


@dataclass(frozen=True)
class DraftConfig:
    """Geometry of the drafter. The defaults are the only values the native loader accepts; tests use smaller ones."""

    hidden: int = contract.HIDDEN
    intermediate: int = contract.INTERMEDIATE
    layers: int = contract.LAYERS
    query_heads: int = contract.QUERY_HEADS
    key_value_heads: int = contract.KEY_VALUE_HEADS
    head_dim: int = contract.HEAD_DIM
    window: int = contract.WINDOW
    vocabulary: int = contract.VOCABULARY
    taps: int = len(contract.TARGET_LAYER_IDS)
    block: int = contract.BLOCK
    conv_kernel: int = contract.CONV_KERNEL
    conv_group: int = contract.CONV_GROUP
    selector_rank: int = contract.SELECTOR_RANK
    selector_top_k: int = contract.SELECTOR_TOP_K
    mask_token_id: int = 248077
    rms_eps: float = 1e-6
    rope_theta: float = 10000.0
    # Selector tables are either full per-token tables, as stored in a checkpoint, or projections of the target's own token rows that are written out in full at export.
    selector_tables: bool = True
    # Optional low-rank semantic residual projections added on top of warm-started selector tables and folded into the exported codebooks.
    selector_delta_projections: bool = False

    @property
    def heads_per_group(self) -> int:
        return self.query_heads // self.key_value_heads

    def describe(self) -> dict:
        return asdict(self)


@dataclass
class TargetRows:
    """Frozen rows borrowed from the target at run time: its input embedding and its output matrix, both [vocabulary, hidden]."""

    embedding: mx.array
    head: mx.array


def rotary(x: mx.array, positions: mx.array, theta: float) -> mx.array:
    """Rotate the two halves of each head by its absolute position, as the Metal kernels do. `positions` broadcasts over x.shape[:-1]."""
    half = x.shape[-1] // 2
    frequency = mx.power(theta, -2.0 * mx.arange(half).astype(mx.float32) / x.shape[-1])
    angle = positions[..., None].astype(mx.float32) * frequency
    cos, sin = mx.cos(angle), mx.sin(angle)
    first, second = x[..., :half], x[..., half:]
    return mx.concatenate(
        [first * cos - second * sin, second * cos + first * sin], axis=-1
    )


def as_bf16(x: mx.array) -> mx.array:
    """Round to bfloat16 and back, the precision the Metal kernels keep queries, keys and values in."""
    return x.astype(mx.bfloat16).astype(mx.float32)


def context_mask(
    anchors: mx.array, rows: int, context_positions: mx.array, window: int
) -> mx.array:
    """Which context rows each block row may attend to: [anchors, rows, context].

    The context holds the target's state for every token before the anchor. A block row sees the context positions inside a window that ends at its own position.
    """
    query = anchors[:, None] + mx.arange(rows)[None]
    key = context_positions[None, None, :]
    return (key < anchors[:, None, None]) & (query[:, :, None] - key < window)


def linear(inputs: int, outputs: int, scale: float) -> nn.Linear:
    layer = nn.Linear(inputs, outputs, bias=False)
    layer.weight = (
        mx.random.normal((outputs, inputs)) * scale
        if scale
        else mx.zeros((outputs, inputs))
    )
    return layer


class GroupedDynamicConv(nn.Module):
    """Two-sided, block-local convolution whose taps are a learned base plus a per-row correction shared by groups of channels."""

    def __init__(self, config: DraftConfig):
        super().__init__()
        self.kernel = config.conv_kernel
        self.group = config.conv_group
        groups = config.hidden // config.conv_group
        # Tap 0 is the row itself and starts at one; every other tap and the whole dynamic part start at zero, so a new model is an exact no-op here.
        identity = mx.concatenate(
            [
                mx.ones((2, 1, config.hidden)),
                mx.zeros((2, config.conv_kernel - 1, config.hidden)),
            ],
            axis=1,
        )
        self.base_kernel = identity
        self.kernel_projection = linear(
            config.hidden, 2 * config.conv_kernel * groups, 0.0
        )

    def prepare(self, hidden: mx.array) -> tuple[mx.array, mx.array]:
        dynamic = self.kernel_projection(hidden).reshape(
            *hidden.shape[:-1], 2, self.kernel, -1
        )
        return self.convolve(
            hidden, dynamic[..., 0, :, :], self.base_kernel[0]
        ), dynamic[..., 1, :, :]

    def finish(self, hidden: mx.array, dynamic: mx.array) -> mx.array:
        return self.convolve(hidden, dynamic, self.base_kernel[1])

    def convolve(self, hidden: mx.array, dynamic: mx.array, base: mx.array) -> mx.array:
        output = mx.zeros_like(hidden)
        for offset in range(self.kernel):
            # The convolution never crosses a block boundary: rows before the block's first row read as zero.
            values = (
                hidden
                if offset == 0
                else mx.concatenate(
                    [mx.zeros_like(hidden[:, :offset]), hidden[:, :-offset]], axis=1
                )
            )
            coefficient = base[offset] + mx.repeat(
                dynamic[:, :, offset], self.group, axis=-1
            )
            output = output + coefficient * values
        return output


class Attention(nn.Module):
    def __init__(self, config: DraftConfig):
        super().__init__()
        self.config = config
        scale = 0.02
        self.q_proj = linear(config.hidden, config.query_heads * config.head_dim, scale)
        self.k_proj = linear(
            config.hidden, config.key_value_heads * config.head_dim, scale
        )
        self.v_proj = linear(
            config.hidden, config.key_value_heads * config.head_dim, scale
        )
        self.o_proj = linear(config.query_heads * config.head_dim, config.hidden, scale)
        self.q_norm = nn.RMSNorm(config.head_dim, eps=config.rms_eps)
        self.k_norm = nn.RMSNorm(config.head_dim, eps=config.rms_eps)
        # The runtime rounds normalised and rotated queries and keys, and values, to bfloat16. Training does not; a parity check against the worker switches this on.
        self.emulate_bf16 = False

    def rotated(self, normalised: mx.array, positions: mx.array) -> mx.array:
        if not self.emulate_bf16:
            return rotary(normalised, positions, self.config.rope_theta)
        return as_bf16(rotary(as_bf16(normalised), positions, self.config.rope_theta))

    def stored(self, values: mx.array) -> mx.array:
        return as_bf16(values) if self.emulate_bf16 else values

    def context(
        self, fused: mx.array, positions: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Keys and values of the fused target context for this layer: each [key_value_heads, context, head_dim]."""
        config = self.config
        rows = fused.shape[0]
        keys = self.k_norm(
            self.k_proj(fused).reshape(rows, config.key_value_heads, config.head_dim)
        )
        keys = self.rotated(keys, positions[:, None])
        values = self.stored(
            self.v_proj(fused).reshape(rows, config.key_value_heads, config.head_dim)
        )
        return keys.transpose(1, 0, 2), values.transpose(1, 0, 2)

    def __call__(
        self,
        hidden: mx.array,
        positions: mx.array,
        context_keys: mx.array,
        context_values: mx.array,
        mask: mx.array,
    ) -> mx.array:
        config = self.config
        blocks, rows, _ = hidden.shape
        groups, per_group, dim = (
            config.key_value_heads,
            config.heads_per_group,
            config.head_dim,
        )
        heads = positions[:, :, None]
        queries = self.q_norm(
            self.q_proj(hidden).reshape(blocks, rows, config.query_heads, dim)
        )
        queries = self.rotated(queries, heads)
        keys = self.k_norm(self.k_proj(hidden).reshape(blocks, rows, groups, dim))
        keys = self.rotated(keys, heads).transpose(0, 2, 1, 3)
        values = self.stored(
            self.v_proj(hidden).reshape(blocks, rows, groups, dim).transpose(0, 2, 1, 3)
        )
        # Query head g * per_group + h reads key-value head g. Fold the heads of one group into extra rows so that the shared context is never copied per block: [blocks, groups, per_group * rows, dim].
        queries = queries.reshape(blocks, rows, groups, per_group, dim).transpose(
            0, 2, 3, 1, 4
        )
        queries = queries.reshape(blocks, groups, per_group * rows, dim)
        scale = dim**-0.5
        context_scores = (queries @ context_keys.swapaxes(-1, -2)[None]) * scale
        context_scores = mx.where(
            mx.tile(mask, (1, per_group, 1))[:, None], context_scores, MASKED
        )
        block_scores = (queries @ keys.swapaxes(-1, -2)) * scale
        weights = mx.softmax(
            mx.concatenate([context_scores, block_scores], axis=-1), axis=-1
        )
        context_rows = context_keys.shape[1]
        output = (
            weights[..., :context_rows] @ context_values[None]
            + weights[..., context_rows:] @ values
        )
        output = output.reshape(blocks, groups, per_group, rows, dim).transpose(
            0, 3, 1, 2, 4
        )
        return self.o_proj(output.reshape(blocks, rows, config.query_heads * dim))


class MLP(nn.Module):
    def __init__(self, config: DraftConfig):
        super().__init__()
        self.gate_proj = linear(config.hidden, config.intermediate, 0.02)
        self.up_proj = linear(config.hidden, config.intermediate, 0.02)
        self.down_proj = linear(config.intermediate, config.hidden, 0.02)

    def __call__(self, hidden: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class Layer(nn.Module):
    def __init__(self, config: DraftConfig):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.hidden, eps=config.rms_eps)
        self.attention_conv = GroupedDynamicConv(config)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden, eps=config.rms_eps)
        self.mlp_conv = GroupedDynamicConv(config)
        self.mlp = MLP(config)

    def __call__(self, hidden, positions, context_keys, context_values, mask):
        residual = hidden
        hidden, kernel = self.attention_conv.prepare(self.input_layernorm(hidden))
        hidden = self.self_attn(hidden, positions, context_keys, context_values, mask)
        hidden = residual + self.attention_conv.finish(hidden, kernel)
        residual = hidden
        hidden, kernel = self.mlp_conv.prepare(self.post_attention_layernorm(hidden))
        return residual + self.mlp_conv.finish(self.mlp(hidden), kernel)


class CandidateSelector(nn.Module):
    """Scores a transition from the previous token to each candidate: <P[previous] * W h, S[candidate]>."""

    def __init__(self, config: DraftConfig):
        super().__init__()
        self.tables = config.selector_tables
        self.delta = bool(config.selector_tables and config.selector_delta_projections)
        self.hidden_projection = linear(config.hidden, config.selector_rank, 0.02)
        # The successor side starts at zero so that a new selector leaves the plain logits unchanged.
        if self.tables:
            self.predecessor_codebook = (
                mx.random.normal((config.vocabulary, config.selector_rank)) * 0.02
            )
            self.successor_codebook = mx.zeros(
                (config.vocabulary, config.selector_rank)
            )
            if self.delta:
                self.delta_predecessor_projection = linear(
                    config.hidden, config.selector_rank, 0.0
                )
                self.delta_successor_projection = linear(
                    config.hidden, config.selector_rank, 0.0
                )
        else:
            self.predecessor_projection = linear(
                config.hidden, config.selector_rank, 0.02
            )
            self.successor_projection = linear(config.hidden, config.selector_rank, 0.0)

    def predecessor(self, tokens: mx.array, target: TargetRows | None) -> mx.array:
        if self.tables:
            base = self.predecessor_codebook[tokens]
            if self.delta:
                if target is None:
                    raise ValueError("selector_delta_projections requires target rows")
                base = base + self.delta_predecessor_projection(
                    target.embedding[tokens].astype(mx.float32)
                )
            return base
        return self.predecessor_projection(target.embedding[tokens].astype(mx.float32))

    def successor(self, tokens: mx.array, target: TargetRows | None) -> mx.array:
        if self.tables:
            base = self.successor_codebook[tokens]
            if self.delta:
                if target is None:
                    raise ValueError("selector_delta_projections requires target rows")
                base = base + self.delta_successor_projection(
                    target.head[tokens].astype(mx.float32)
                )
            return base
        return self.successor_projection(target.head[tokens].astype(mx.float32))

    def edges(
        self,
        hidden: mx.array,
        previous: mx.array,
        candidates: mx.array,
        target: TargetRows | None,
    ) -> mx.array:
        """Transition scores [..., candidates] for hidden [..., hidden], previous [...] and candidates [..., candidates]."""
        gate = self.predecessor(previous, target) * self.hidden_projection(hidden)
        return mx.sum(gate[..., None, :] * self.successor(candidates, target), axis=-1)


class DraftModel(nn.Module):
    def __init__(self, config: DraftConfig | None = None):
        super().__init__()
        self.config = config = config or DraftConfig()
        self.fc = linear(config.taps * config.hidden, config.hidden, 0.02)
        self.hidden_norm = nn.RMSNorm(config.hidden, eps=config.rms_eps)
        self.layers = [Layer(config) for _ in range(config.layers)]
        self.norm = nn.RMSNorm(config.hidden, eps=config.rms_eps)
        self.candidate_selector = CandidateSelector(config)

    def context(
        self, taps: mx.array, positions: mx.array
    ) -> list[tuple[mx.array, mx.array]]:
        """Per-layer keys and values for target taps [context, taps * hidden] at absolute positions [context]."""
        fused = self.hidden_norm(self.fc(taps))
        return [layer.self_attn.context(fused, positions) for layer in self.layers]

    def hidden(
        self,
        blocks: mx.array,
        anchors: mx.array,
        context: list[tuple[mx.array, mx.array]],
        context_positions: mx.array,
    ) -> mx.array:
        """Final hidden states [anchors, rows, hidden] for block inputs [anchors, rows, hidden] whose first row sits at absolute position `anchors`."""
        rows = blocks.shape[1]
        positions = anchors[:, None] + mx.arange(rows)[None]
        mask = context_mask(anchors, rows, context_positions, self.config.window)
        hidden = blocks
        for layer, (keys, values) in zip(self.layers, context):
            hidden = layer(hidden, positions, keys, values, mask)
        return self.norm(hidden)

    def block_inputs(
        self, anchor_tokens: mx.array, target: TargetRows, rows: int | None = None
    ) -> mx.array:
        """Block inputs: the anchor token's embedding followed by mask-token embeddings."""
        rows = rows or self.config.block
        anchor = target.embedding[anchor_tokens].astype(mx.float32)[:, None]
        mask = target.embedding[self.config.mask_token_id].astype(mx.float32)
        return mx.concatenate(
            [
                anchor,
                mx.broadcast_to(mask, (anchor.shape[0], rows - 1, mask.shape[-1])),
            ],
            axis=1,
        )
