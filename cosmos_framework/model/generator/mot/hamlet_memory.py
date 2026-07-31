# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""HAMLET-style episodic memory for Cosmos action policy.

Ports the block-causal ``MemoryTransformer`` algorithm from HAMLET
(Koo et al., ICLR 2026; arXiv:2510.00695 / HAMLET-Isaac-GR00T) into the
Cosmos Omni-MoT action-policy stack.

This module is intentionally self-contained and CPU-testable. Full MoT
wiring (moment-token packing, training-time history windows) is layered
on separately behind a config flag.

Shapes
------
Input / output of ``MemoryTransformer``: ``[B, T * n_q, D]`` with blocks
oldest → newest along the sequence axis. Use ``current_slice`` to take the
last ``n_q`` tokens (current-step memory summary).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class HamletConfig:
    """Hyperparameters for HAMLET-style memory (disabled by default)."""

    enabled: bool = False
    n_moment_tokens: int = 4
    memory_window: int = 4
    memory_num_layers: int = 2
    num_heads: int = 8
    ffn_mult: int = 4
    mem_cond_type: str = "cross_attn"  # "cross_attn" | "adaln"
    init_range: float = 0.02
    rms_eps: float = 1e-5
    # Inner width of MemoryTransformer / moment bank. 0 = use outer model dim.
    # Use e.g. 512 to avoid a ~0.5B-param memory tower at hidden_size=4096.
    memory_dim: int = 0


def build_block_causal_allow_mask(window: int, n_q: int) -> torch.Tensor:
    """Boolean allow-mask ``[L, L]`` for block-causal attention.

    Tokens within the same timestep block attend bidirectionally. Across
    blocks, attention is causal (a block may attend to itself and earlier
    blocks, never to the future). ``L = window * n_q``.
    """
    if window < 1 or n_q < 1:
        raise ValueError(f"window and n_q must be >= 1, got {window=}, {n_q=}")
    positions = torch.arange(window, dtype=torch.long).repeat_interleave(n_q)
    # allow[i, j] True iff block(j) <= block(i)
    return positions.unsqueeze(0) <= positions.unsqueeze(1)


def pool_primary_view(
    backbone_features: torch.Tensor,
    image_mask: torch.Tensor,
    tokens_per_view: int,
    grid_hw: tuple[int, int],
    out_side: int = 8,
) -> torch.Tensor:
    """Avg-pool the primary view's image tokens to a fixed ``out_side**2`` grid.

    Optional HAMLET ``memory_type="vision_feature"`` path. ``backbone_features``
    is not modified for other consumers.
    """
    batch_keep, _, dim = backbone_features.shape
    img = backbone_features[image_mask].view(batch_keep, -1, dim)[:, :tokens_per_view, :]
    grid_h, grid_w = grid_hw
    if grid_h * grid_w != tokens_per_view:
        raise ValueError(f"grid {grid_h}x{grid_w} != tokens_per_view {tokens_per_view}")
    grid = img.transpose(1, 2).reshape(batch_keep, dim, grid_h, grid_w)
    pooled = F.adaptive_avg_pool2d(grid.float(), (out_side, out_side)).to(img.dtype)
    return pooled.flatten(2).transpose(1, 2).contiguous()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    q_out = (q * cos) + (_rotate_half(q) * sin)
    k_out = (k * cos) + (_rotate_half(k) * sin)
    return q_out, k_out


class _RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = torch.einsum("i,j->ij", positions.to(torch.float32), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().unsqueeze(0).unsqueeze(0), emb.sin().unsqueeze(0).unsqueeze(0)


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        rms = x32.pow(2).mean(-1, keepdim=True).clamp_min(self.eps).rsqrt()
        return ((x32 * rms) * self.weight).to(dtype)


class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        out = out.transpose(1, 2).contiguous().view(batch, length, -1)
        return self.o_proj(out)


class _SwiGLU(nn.Module):
    def __init__(self, dim: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate, bias=False)
        self.up_proj = nn.Linear(dim, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class _Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_mult: int = 4, rms_eps: float = 1e-5):
        super().__init__()
        self.attn_norm = _RMSNorm(dim, rms_eps)
        self.attn = _Attention(dim, num_heads)
        self.ffn_norm = _RMSNorm(dim, rms_eps)
        self.ffn = _SwiGLU(dim, ffn_mult * dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), attn_mask, cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MemoryTransformer(nn.Module):
    """Block-causal Transformer for HAMLET history aggregation."""

    def __init__(
        self,
        dim: int,
        n_q: int,
        window: int,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_mult: int = 4,
        rms_eps: float = 1e-5,
        init_range: float = 0.02,
    ):
        super().__init__()
        self.dim = dim
        self.n_q = n_q
        self.window = window
        self.num_layers = num_layers
        seq_len = window * n_q
        self.blocks = nn.ModuleList(
            [_Block(dim, num_heads, ffn_mult, rms_eps) for _ in range(num_layers)]
        )
        self.final_norm = _RMSNorm(dim, rms_eps)
        head_dim = dim // num_heads
        self.rope = _RotaryEmbedding(head_dim)

        allow = build_block_causal_allow_mask(window, n_q)
        mask = torch.zeros(seq_len, seq_len, dtype=torch.float32)
        mask.masked_fill_(~allow, float("-inf"))
        positions = torch.arange(window, dtype=torch.long).repeat_interleave(n_q)
        self.register_buffer("attn_mask", mask.view(1, 1, seq_len, seq_len), persistent=False)
        self.register_buffer("positions", positions, persistent=False)

        self._init_range = init_range
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self._init_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: ``[B, T * n_q, D]``, block 0 = oldest, last block = current.
        Returns:
            ``[B, T * n_q, D]``.
        """
        _, length, dim = x.shape
        expected = self.window * self.n_q
        if length != expected:
            raise ValueError(f"expected seq_len {expected}, got {length}")
        if dim != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {dim}")
        cos, sin = self.rope(self.positions)
        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)
        attn_mask = self.attn_mask.to(dtype=x.dtype)
        for block in self.blocks:
            x = block(x, attn_mask, cos, sin)
        return self.final_norm(x)

    def current_slice(self, x: torch.Tensor) -> torch.Tensor:
        """Return the last ``n_q`` rows (current-step memory tokens)."""
        return x[:, -self.n_q :, :]


class MomentTokenBank(nn.Module):
    """Learnable moment tokens ``[n_q, D]`` expanded per batch/timestep."""

    def __init__(self, n_q: int, dim: int, init_range: float = 0.02):
        super().__init__()
        self.n_q = n_q
        self.dim = dim
        self.tokens = nn.Parameter(torch.empty(n_q, dim))
        nn.init.normal_(self.tokens, mean=0.0, std=init_range)

    def expand(self, batch: int, window: int = 1) -> torch.Tensor:
        """Return ``[B, window * n_q, D]`` copies of the learnable tokens."""
        if batch < 1 or window < 1:
            raise ValueError(f"batch and window must be >= 1, got {batch=}, {window=}")
        # [n_q, D] -> [1, 1, n_q, D] -> [B, window, n_q, D] -> [B, window*n_q, D]
        tokens = self.tokens.view(1, 1, self.n_q, self.dim).expand(batch, window, self.n_q, self.dim)
        return tokens.reshape(batch, window * self.n_q, self.dim).contiguous()


def condition_action_features(
    action_features: torch.Tensor,
    memory_current: torch.Tensor,
    mem_cond_type: str = "cross_attn",
) -> torch.Tensor:
    """Combine current-step memory with action-path features.

    Args:
        action_features: ``[B, T_a, D]`` action (or gen) tokens.
        memory_current: ``[B, n_q, D]`` current memory slice.
        mem_cond_type:
            - ``cross_attn``: concatenate memory tokens in front of action tokens
              (caller can feed the result into SA / CA as extra KV context).
            - ``adaln``: mean-pool memory and add it as a broadcast bias.

    Returns:
        Conditioned features. Shape is ``[B, n_q + T_a, D]`` for ``cross_attn``,
        or ``[B, T_a, D]`` for ``adaln``.
    """
    if mem_cond_type == "cross_attn":
        return torch.cat([memory_current, action_features], dim=1)
    if mem_cond_type == "adaln":
        bias = memory_current.mean(dim=1, keepdim=True)
        return action_features + bias
    raise ValueError(f"Unknown mem_cond_type={mem_cond_type!r}; expected 'cross_attn' or 'adaln'")


def append_moment_embeddings(
    und_embeddings: torch.Tensor,
    moment_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Append moment-token embeddings after understanding (text) embeddings.

    Args:
        und_embeddings: ``[B, L_text, D]`` or ``[L_text, D]`` packed text embeddings.
        moment_embeddings: ``[B, n_q, D]`` or ``[n_q, D]`` moment embeddings.

    Returns:
        Concatenation along the sequence axis with moments last.
    """
    if und_embeddings.ndim != moment_embeddings.ndim:
        raise ValueError(
            f"rank mismatch: und_embeddings.ndim={und_embeddings.ndim} "
            f"moment_embeddings.ndim={moment_embeddings.ndim}"
        )
    if und_embeddings.shape[-1] != moment_embeddings.shape[-1]:
        raise ValueError(
            f"dim mismatch: und {und_embeddings.shape[-1]} vs moment {moment_embeddings.shape[-1]}"
        )
    return torch.cat([und_embeddings, moment_embeddings], dim=-2)


def slice_moment_hidden(
    und_hidden: torch.Tensor,
    n_text: int,
    n_q: int,
) -> torch.Tensor:
    """Slice the trailing ``n_q`` und hidden states as moment outputs.

    Assumes moments were appended after ``n_text`` text tokens.
    """
    if n_text < 0 or n_q < 1:
        raise ValueError(f"need n_text>=0 and n_q>=1, got {n_text=}, {n_q=}")
    seq = und_hidden.shape[-2]
    if seq < n_text + n_q:
        raise ValueError(f"und_hidden seq {seq} < n_text+n_q ({n_text}+{n_q})")
    return und_hidden[..., n_text : n_text + n_q, :]


def stack_moment_history(moment_steps: list[torch.Tensor]) -> torch.Tensor:
    """Stack per-timestep moment hiddens ``[B, n_q, D]`` oldest→newest into ``[B, T*n_q, D]``."""
    if not moment_steps:
        raise ValueError("moment_steps must be non-empty")
    return torch.cat(moment_steps, dim=-2)


def apply_hamlet_to_packed_actions(
    packed_sequence: torch.Tensor,
    action_sequence_indexes: torch.Tensor,
    token_shapes: list[tuple[int, ...]],
    hamlet: "HamletMemory",
    moment_history: torch.Tensor | None = None,
) -> None:
    """Condition flat packed action slots in-place with HAMLET (length-preserving).

    Multi-sample packed sequences are handled sample-by-sample using
    ``token_shapes``. When ``moment_history`` is None (no dataset history yet),
    the learnable moment bank is repeated across the memory window so the
    memory transformer still runs and receives gradients.

    Both ``adaln`` and ``cross_attn`` write back with a length-preserving fuse
    (mean-pool memory → add to action tokens). Full KV-concat ``cross_attn``
    needs extra packed slots and is deferred.
    """
    if action_sequence_indexes.numel() == 0:
        return
    cursor = 0
    sample_i = 0
    for shape in token_shapes:
        num_tokens = int(shape[0])
        if num_tokens <= 0:
            continue
        idxs = action_sequence_indexes[cursor : cursor + num_tokens]
        cursor += num_tokens
        action_feats = packed_sequence[idxs].unsqueeze(0)  # [1, T, D]
        weight_dtype = next(hamlet.parameters()).dtype
        if moment_history is None:
            # Synthetic: expand learnable bank directly in mem_dim (no in/out roundtrip).
            hist = hamlet.moment_tokens.expand(1, window=hamlet.config.memory_window)
            hist = hist.to(device=action_feats.device, dtype=weight_dtype)
            mem_out = hamlet.memory(hist)
        else:
            if sample_i >= moment_history.shape[0]:
                raise ValueError(
                    f"moment_history batch {moment_history.shape[0]} < action samples needed "
                    f"(at least {sample_i + 1})"
                )
            hist = moment_history[sample_i : sample_i + 1]
            hist = hist.to(device=action_feats.device, dtype=weight_dtype)
            mem_out = hamlet.encode_history(hist)
        action_in = action_feats.to(dtype=weight_dtype)
        current = hamlet.current_to_outer(hamlet.memory.current_slice(mem_out))
        # Packing-safe length-preserving fuse (adaln-style) for both cond types.
        conditioned = action_in + current.mean(dim=1, keepdim=True)
        packed_sequence[idxs] = conditioned.squeeze(0).to(dtype=packed_sequence.dtype)
        sample_i += 1


class HamletMemory(nn.Module):
    """Bundle: moment tokens + memory transformer + conditioning helper.

    ``dim`` is the outer model width (action token dim). When
    ``config.memory_dim > 0`` and differs from ``dim``, history / moments run
    at ``memory_dim`` with Linear in/out projections.
    """

    def __init__(self, dim: int, config: HamletConfig | None = None):
        super().__init__()
        self.config = config or HamletConfig(enabled=True)
        cfg = self.config
        self.outer_dim = int(dim)
        mem_dim = int(cfg.memory_dim) if cfg.memory_dim and cfg.memory_dim > 0 else self.outer_dim
        self.mem_dim = mem_dim
        if mem_dim % cfg.num_heads != 0:
            raise ValueError(f"memory_dim={mem_dim} must be divisible by num_heads={cfg.num_heads}")
        self.in_proj = nn.Identity() if mem_dim == self.outer_dim else nn.Linear(self.outer_dim, mem_dim, bias=False)
        self.out_proj = nn.Identity() if mem_dim == self.outer_dim else nn.Linear(mem_dim, self.outer_dim, bias=False)
        if not isinstance(self.in_proj, nn.Identity):
            nn.init.normal_(self.in_proj.weight, mean=0.0, std=cfg.init_range)
            nn.init.normal_(self.out_proj.weight, mean=0.0, std=cfg.init_range)
        self.moment_tokens = MomentTokenBank(cfg.n_moment_tokens, mem_dim, cfg.init_range)
        self.memory = MemoryTransformer(
            dim=mem_dim,
            n_q=cfg.n_moment_tokens,
            window=cfg.memory_window,
            num_layers=cfg.memory_num_layers,
            num_heads=cfg.num_heads,
            ffn_mult=cfg.ffn_mult,
            rms_eps=cfg.rms_eps,
            init_range=cfg.init_range,
        )

    def current_moment_embeddings(self, batch: int) -> torch.Tensor:
        """Learnable moment embeddings in **outer** dim: ``[B, n_q, D_outer]``."""
        emb = self.moment_tokens.expand(batch, window=1)  # [B, n_q, mem_dim]
        return self.out_proj(emb)

    def encode_history(self, moment_history: torch.Tensor | None = None) -> torch.Tensor:
        """Run memory over a ``[B, T*n_q, D_outer]`` history (projected to mem_dim)."""
        if moment_history is None:
            raise ValueError("moment_history is required; pass stacked post-VLM moment states")
        return self.memory(self.in_proj(moment_history))

    def current_to_outer(self, current: torch.Tensor) -> torch.Tensor:
        """Map ``[B, n_q, mem_dim]`` current slice back to outer dim."""
        return self.out_proj(current)

    def forward(
        self,
        action_features: torch.Tensor,
        moment_history: torch.Tensor,
    ) -> torch.Tensor:
        """Encode history and condition action features with the current slice."""
        mem_out = self.encode_history(moment_history)
        current = self.current_to_outer(self.memory.current_slice(mem_out))
        return condition_action_features(action_features, current, self.config.mem_cond_type)
