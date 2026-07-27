# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU unit tests for HAMLET-style memory (no GPU required)."""

from __future__ import annotations

import torch

from cosmos_framework.model.generator.mot.hamlet_memory import (
    HamletConfig,
    HamletMemory,
    MemoryTransformer,
    MomentTokenBank,
    append_moment_embeddings,
    build_block_causal_allow_mask,
    condition_action_features,
    slice_moment_hidden,
    stack_moment_history,
)


def test_block_causal_allow_mask_structure() -> None:
    window, n_q = 3, 2
    allow = build_block_causal_allow_mask(window, n_q)
    assert allow.shape == (window * n_q, window * n_q)
    # Within block 0 (tokens 0,1): full bidirectional.
    assert bool(allow[0, 0] and allow[0, 1] and allow[1, 0] and allow[1, 1])
    # Block 1 can see block 0, not vice versa for future.
    assert bool(allow[2, 0] and allow[3, 1])
    assert not bool(allow[0, 2])
    assert not bool(allow[1, 3])
    # Last block sees everything.
    assert bool(allow[-1].all())


def test_memory_transformer_forward_and_current_slice() -> None:
    torch.manual_seed(0)
    dim, n_q, window = 32, 4, 4
    model = MemoryTransformer(dim=dim, n_q=n_q, window=window, num_layers=2, num_heads=4)
    x = torch.randn(2, window * n_q, dim)
    y = model(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    cur = model.current_slice(y)
    assert cur.shape == (2, n_q, dim)
    assert torch.equal(cur, y[:, -n_q:, :])


def test_memory_transformer_rejects_bad_shapes() -> None:
    model = MemoryTransformer(dim=16, n_q=2, window=2, num_layers=1, num_heads=4)
    with_bad_len = torch.randn(1, 3, 16)
    try:
        model(with_bad_len)
        raise AssertionError("expected ValueError for wrong seq_len")
    except ValueError as err:
        assert "seq_len" in str(err)


def test_moment_token_bank_expand_and_grad() -> None:
    bank = MomentTokenBank(n_q=4, dim=16)
    out = bank.expand(batch=3, window=2)
    assert out.shape == (3, 8, 16)
    loss = out.square().mean()
    loss.backward()
    assert bank.tokens.grad is not None
    assert torch.isfinite(bank.tokens.grad).all()


def test_condition_action_cross_attn_and_adaln() -> None:
    action = torch.randn(2, 5, 16)
    memory = torch.randn(2, 4, 16)
    cat = condition_action_features(action, memory, "cross_attn")
    assert cat.shape == (2, 9, 16)
    assert torch.equal(cat[:, :4], memory)
    assert torch.equal(cat[:, 4:], action)

    adaln = condition_action_features(action, memory, "adaln")
    assert adaln.shape == action.shape
    expected = action + memory.mean(dim=1, keepdim=True)
    assert torch.allclose(adaln, expected)


def test_hamlet_memory_bundle_forward() -> None:
    torch.manual_seed(1)
    cfg = HamletConfig(
        enabled=True,
        n_moment_tokens=4,
        memory_window=3,
        memory_num_layers=1,
        num_heads=4,
        mem_cond_type="cross_attn",
    )
    bundle = HamletMemory(dim=32, config=cfg)
    history = torch.randn(2, cfg.memory_window * cfg.n_moment_tokens, 32)
    action = torch.randn(2, 6, 32)
    out = bundle(action, history)
    assert out.shape == (2, cfg.n_moment_tokens + 6, 32)
    assert torch.isfinite(out).all()

    bundle.config.mem_cond_type = "adaln"
    out_adaln = bundle(action, history)
    assert out_adaln.shape == action.shape


def test_append_and_slice_moment_embeddings() -> None:
    text = torch.randn(2, 7, 16)
    moments = torch.randn(2, 4, 16)
    fused = append_moment_embeddings(text, moments)
    assert fused.shape == (2, 11, 16)
    sliced = slice_moment_hidden(fused, n_text=7, n_q=4)
    assert torch.equal(sliced, moments)


def test_stack_moment_history_oldest_to_newest() -> None:
    steps = [torch.full((1, 2, 4), float(i)) for i in range(3)]
    hist = stack_moment_history(steps)
    assert hist.shape == (1, 6, 4)
    assert torch.equal(hist[:, :2], steps[0])
    assert torch.equal(hist[:, -2:], steps[2])


def test_current_moment_embeddings_from_bank() -> None:
    bundle = HamletMemory(dim=16, config=HamletConfig(enabled=True, n_moment_tokens=3, num_heads=4))
    emb = bundle.current_moment_embeddings(batch=2)
    assert emb.shape == (2, 3, 16)
