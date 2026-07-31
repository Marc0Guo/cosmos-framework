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


def test_apply_hamlet_to_packed_actions_preserves_length() -> None:
    torch.manual_seed(0)
    dim = 32
    hamlet = HamletMemory(
        dim=dim,
        config=HamletConfig(
            enabled=True,
            n_moment_tokens=4,
            memory_window=3,
            memory_num_layers=1,
            num_heads=4,
            mem_cond_type="adaln",
        ),
    )
    # Two samples, lengths 5 and 3, packed into a larger sequence buffer.
    packed = torch.randn(20, dim)
    idxs = torch.tensor([2, 3, 4, 5, 6, 10, 11, 12], dtype=torch.long)
    shapes = [(5,), (3,)]
    before = packed[idxs].clone()
    from cosmos_framework.model.generator.mot.hamlet_memory import apply_hamlet_to_packed_actions

    apply_hamlet_to_packed_actions(packed, idxs, shapes, hamlet)
    after = packed[idxs]
    assert after.shape == before.shape
    assert torch.isfinite(after).all()
    assert not torch.equal(after, before)


def test_apply_hamlet_uses_per_sample_moment_history() -> None:
    torch.manual_seed(0)
    dim, n_q, window = 32, 4, 3
    hamlet = HamletMemory(
        dim=dim,
        config=HamletConfig(
            enabled=True,
            n_moment_tokens=n_q,
            memory_window=window,
            memory_num_layers=1,
            num_heads=4,
        ),
    )
    packed = torch.zeros(20, dim)
    idxs = torch.tensor([2, 3, 4, 5, 10, 11], dtype=torch.long)
    shapes = [(4,), (2,)]
    # Distinct histories per sample so per-sample indexing is exercised.
    hist = torch.randn(2, window * n_q, dim)
    from cosmos_framework.model.generator.mot.hamlet_memory import apply_hamlet_to_packed_actions

    apply_hamlet_to_packed_actions(packed, idxs, shapes, hamlet, moment_history=hist)
    assert torch.isfinite(packed[idxs]).all()
    # Different histories should yield different offsets on the two samples.
    s0 = packed[idxs[:4]].mean()
    s1 = packed[idxs[4:]].mean()
    assert not torch.isclose(s0, s1)


def test_memory_dim_bottleneck_shapes_and_params() -> None:
    outer, mem = 64, 16
    hamlet = HamletMemory(
        dim=outer,
        config=HamletConfig(
            enabled=True,
            n_moment_tokens=4,
            memory_window=2,
            memory_num_layers=1,
            num_heads=4,
            memory_dim=mem,
            mem_cond_type="adaln",
        ),
    )
    assert hamlet.mem_dim == mem
    assert hamlet.outer_dim == outer
    n = sum(p.numel() for p in hamlet.parameters())
    # Far smaller than a full-width tower at outer=64.
    full = HamletMemory(
        dim=outer,
        config=HamletConfig(
            enabled=True,
            n_moment_tokens=4,
            memory_window=2,
            memory_num_layers=1,
            num_heads=4,
            memory_dim=0,
            mem_cond_type="adaln",
        ),
    )
    n_full = sum(p.numel() for p in full.parameters())
    assert n < n_full
    hist = torch.randn(2, 2 * 4, outer)
    out = hamlet(torch.randn(2, 5, outer), hist)
    assert out.shape == (2, 5, outer)
    from cosmos_framework.model.generator.mot.hamlet_memory import apply_hamlet_to_packed_actions

    packed = torch.randn(10, outer)
    apply_hamlet_to_packed_actions(packed, torch.arange(6), [(6,)], hamlet)
    assert torch.isfinite(packed[:6]).all()


def test_failure_moment_buffer_write_and_history() -> None:
    from cosmos_framework.model.generator.mot.hamlet_memory import FailureMomentBuffer

    buf = FailureMomentBuffer(n_q=2, dim=8, window=3, delta_r_threshold=-0.02)
    assert not buf.maybe_write(torch.randn(2, 8), r_t=0.5, r_prev=0.4)  # progress
    assert buf.num_slots == 0
    m0 = torch.ones(2, 8)
    assert buf.maybe_write(m0, r_t=0.3, r_prev=0.5)  # drop
    assert buf.num_slots == 1
    assert buf.write_count == 1
    # fill beyond window → keep last window
    for i in range(5):
        buf.maybe_write(torch.full((2, 8), float(i)), r_t=0.0, r_prev=1.0)
    assert buf.num_slots == 3
    hist = buf.as_history(batch=2)
    assert hist.shape == (2, 3 * 2, 8)
    assert torch.equal(hist[0], hist[1])


def test_failure_buffer_conditions_hamlet_without_weight_update() -> None:
    from cosmos_framework.model.generator.mot.hamlet_memory import FailureMomentBuffer

    torch.manual_seed(0)
    cfg = HamletConfig(
        enabled=True,
        n_moment_tokens=4,
        memory_window=3,
        memory_num_layers=1,
        num_heads=4,
        memory_dim=16,
        mem_cond_type="adaln",
    )
    hamlet = HamletMemory(dim=32, config=cfg)
    params_before = {k: v.detach().clone() for k, v in hamlet.state_dict().items()}
    buf = FailureMomentBuffer.from_hamlet(hamlet, delta_r_threshold=-0.01)
    for i in range(2):
        buf.maybe_write(torch.randn(4, 32), r_t=0.2 - 0.1 * i, r_prev=0.5)
    action = torch.randn(2, 5, 32)
    out = hamlet.condition_with_failure_buffer(action, buf)
    assert out.shape == action.shape
    assert torch.isfinite(out).all()
    # No parameter updates from the write/condition path.
    for k, v in hamlet.state_dict().items():
        assert torch.equal(v, params_before[k])


def test_failure_buffer_config_defaults() -> None:
    cfg = HamletConfig(enabled=True, failure_buffer=True, failure_jump_threshold=0.4)
    assert cfg.failure_buffer is True
    assert cfg.failure_jump_threshold == 0.4
    hamlet = HamletMemory(dim=32, config=HamletConfig(enabled=True, num_heads=4, failure_buffer=True))
    assert hamlet.config.failure_buffer is True
