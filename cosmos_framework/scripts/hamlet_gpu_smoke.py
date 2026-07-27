# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""GPU smoke / micro-benchmark for HAMLET memory on Cosmos.

Runs without LIBERO data. Exercises ``HamletMemory`` on CUDA with several
hyperparameter variants and prints a machine-readable summary block for
autoresearch logging.

Usage (from cosmos-framework root, venv active):

    python -m cosmos_framework.scripts.hamlet_gpu_smoke
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import torch

from cosmos_framework.model.generator.mot.hamlet_memory import HamletConfig, HamletMemory


@dataclass
class SmokeResult:
    name: str
    ok: bool
    forward_ms: float
    peak_vram_mb: float
    out_norm: float
    n_params_m: float
    error: str = ""


def _run_one(
    name: str,
    *,
    dim: int,
    batch: int,
    action_len: int,
    cfg: HamletConfig,
    steps: int = 20,
    warmup: int = 5,
) -> SmokeResult:
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        model = HamletMemory(dim=dim, config=cfg).to(device=device, dtype=torch.bfloat16)
        model.train()
        n_params_m = sum(p.numel() for p in model.parameters()) / 1e6
        history = torch.randn(
            batch,
            cfg.memory_window * cfg.n_moment_tokens,
            dim,
            device=device,
            dtype=torch.bfloat16,
        )
        action = torch.randn(batch, action_len, dim, device=device, dtype=torch.bfloat16)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

        for _ in range(warmup):
            out = model(action, history)
            loss = out.float().pow(2).mean()
            loss.backward()
            opt.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        last_norm = 0.0
        for _ in range(steps):
            out = model(action, history)
            loss = out.float().pow(2).mean()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            last_norm = float(out.float().norm().item())
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0 / steps
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        ok = torch.isfinite(torch.tensor(last_norm)).item() and last_norm > 0
        return SmokeResult(
            name=name,
            ok=bool(ok),
            forward_ms=elapsed_ms,
            peak_vram_mb=peak_mb,
            out_norm=last_norm,
            n_params_m=n_params_m,
        )
    except Exception as exc:  # noqa: BLE001 — smoke harness must never crash the loop
        return SmokeResult(
            name=name,
            ok=False,
            forward_ms=0.0,
            peak_vram_mb=0.0,
            out_norm=0.0,
            n_params_m=0.0,
            error=f"{type(exc).__name__}: {exc}",
        )


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for hamlet_gpu_smoke")

    # Nano-ish hidden size stand-in; real Qwen3-VL-2B/8B dims differ — this
    # validates kernels/shapes on A40 without loading the full MoT.
    dim = 1024
    batch = 4
    action_len = 16

    variants = [
        (
            "baseline_k4_l2_cross",
            HamletConfig(
                enabled=True,
                n_moment_tokens=4,
                memory_window=4,
                memory_num_layers=2,
                num_heads=8,
                mem_cond_type="cross_attn",
            ),
        ),
        (
            "adaln_k4_l2",
            HamletConfig(
                enabled=True,
                n_moment_tokens=4,
                memory_window=4,
                memory_num_layers=2,
                num_heads=8,
                mem_cond_type="adaln",
            ),
        ),
        (
            "wider_k8_l2_cross",
            HamletConfig(
                enabled=True,
                n_moment_tokens=4,
                memory_window=8,
                memory_num_layers=2,
                num_heads=8,
                mem_cond_type="cross_attn",
            ),
        ),
        (
            "deeper_k4_l4_cross",
            HamletConfig(
                enabled=True,
                n_moment_tokens=4,
                memory_window=4,
                memory_num_layers=4,
                num_heads=8,
                mem_cond_type="cross_attn",
            ),
        ),
        (
            "more_tokens_nq8_k4",
            HamletConfig(
                enabled=True,
                n_moment_tokens=8,
                memory_window=4,
                memory_num_layers=2,
                num_heads=8,
                mem_cond_type="cross_attn",
            ),
        ),
    ]

    results: list[SmokeResult] = []
    for name, cfg in variants:
        print(f"running {name} ...", flush=True)
        results.append(_run_one(name, dim=dim, batch=batch, action_len=action_len, cfg=cfg))

    print("---")
    all_ok = all(r.ok for r in results)
    print(f"all_ok:           {int(all_ok)}")
    print(f"num_variants:     {len(results)}")
    print(f"device:           {torch.cuda.get_device_name(0)}")
    for r in results:
        print(
            f"variant[{r.name}]: ok={int(r.ok)} forward_ms={r.forward_ms:.3f} "
            f"peak_vram_mb={r.peak_vram_mb:.1f} out_norm={r.out_norm:.4f} "
            f"params_M={r.n_params_m:.3f} error={r.error!r}"
        )
    # Autoresearch primary smoke metric: 1.0 if every variant finished finite.
    print(f"val_metric:       {1.0 if all_ok else 0.0:.6f}")
    peak = max((r.peak_vram_mb for r in results), default=0.0)
    print(f"peak_vram_mb:     {peak:.1f}")
    print("---")
    # JSON-ish dump for debugging
    for r in results:
        print("RESULT_JSON:", asdict(r))


if __name__ == "__main__":
    main()
