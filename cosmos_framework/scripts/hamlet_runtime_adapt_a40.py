#!/usr/bin/env python3
"""A40 smoke: FailureMomentBuffer + critic MPC effect sizes (no full SFT).

Measures:
  1) TinyDenseCritic val MAE (S1b)
  2) Stage2 write rate on fall episodes
  3) Stage3 MPC vs random (distance + myopic) over N seeds
  4) HamletMemory action-feature L2 shift when conditioned on failure buffer

Run on RunPod:
  cd /workspace/cosmos-framework-hamlet
  PYTHONPATH=. .venv/bin/python scripts/hamlet_runtime_adapt_a40.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

# --- minimal env (inline so script is self-contained on the pod) ---
PHASE_LEN = 20
OBJ = np.array([0.25, 0.0, 0.05])
GOAL = np.array([0.45, 0.15, 0.05])
ABOVE_GOAL = np.array([0.45, 0.15, 0.20])
HOME = np.array([0.0, 0.0, 0.3])
ABOVE = np.array([0.25, 0.0, 0.20])
MODES = ("success", "collision", "miss", "fall", "smooth", "recover")
FEAT_PER = 11


def _lin(a, b, n):
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) * a + t * b


def success_ep():
    xyz = np.concatenate(
        [
            _lin(HOME, OBJ, PHASE_LEN),
            np.repeat(OBJ[None], PHASE_LEN, 0),
            _lin(OBJ, ABOVE, PHASE_LEN),
            _lin(ABOVE, ABOVE_GOAL, PHASE_LEN),
            _lin(ABOVE_GOAL, GOAL, PHASE_LEN),
        ]
    )
    g = np.ones(len(xyz))
    g[PHASE_LEN : 5 * PHASE_LEN - 3] = 0.0
    g[5 * PHASE_LEN - 3 :] = 1.0
    return xyz, g, "success"


def perturb(xyz, g, mode, rng):
    xyz, g = xyz.copy(), g.copy()
    T = len(xyz)
    i_grasp, i_lift, i_move, i_place = [k * PHASE_LEN for k in range(1, 5)]
    if mode == "collision":
        hit = i_move + PHASE_LEN // 2
        xyz[hit:] = xyz[hit] + rng.normal(0, 0.01, (T - hit, 3))
        xyz[hit:, 2] = np.maximum(xyz[hit:, 2] - 0.02, 0.02)
    elif mode == "miss":
        xyz[i_grasp:i_lift] += np.array([0.06, 0.04, 0.0])
        g[i_grasp:] = 1.0
        xyz[i_lift:] = xyz[i_grasp] + rng.normal(0, 0.005, (T - i_lift, 3))
    elif mode == "fall":
        drop = i_move + PHASE_LEN // 3
        g[drop:] = 1.0
        xyz[drop:, 2] = np.linspace(xyz[drop, 2], 0.05, T - drop)
        xyz[drop:, :2] += rng.normal(0, 0.01, (T - drop, 2))
    elif mode == "smooth":
        xyz[i_move:] += rng.normal(0, 0.02, xyz[i_move:].shape)
        g[i_place:] = 0.0
    elif mode == "recover":
        xyz[i_grasp:i_lift] += np.array([0.06, 0.04, 0.0])
        g[i_grasp:i_lift] = 1.0
        splice = i_lift + PHASE_LEN // 2
        sx, sg, _ = success_ep()
        xyz[splice:], g[splice:] = sx[splice:], sg[splice:]
        xyz[splice : splice + 3] = OBJ
        g[splice : splice + 3] = 0.0
    return xyz, g, mode


def progress_reward(xyz, g, mode):
    out = np.zeros(len(xyz))
    best, grasped, dropped = 0.0, False, False
    for t, (p, gv) in enumerate(zip(xyz, g)):
        d_obj = float(np.linalg.norm(p - OBJ))
        d_goal = float(np.linalg.norm(p - GOAL))
        d_ag = float(np.linalg.norm(p - ABOVE_GOAL))
        closed = gv < 0.5
        if closed and d_obj < 0.05:
            grasped = True
        if grasped and (not closed) and d_goal >= 0.06 and p[2] > 0.08:
            dropped = True
        score = 0.20 * float(np.clip(1.0 - d_obj / 0.30, 0, 1))
        if grasped:
            score = max(score, 0.35)
            score = max(
                score,
                0.35
                + 0.20 * float(np.clip((p[2] - 0.05) / 0.15, 0, 1))
                + 0.25 * float(np.clip(1.0 - d_ag / 0.35, 0, 1)),
            )
        if grasped and d_goal < 0.06 and closed:
            score = max(score, 0.90)
        if grasped and d_goal < 0.06 and not closed:
            score = 1.0
        if dropped:
            score = min(score, best * 0.55)
            best = 0.85 * best + 0.15 * score
        else:
            if (not grasped) and t >= 2 * PHASE_LEN:
                score = min(score, 0.25)
            if mode == "collision" and t >= 3 * PHASE_LEN + PHASE_LEN // 2:
                score = min(score, 0.55)
            best = max(best, score)
        out[t] = best
    return np.clip(out, 0, 1)


def frame_feat(xyz, g, t, history=2):
    feats = []
    for k in range(history - 1, -1, -1):
        i = max(0, t - k)
        p = xyz[i]
        gv = float(g[i])
        feats.append(np.concatenate([p, [gv], p - OBJ, p - GOAL, [1.0 if gv < 0.5 else 0.0]]))
    return np.concatenate(feats).astype(np.float32)


def build_xy(n_per=12, history=2, seed=0):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for mode in MODES:
        for _ in range(n_per):
            xyz, g, m = success_ep()
            xyz, g, m = perturb(xyz, g, mode, np.random.default_rng(int(rng.integers(0, 1e9))))
            r = progress_reward(xyz, g, m)
            for t in range(len(xyz)):
                xs.append(frame_feat(xyz, g, t, history))
                ys.append(float(r[t]))
    return np.stack(xs), np.asarray(ys, np.float32)


class TinyDenseCritic(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x).squeeze(-1))


def train_critic(device, seed=0):
    torch.manual_seed(seed)
    x, y = build_xy(seed=seed)
    n = len(y)
    perm = np.random.default_rng(seed).permutation(n)
    n_val = n // 5
    val_i, tr_i = perm[:n_val], perm[n_val:]
    model = TinyDenseCritic(2 * FEAT_PER).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(40):
        for s in range(0, len(tr_i), 256):
            sl = tr_i[s : s + 256]
            xb = torch.from_numpy(x[sl]).to(device)
            yb = torch.from_numpy(y[sl]).to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        va = model(torch.from_numpy(x[val_i]).to(device))
        yt = torch.from_numpy(y[val_i]).to(device)
        mae = float((va - yt).abs().mean())
        fail = yt < 0.85
        fail_mae = float((va[fail] - yt[fail]).abs().mean()) if fail.any() else mae
    return model, {"val_mae": mae, "fail_mae": fail_mae}


def apply_action(xyz, grip, action):
    nxt = xyz + action[:3]
    nxt[2] = max(0.02, float(nxt[2]))
    return nxt, float(np.clip(grip + float(action[3]), 0, 1))


def mpc_one(model, device, seed):
    rng = np.random.default_rng(seed)
    xyz, g, m = success_ep()
    xyz, g, m = perturb(xyz, g, "fall", rng)
    r = progress_reward(xyz, g, m)
    dr = np.diff(r, prepend=r[0])
    drop_t = int(np.clip(np.argmin(dr), PHASE_LEN * 3, len(xyz) - 2))
    K = 16
    cands = rng.normal(0, 0.015, (K, 4)).astype(np.float32)
    for i in range(4):
        cands[i, :3] = (0.25 + 0.2 * i) * (OBJ - xyz[drop_t])
        cands[i, 3] = -0.6
    scores = []
    with torch.no_grad():
        for k in range(K):
            xyz2, g2 = apply_action(xyz[drop_t], float(g[drop_t]), cands[k])
            xyz_t, g_t = xyz.copy(), g.copy()
            xyz_t[drop_t + 1], g_t[drop_t + 1] = xyz2, g2
            feat = frame_feat(xyz_t, g_t, drop_t + 1)
            scores.append(float(model(torch.from_numpy(feat).unsqueeze(0).to(device))[0]))
    best = int(np.argmax(scores))
    rand = int(rng.integers(4, K))

    def metr(a):
        p2, g2 = apply_action(xyz[drop_t], float(g[drop_t]), a)
        dist = float(np.linalg.norm(p2 - OBJ))
        myopic = float(np.clip(1.0 - dist / 0.35, 0, 1)) * 0.7 + (0.3 if g2 < 0.5 else 0.0)
        return dist, myopic

    d_m, m_m = metr(cands[best])
    d_r, m_r = metr(cands[rand])
    return {
        "dist_mpc": d_m,
        "dist_rand": d_r,
        "myopic_mpc": m_m,
        "myopic_rand": m_r,
        "closer": float(d_m <= d_r),
        "myopic_win": float(m_m >= m_r),
        "writes": float(np.sum(dr < -0.02)),
    }


def hamlet_buffer_effect():
    from cosmos_framework.model.generator.mot.hamlet_memory import (
        FailureMomentBuffer,
        HamletConfig,
        HamletMemory,
    )

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hamlet = HamletMemory(
        dim=4096 if device.type == "cuda" else 64,
        config=HamletConfig(
            enabled=True,
            n_moment_tokens=4,
            memory_window=4,
            memory_num_layers=2,
            num_heads=8,
            memory_dim=512,
            mem_cond_type="adaln",
        ),
    ).to(device)
    hamlet.eval()
    buf = FailureMomentBuffer.from_hamlet(hamlet, delta_r_threshold=-0.01)
    # write 4 failure moments
    for i in range(4):
        buf.maybe_write(torch.randn(4, hamlet.outer_dim, device="cpu"), r_t=0.2 - 0.05 * i, r_prev=0.6)
    action = torch.randn(2, 16, hamlet.outer_dim, device=device)
    with torch.no_grad():
        out_fail = hamlet.condition_with_failure_buffer(action, buf)
        buf.clear()
        out_empty = hamlet.condition_with_failure_buffer(action, buf)
    delta = float((out_fail - out_empty).norm(dim=-1).mean())
    rel = float((out_fail - action).norm() / (action.norm() + 1e-6))
    return {
        "action_l2_fail_vs_empty": delta,
        "action_rel_shift_fail": rel,
        "outer_dim": float(hamlet.outer_dim),
        "mem_dim": float(hamlet.mem_dim),
        "n_params_m": sum(p.numel() for p in hamlet.parameters()) / 1e6,
        "device": str(device),
    }


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    model, crit = train_critic(device)
    print("critic", crit, flush=True)

    seeds = list(range(20))
    rows = [mpc_one(model, device, s) for s in seeds]
    closer = float(np.mean([r["closer"] for r in rows]))
    myopic_win = float(np.mean([r["myopic_win"] for r in rows]))
    dist_mpc = float(np.mean([r["dist_mpc"] for r in rows]))
    dist_rand = float(np.mean([r["dist_rand"] for r in rows]))
    myopic_mpc = float(np.mean([r["myopic_mpc"] for r in rows]))
    myopic_rand = float(np.mean([r["myopic_rand"] for r in rows]))
    writes = float(np.mean([r["writes"] for r in rows]))
    mpc = {
        "n_seeds": len(seeds),
        "mpc_closer_rate": closer,
        "mpc_myopic_win_rate": myopic_win,
        "mean_dist_mpc": dist_mpc,
        "mean_dist_rand": dist_rand,
        "mean_myopic_mpc": myopic_mpc,
        "mean_myopic_rand": myopic_rand,
        "mean_failure_writes": writes,
        "effect_dist_delta": dist_rand - dist_mpc,
        "effect_myopic_delta": myopic_mpc - myopic_rand,
    }
    print("mpc", mpc, flush=True)

    ham = hamlet_buffer_effect()
    print("hamlet_buffer", ham, flush=True)

    out = {
        "commit": "57ace63",
        "elapsed_s": time.time() - t0,
        "critic": crit,
        "mpc": mpc,
        "hamlet_buffer": ham,
        "verdict": {
            "critic_ok": crit["val_mae"] < 0.05,
            "mpc_effect": closer >= 0.6 and (dist_rand - dist_mpc) > 0.01,
            "buffer_shifts_actions": ham["action_l2_fail_vs_empty"] > 1e-3,
        },
    }
    out_path = Path("/workspace/outputs/hamlet_runtime_adapt_a40.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print("wrote", out_path, flush=True)
    print("VERDICT", out["verdict"], flush=True)
    # non-zero exit if no effect
    ok = all(out["verdict"].values())
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
