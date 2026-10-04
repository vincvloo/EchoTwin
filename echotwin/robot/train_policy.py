"""Train the chunk policy on demonstrations (torch). Run it with the perception environment (PERCEPTION_PY): it has torch and a GPU.

    python -m echotwin.robot.train_policy data/policy/sim_demos [more folders] --out data/policy/act_lite.npz

Reads the `shard_*.npz` files of `demos.py` (numpy only, no MuJoCo), makes for every time step the target "the next K actions of
this episode" (the last action repeated past the end), and trains a small MLP with an L1 loss, as ACT does. The weights are saved as plain
numpy arrays for `policy.ChunkPolicy`, which runs without torch.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

OBS_DIM, ACT_DIM = 18, 5


def load(folders, human_weight: float = 1.0):
    """(obs, act, episode id, sample weight) over all shards; episode ids are made unique across folders."""
    O, A, E, W, base = [], [], [], [], 0
    for folder in folders:
        for p in sorted(Path(folder).glob("shard_*.npz")):
            z = np.load(p)
            O.append(z["obs"].astype(np.float32))
            A.append(z["act"].astype(np.float32))
            ep = z["episode"].astype(np.int64)
            src = z["source"]
            human = {"human", "video", "correction"}
            per_ep = np.array([human_weight if (str(src[min(e, len(src) - 1)]) in human) else 1.0 for e in ep])   # "sim" shards carry one source
            E.append(ep + base)
            W.append(per_ep)
            base += int(ep.max()) + 1
    if not O:
        raise SystemExit("no shard_*.npz found: run echotwin.robot.demos first")
    return np.concatenate(O), np.concatenate(A), np.concatenate(E), np.concatenate(W).astype(np.float32)


def chunk_targets(act: np.ndarray, ep: np.ndarray, K: int) -> np.ndarray:
    """(N, K, 5): the next K actions of the same episode; past its end the last action is repeated."""
    N = len(act)
    idx = np.arange(N)[:, None] + np.arange(K)[None, :]
    last = np.zeros(N, np.int64)
    ends = np.r_[np.nonzero(np.diff(ep))[0], N - 1]
    starts = np.r_[0, ends[:-1] + 1]
    for s, e in zip(starts, ends):
        last[s:e + 1] = e
    idx = np.minimum(idx, last[:, None])
    return act[idx]


def train(folders, out: str, K: int = 10, hidden: int = 256, layers: int = 3, epochs: int = 60, batch: int = 1024, lr: float = 1e-3,
          human_weight: float = 1.0, seed: int = 0, log=print) -> dict:
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs, act, ep, w = load(folders, human_weight)
    Y = chunk_targets(act, ep, K)
    eps = np.unique(ep)
    val_eps = set(rng.choice(eps, max(1, len(eps) // 10), replace=False).tolist()) if len(eps) > 1 else set()
    is_val = np.array([e in val_eps for e in ep])
    om, os_ = obs[~is_val].mean(0), obs[~is_val].std(0) + 1e-6
    am, as_ = act[~is_val].mean(0), act[~is_val].std(0) + 1e-6
    X = torch.tensor((obs - om) / os_)
    T = torch.tensor(((Y - am) / as_).reshape(len(Y), -1))
    Wt = torch.tensor(w)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dims = [OBS_DIM] + [hidden] * layers + [K * ACT_DIM]
    net = torch.nn.Sequential(*[m for i in range(len(dims) - 1) for m in
                                (torch.nn.Linear(dims[i], dims[i + 1]),) + ((torch.nn.ReLU(),) if i < len(dims) - 2 else ())]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tr, va = np.nonzero(~is_val)[0], np.nonzero(is_val)[0]
    Xd, Td, Wd = X.to(dev), T.to(dev), Wt.to(dev)
    t0, hist = time.time(), []
    for e in range(epochs):
        net.train()
        perm = torch.tensor(rng.permutation(tr), device=dev)
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            loss = ((net(Xd[b]) - Td[b]).abs().mean(1) * Wd[b]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            v = float((net(Xd[va]) - Td[va]).abs().mean()) if len(va) else float("nan")
            t_ = float((net(Xd[tr]) - Td[tr]).abs().mean())
        hist.append((t_, v))
        if e % 10 == 0 or e == epochs - 1:
            log(f"  epoch {e:3d}  train L1 {t_:.4f}  val L1 {v:.4f}  ({time.time() - t0:.0f} s, {dev})")
    params = {"K": np.array(K), "obs_mean": om, "obs_std": os_, "act_mean": am, "act_std": as_}
    lin = [m for m in net if isinstance(m, torch.nn.Linear)]
    for i, m in enumerate(lin):
        params[f"W{i}"] = m.weight.detach().cpu().numpy().T.copy()
        params[f"b{i}"] = m.bias.detach().cpu().numpy().copy()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **params)
    metrics = {"episodes": int(len(eps)), "steps": int(len(obs)), "val_episodes": len(val_eps), "K": K, "epochs": epochs,
               "train_l1": hist[-1][0], "val_l1": hist[-1][1], "seconds": round(time.time() - t0), "device": dev}
    Path(out).with_suffix(".json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    return metrics


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--out", default="data/policy/act_lite.npz")
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--human-weight", type=float, default=1.0, help="count human / video demos this many times as much")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    print(train(a.folders, a.out, a.chunk, a.hidden, epochs=a.epochs, human_weight=a.human_weight, seed=a.seed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
