"""Demonstrations for a learned policy: made in bulk by the scripted skill, or converted from the ones you recorded.

    python -m echotwin.robot.demos --n 600 --noise 0.3 --out data/policy/sim_demos
    python -m echotwin.robot.demos --episodes data/robot/episodes --out data/policy/human_demos

Each run writes `shard_XXXX.npz` files with, per time step: the observation (policy_obs.py), the action and the episode number.
Only moves that worked are kept. `--noise` makes the expert's *executed* actions noisy while the *recorded* action stays the
clean one (DART): the policy then sees states a little off the expert's path, and learns how to come back to it.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from . import policy_obs as O
from .features import move_things as MT
from .features import prop_skills as PS
from .scene import Layout
from .world import World

MAX_TICKS = 1500


def _setup(rng):
    """A random object and spot, a random task, a world with the arm able to do it. Returns (world, task) or None."""
    from . import skillcheck as S
    probe = World(Layout())
    if rng.random() < 0.4:                                   # the named test objects (a 10 x 6 cm bar, a box wider than deep, ...)
        shape = str(rng.choice(list(S.SHAPES)))
        size = S.SHAPES[shape]
    else:
        shape, size = S.random_object(rng)
        if rng.random() < 0.5:                               # turn it a quarter: the jaws must close the other way
            size = (size[1], size[0], size[2])
    task_name = str(rng.choice(S.TASKS))
    for _ in range(60):
        (ax, ay), (bx, by) = S._spawn(rng, probe, 1.0)
        lay = Layout()
        lay.props = [S._prop("mover", shape, size, (ax, ay), 1.0), S._prop("other", "box", S.OTHER, (bx, by), 1.0)]
        w = World(lay)
        w.settle(20)
        plan = {"prop": 0, "goal": ("near", 1, task_name) if task_name != "to the left" else ("dir", (-1, 0), 0.12)}
        if not w.can_grasp("prop_0")[0]:
            return None
        if not w.refusal("prop_0", MT.goal_xy(w, plan)):
            return w, S.make_task(w, plan, f"put the mover {task_name}")
    return None


def one_episode(rng, noise: float = 0.0, style: dict | None = None):
    """Run the scripted skill once. Returns (obs (N, 18), actions (N, 5), ok, task) with the clean expert actions recorded."""
    from . import skillcheck as S
    got = _setup(rng)
    if got is None:
        return None
    w, task = got
    style = style or (dict(PS.DEFAULTS) if rng.random() < 0.5 else S.random_style(rng))
    r = {"wps": PS.waypoints(w, task, style), "i": 0, "speed": style["speed"], "loop": None, "yaw": w.grasp_yaw(task["object"])}
    before = {n: w.obj_pos(n)[:2].copy() for n in w.things()}
    obs, acts = [], []
    for t in range(MAX_TICKS):
        a = PS.waypoint_action(w, r)
        if a is None:
            break
        a = np.asarray(a, float)
        obs.append(O.from_world(w, task, t))
        acts.append(a.astype(np.float32))
        run = a.copy()
        if noise:
            run[:3] += rng.normal(0.0, noise * 0.1, 3)         # m/s: the executed motion is a little off, the label is not
        w.step(run)
    w.settle(25)
    res = PS.outcome(w, task, before)
    ok = bool(res["ok"] and PS.goal_met(w, task)) and len(obs) < MAX_TICKS
    return np.stack(obs), np.stack(acts), ok, task


def collect(n: int, out: Path, noise: float = 0.0, seed: int = 0, shard: int = 100, log=print) -> dict:
    """Make `n` successful episodes in shards of `shard` episodes."""
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    buf, kept, tried, t0, shards = [], 0, 0, time.time(), 0

    def flush():
        nonlocal buf, shards
        if not buf:
            return
        O_, A_, E_ = (np.concatenate([b[i] for b in buf]) for i in range(3))
        np.savez_compressed(out / f"shard_{shards:04d}.npz", obs=O_, act=A_, episode=E_, source=np.array(["sim"]))
        shards += 1
        buf = []
    while kept < n and tried < 6 * n + 20:
        tried += 1
        got = one_episode(rng, noise)
        if got is None or not got[2]:
            continue
        o, a, _, _ = got
        buf.append((o, a, np.full(len(o), kept, np.int32)))
        kept += 1
        if len(buf) >= shard:
            flush()
        if kept % 25 == 0:
            log(f"  {kept}/{n} episodes ({tried} tries, {time.time() - t0:.0f} s)")
    flush()
    return {"episodes": kept, "tried": tried, "seconds": round(time.time() - t0), "noise": noise, "shards": shards}


def convert(episodes_dir: Path, out: Path, only_success: bool = True) -> dict:
    """The JSON demos of `Dataset` (human, video, practice) into one shard, with their source recorded."""
    import json
    out.mkdir(parents=True, exist_ok=True)
    O_, A_, E_, src, legacy_n = [], [], [], [], 0
    for i, p in enumerate(sorted(Path(episodes_dir).glob("ep_*.json"))):
        ep = json.loads(p.read_text(encoding="utf-8"))
        if (only_success and not ep.get("success", True)) or ep.get("task", {}).get("kind") != "prop" or len(ep["frames"]) < 5:
            continue
        o, a, legacy = O.from_episode(ep)
        legacy_n += int(legacy)
        O_.append(o)
        A_.append(a)
        E_.append(np.full(len(o), len(src), np.int32))
        src.append(ep.get("source", "human"))
    if not O_:
        return {"episodes": 0}
    np.savez_compressed(out / "shard_0000.npz", obs=np.concatenate(O_), act=np.concatenate(A_), episode=np.concatenate(E_),
                        source=np.array(src))
    return {"episodes": len(src), "legacy": legacy_n, "sources": {s: src.count(s) for s in sorted(set(src))}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=0, help="episodes to make with the scripted skill")
    ap.add_argument("--noise", type=float, default=0.0, help="noise on the executed motion, 0 to 1 (0.3 = 3 cm/s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", help="convert the JSON demos in this folder instead")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.episodes:
        print(convert(Path(a.episodes), Path(a.out)))
    elif a.n:
        print(collect(a.n, Path(a.out), a.noise, a.seed))
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
