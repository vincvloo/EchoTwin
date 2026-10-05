"""How well do the default skills work in the simulation? Run each kind of move many times and count.

    python -m echotwin.robot.skillcheck                 # 6 trials per shape and task, writes out/skillcheck.json
    python -m echotwin.robot.skillcheck --trials 12 --shapes box cylinder

One trial: a table with the object to move (one of the four shapes, real size) and a small box, the object's
position drawn at random inside what the arm can reach, then the whole move with the default skill (grip height,
lift, drop, speed from `prop_skills.DEFAULTS`), judged like a real run: does it land where it should, upright,
without knocking anything over. The failure reason is the first thing that went wrong.

It only uses the public pieces (World, prop_skills, move_things), so the same file runs on the older kinematic
simulation and on the arm with a contact grasp: that is how the two were compared (docs/RESULTS.md).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .features import move_things as MT
from .features.measure import measure
from .features import prop_skills as PS
from .scene import Layout
from .world import CTRL_DT, World

# real sizes, full (x, y, z) in metres
SHAPES = {"flat": (0.10, 0.06, 0.02), "box": (0.05, 0.04, 0.04), "cylinder": (0.07, 0.07, 0.10), "round": (0.05, 0.05, 0.05)}
OTHER = (0.05, 0.05, 0.05)
TASKS = ("next to", "to the left", "on top of")
RGB = {"flat": (0.9, 0.9, 0.85), "box": (0.2, 0.4, 0.8), "cylinder": (0.8, 0.8, 0.8), "round": (0.2, 0.2, 0.2)}
MAX_TICKS = 1200


def _scale() -> float:
    """1.0 in the real-scale simulation. The older one drew everything twice as big."""
    return 1.0 if hasattr(Layout(), "table_half") else 2.0


def _prop(name, shape, size, pos, k):
    return {"name": name, "shape": shape, "pos": (pos[0] * k, pos[1] * k), "yaw": 0.0,
            "size": tuple(v * k / 2 for v in size), "rgb": RGB[shape]}


def _spawn(rng, w_probe: World | None, k: float):
    """Two spots, at least 12 cm apart, inside what the arm can reach."""
    for _ in range(500):
        pts = []
        for _ in range(2):
            if w_probe is not None and hasattr(w_probe, "workspace_sample"):
                pts.append(w_probe.workspace_sample(rng))
            else:
                pts.append((rng.uniform(-0.18, 0.18), rng.uniform(-0.10, 0.12)))
        if np.hypot(pts[0][0] - pts[1][0], pts[0][1] - pts[1][1]) > 0.12:
            return pts
    raise RuntimeError("no two spots found")


def make_task(w: World, plan: dict, heard: str) -> dict:
    """The same task description the robot makes for a spoken move (see Sim._make_prop_task)."""
    me = "prop_0"
    pr = w.layout.props[0]
    goal = MT.goal_xy(w, plan)
    o = w.obj_pos(me)
    shape = pr.get("shape", "box")
    task = {"kind": "prop", "plan": plan, "object": me, "target": f"{shape} things", "shape": shape, "name": pr["name"],
            "instruction": heard, "goal": [float(goal[0]), float(goal[1])], "ref": "prop_1" if plan["goal"][0] == "near" else None,
            "m": measure(w, me),
            "h": float(w.half(me)), "tallest": float(w.tallest() if hasattr(w, "tallest") else max(2 * w.half(n) for n in w.things())),
            "start": [float(o[0]), float(o[1])]}
    if plan["goal"][0] == "near" and plan["goal"][2] == "on top of":
        task["stack"] = True
        task["ref_name"] = w.layout.props[1]["name"]
    return task


def trial(shape: str, task_name: str, rng: np.random.Generator, size=None, skill_fn=None, backend: str = "sim",
          loop: bool = False, disturb: str | None = None, policy=None, align: bool | None = None, encoder_offset: float | None = None) -> dict:
    """One move. `size` overrides the shape's size; `skill_fn(task) -> style` replaces the default skill;
    `backend` is sim (the simulation is the robot) or mock (the mock arm: a perturbed second world, judged by its own objects).
    `loop`: look before gripping, check, retry (prop_skills.loop_start). `disturb`: push the object 3 to 6 cm, either
    "before" the move starts (after it was planned) or "during" it (just as the jaws are about to close)."""
    k = _scale()
    probe = None
    try:
        probe = World(Layout())
    except Exception:
        pass
    for _ in range(60):             # draw until the arm can reach both the object and the spot (else it would decline)
        (ax, ay), (bx, by) = _spawn(rng, probe, k)
        lay = Layout()
        lay.props = [_prop("mover", shape, size or SHAPES[shape], (ax, ay), k), _prop("other", "box", OTHER, (bx, by), k)]
        w = World(lay)
        w.settle(20)
        plan = {"prop": 0, "goal": ("near", 1, task_name) if task_name != "to the left" else ("dir", (-1, 0), 0.12 * k)}
        if not hasattr(w, "refusal"):
            break
        if w.can_grasp("prop_0")[0] is False:
            return {"ok": False, "why": "refused: too wide for the gripper", "seconds": 0.0}
        goal = MT.goal_xy(w, plan)
        if not w.refusal("prop_0", goal):
            break
    if backend == "mock":                       # the same move, on an arm that is not the simulation
        from .drivers import MockDriver
        from .real import RealBackend, mock_camera
        drv = MockDriver(w.layout, w.arm, **({"encoder_offset": encoder_offset} if encoder_offset is not None else {}))
        w = RealBackend(w, drv, camera=mock_camera(drv))
        w.settle(10)
    truth = w.driver.plant if backend == "mock" else w     # what really happened: judged here, not on what the robot believes
    task = make_task(w, plan, f"put the mover {task_name}")
    style = skill_fn(task) if skill_fn else dict(PS.DEFAULTS)
    before = {n: truth.obj_pos(n)[:2].copy() for n in truth.things()}
    push = None
    if disturb:                                            # somebody moves it by 3 to 6 cm, towards the middle of the table
        ang, mag = rng.uniform(0, 2 * np.pi), rng.uniform(0.03, 0.06)
        push = (mag * np.cos(ang), mag * np.sin(ang))
    if disturb == "before":
        truth.nudge("prop_0", push)
    if loop:
        wps, lp = PS.loop_start(w, task, style, align=align)
    else:
        wps, lp = PS.waypoints(w, task, style), None
    r = {"wps": wps, "i": 0, "speed": style["speed"], "loop": lp}
    if policy is not None:                                  # a learned policy instead of the scripted steps
        from .policy import LearnedExecutor
        r["policy"] = LearnedExecutor(policy, task)
    if hasattr(w, "grasp_yaw"):
        r["yaw"] = w.grasp_yaw(task["object"])
    ticks = 0
    limit = MAX_TICKS * (2 if (backend == "mock" or loop) else 1)       # a real arm moves at half speed; looking takes time
    pushed = disturb != "during"
    while ticks < limit:
        if not pushed and r["i"] < len(r["wps"]) and r["wps"][r["i"]] == ("grip", 1.0):
            truth.nudge("prop_0", push)
            pushed = True
        a = PS.waypoint_action(w, r)
        if a is None:
            break
        w.step(a)
        ticks += 1
    w.settle(25)
    res = PS.outcome(truth, task, before)
    ok = bool(res["ok"] and PS.goal_met(truth, task))
    why = "" if ok else (res["text"] if not res["ok"] else "it did not end where it should")
    if ticks >= limit:
        why, ok = "ran out of time", False
    return {"ok": ok, "why": why, "seconds": ticks * CTRL_DT, "style": style, "task": task,
            "attempts": (r.get("loop") or {}).get("attempts"), "log": (r.get("loop") or {}).get("log")}


def reason_key(why: str) -> str:
    """Group the long sentences into a few causes."""
    for key in ("tips over", "bumps", "slides off", "lands", "refused", "ran out of time", "did not end"):
        if key in why:
            return key
    return why[:40] or "ok"


def run(trials: int, shapes, tasks, seed: int = 7, backend: str = "sim", loop: bool = False, disturb: str | None = None, policy=None,
        align: bool | None = None, encoder_offset: float | None = None) -> dict:
    rng = np.random.default_rng(seed)
    out = {"scale": _scale(), "trials": trials, "backend": backend, "loop": loop, "disturb": disturb, "cells": {}}
    for s in shapes:
        for t in tasks:
            t0 = time.time()
            rs = [trial(s, t, rng, backend=backend, loop=loop, disturb=disturb, policy=policy, align=align, encoder_offset=encoder_offset)
                  for _ in range(trials)]
            wins = [r for r in rs if r["ok"]]
            out["cells"][f"{s} | {t}"] = {
                "success": len(wins) / trials, "seconds": float(np.mean([r["seconds"] for r in wins])) if wins else None,
                "failures": dict(Counter(reason_key(r["why"]) for r in rs if not r["ok"])), "wall": round(time.time() - t0, 1)}
            c = out["cells"][f"{s} | {t}"]
            print(f"  {s:9s} {t:12s} {c['success']:4.0%}  {c['failures'] or ''}", flush=True)
    return out


def table(res: dict) -> str:
    shapes = sorted({k.split(" | ")[0] for k in res["cells"]}, key=list(SHAPES).index)
    tasks = [t for t in TASKS if any(k.endswith(" | " + t) for k in res["cells"])]
    lines = ["| Shape | " + " | ".join(tasks) + " |", "|---|" + "---|" * len(tasks)]
    for s in shapes:
        cells = []
        for t in tasks:
            c = res["cells"].get(f"{s} | {t}")
            cells.append(f"{c['success']:.0%}" if c else "-")
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    ok = np.mean([c["success"] for c in res["cells"].values()])
    lines.append("| **all** | " + " | ".join([""] * (len(tasks) - 1) + [f"**{ok:.0%}**"]) + " |")
    return "\n".join(lines)


# ---------------- do skills carry over to other sizes? ----------------
HARD = False        # --hard: only tall, wide-ish things, where the default skill fails most


def random_object(rng) -> tuple[str, tuple]:
    """A random everyday-sized object, full size (x, y, z) in metres: width 3 to 7 cm, height 2 to 12 cm."""
    shape = str(rng.choice(["box", "cylinder"] if HARD else ["box", "cylinder", "round"]))
    wd = float(rng.uniform(0.055, 0.07) if HARD else rng.uniform(0.03, 0.07))
    if shape == "round":
        return shape, (wd, wd, wd)
    h = float(rng.uniform(0.08, 0.13) if HARD else rng.uniform(0.02, 0.12))
    return shape, ((wd, wd * float(rng.uniform(1.0, 1.4)), h) if shape == "box" else (wd, wd, h))


def random_style(rng) -> dict:
    return {"grip": float(rng.uniform(-0.5, 1.0)), "lift": float(rng.uniform(0.03, 0.09)),
            "drop": float(rng.uniform(0.002, 0.012)), "speed": float(rng.uniform(0.14, 0.25))}


def _median_style(demos: list[dict]) -> dict:
    if not demos:
        return dict(PS.DEFAULTS)
    return {k: float(np.median([d[k] for d in demos])) for k in ("grip", "lift", "drop", "speed")}


def transfer(n_objects: int = 8, tries: int = 6, seed: int = 11) -> dict:
    """Learn from practice on some objects, then move seen-size and unseen-size objects with four ways of choosing
    the skill: the defaults, the old per-shape key, per size class, and from measurements (nearest sizes)."""
    from .features import measure as M
    rng = np.random.default_rng(seed)
    objs = [random_object(rng) for _ in range(n_objects)]
    probe = World(Layout())
    objs = [o for o in objs if _can(o, probe)]
    demos = []                                           # practice: varied styles, keep the ones that worked
    for i, (shape, size) in enumerate(objs):
        for j in range(tries):
            r = trial(shape, TASKS[(i + j) % 3], np.random.default_rng(seed * 100 + i * 10 + j), size, lambda t, s=random_style(rng): s)
            if r["ok"]:
                t = r["task"]
                demos.append({**r["style"], "dist": float(np.linalg.norm(np.array(t["goal"]) - np.array(t["start"]))),
                              "m": t["m"], "shape": shape, "cls": M.size_class(t["m"])})
    skills = PS.PropSkills()
    skills.demos = demos
    ways = {
        "defaults": lambda t: dict(PS.DEFAULTS),
        "per shape (old)": lambda t: _median_style([d for d in demos if d["shape"] == t["shape"]]),
        "per size class": lambda t: _median_style([d for d in demos if d["cls"] == M.size_class(t["m"])]),
        "from measurements": lambda t: {k: v for k, v in skills.plan(t["m"], 0.2).items() if k in PS.DEFAULTS},
    }
    unseen = []
    while len(unseen) < n_objects * 2:
        o = random_object(rng)
        if _can(o, probe):
            unseen.append(o)
    out = {"objects": len(objs), "demos": len(demos), "rows": {}}
    for label, group in (("seen sizes", objs), ("unseen sizes", unseen)):
        for way, fn in ways.items():
            rs = [trial(s, tk, np.random.default_rng(seed * 7 + i * 3 + j), z, fn)
                  for i, (s, z) in enumerate(group) for j, tk in enumerate(TASKS)]
            asks = sum(not skills.known(r["task"]["m"]) for r in rs if "task" in r)
            out["rows"][f"{label} | {way}"] = {"success": float(np.mean([r["ok"] for r in rs])), "n": len(rs), "ask": asks}
            print(f"  {label:13s} {way:18s} {out['rows'][f'{label} | {way}']['success']:4.0%} of {len(rs)}  (would ask {asks})", flush=True)
    return out


def _can(obj, probe: World) -> bool:
    shape, size = obj
    lay = Layout()
    lay.props = [_prop("x", shape, size, (0.0, -0.05), 1.0)]
    return World(lay).can_grasp("prop_0")[0]


def gap_table(sim: dict, mock: dict, labels=("sim", "mock arm")) -> str:
    """Success per cell for two runs, side by side."""
    lines = [f"| Shape | Task | {labels[0]} | {labels[1]} | gap |", "|---|---|---|---|---|"]
    for k, a in sim["cells"].items():
        b = mock["cells"][k]
        s, t = k.split(" | ")
        lines.append(f"| {s} | {t} | {a['success']:.0%} | {b['success']:.0%} | {(b['success'] - a['success']) * 100:+.0f} |")
    ma, mb = (np.mean([c["success"] for c in r["cells"].values()]) for r in (sim, mock))
    lines.append(f"| **all** | | **{ma:.0%}** | **{mb:.0%}** | **{(mb - ma) * 100:+.0f}** |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--shapes", nargs="+", default=list(SHAPES), choices=list(SHAPES))
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="out/skillcheck.json")
    ap.add_argument("--backend", choices=["sim", "mock", "both"], default="sim",
                    help="mock: the mock arm (lag, encoder offsets, heavier objects); both: run each and show the gap")
    ap.add_argument("--align", choices=["on", "off"], default=None, help="with --loop: look at the gripper and line it up before going down (default: auto)")
    ap.add_argument("--ab-align", action="store_true", help="run the closed loop without and with the alignment and show both (mock arm)")
    ap.add_argument("--encoder-offset", type=float, default=None, help="mock arm: spread of the encoder error per joint, rad (default 0.012)")
    ap.add_argument("--policy", metavar="NPZ", help="move with this trained policy (train_policy.py) instead of the scripted skill")
    ap.add_argument("--loop", action="store_true", help="look before gripping, check the grasp and the goal, retry (PR12)")
    ap.add_argument("--ab", action="store_true", help="run open loop and closed loop on the chosen backend and show both")
    ap.add_argument("--disturb", choices=["before", "during"], default=None,
                    help="push the object 3 to 6 cm before the move starts, or just as the jaws are about to close")
    ap.add_argument("--transfer", action="store_true", help="do skills learned on some sizes work on other sizes?")
    ap.add_argument("--hard", action="store_true", help="with --transfer: only tall, wide things")
    ap.add_argument("--objects", type=int, default=8, help="with --transfer: objects to practise on")
    a = ap.parse_args(argv)
    if a.transfer:
        global HARD
        HARD = a.hard
        res = transfer(a.objects, seed=a.seed)
        Path("out").mkdir(exist_ok=True)
        Path("out/skillcheck_transfer.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        return 0
    if a.backend == "both":
        res = {b: run(a.trials, a.shapes, a.tasks, a.seed, b) for b in ("sim", "mock")}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path("out/skillcheck_backends.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print("\n" + gap_table(res["sim"], res["mock"]))
        return 0
    if a.ab_align:
        res = {m: run(a.trials, a.shapes, a.tasks, a.seed, "mock", loop=True, disturb=a.disturb, align=(m == "aligned"),
                      encoder_offset=a.encoder_offset) for m in ("plain", "aligned")}
        Path("out/skillcheck_align.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(chr(10) + gap_table(res["plain"], res["aligned"], ("closed loop", "+ align")))
        return 0
    if a.ab:
        res = {m: run(a.trials, a.shapes, a.tasks, a.seed, a.backend, loop=(m == "closed"), disturb=a.disturb) for m in ("open", "closed")}
        Path("out/skillcheck_loop.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print("\n" + gap_table(res["open"], res["closed"], ("open loop", "closed loop")))
        return 0
    pol = None
    if a.policy:
        from .policy import ChunkPolicy
        pol = ChunkPolicy.load(a.policy)
    res = run(a.trials, a.shapes, a.tasks, a.seed, a.backend, loop=a.loop, disturb=a.disturb, policy=pol,
              align=None if a.align is None else a.align == "on", encoder_offset=a.encoder_offset)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("\n" + table(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
