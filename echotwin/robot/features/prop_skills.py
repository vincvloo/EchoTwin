"""Know-how for everyday objects: learned from demos, checked by imagining the move first.

A skill belongs to a *size*, not to a name or a shape: the grip width, height and length of the object (`measure.py`).
From every kept demo we learn, relative to the object:
  grip     how high to grip it (fraction of its half height above/below the centre)
  lift     how much clearance to lift it above the tallest thing on the table
  drop     how gently to set it down (hand height above the resting height at release)
  speed    how fast the demonstrator moved
A new object is planned from the demos of objects of a similar size (the closer, the more they count).
Before acting, the robot imagines the move on a copy of the world and checks the outcome:
lands near the goal, stays upright, does not knock anything over.
"""
import os

import numpy as np

from ..world import VMAX, World
from . import measure as M

MIN_PROP_DEMOS = 1
ASK = 1.0
KNOWN = 1.0                  # a demo this close in size (see measure.distance) counts as "a similar object"
KIND_LABEL = M.CLASS_LABEL   # the dashboard cards: demos per size class
DEFAULTS = {"grip": 1.0, "lift": 0.05, "drop": 0.004, "speed": 0.2}


# ---------------- geometry of success ----------------
def tilt_deg(world: World, name: str, d=None) -> float:
    return world.tilt(name, d)


def goal_met(world: World, task: dict, d=None, hs=None) -> bool:
    """Is the requested relation true right now (object resting, not held)?"""
    hs = hs if hs is not None else world.hand
    me = task["object"]
    if hs.attached == me or hs.grip:
        return False
    if task.get("stack"):
        o, q = world.obj_pos(me, d), world.obj_pos(task["ref"], d)
        on = np.linalg.norm(o[:2] - q[:2]) < world.radius(task["ref"]) * 0.9
        return bool(on and o[2] > q[2] + world.half(task["ref"]) * 0.8)
    p = world.obj_pos(me, d)[:2]
    if task.get("ref"):
        q = world.obj_pos(task["ref"], d)[:2]
        gap = np.linalg.norm(p - q) - world.radius(me) - world.radius(task["ref"])
        return -0.01 < gap < 0.07 and _side_ok(task, p, q)
    return float(np.linalg.norm(p - np.array(task["goal"]))) < 0.06


def _side_ok(task, p, q) -> bool:
    rel = task["plan"]["goal"][2] if task["plan"]["goal"][0] == "near" else "next to"
    d = p - q
    return {"left of": d[0] < 0, "right of": d[0] > 0, "in front of": d[1] < 0, "behind": d[1] > 0}.get(rel, True)


def outcome(world: World, task: dict, before: dict, d=None, hs=None) -> dict:
    me = task["object"]
    err = float(np.linalg.norm(world.obj_pos(me, d)[:2] - np.array(task["goal"])))
    tilt = tilt_deg(world, me, d)
    moved = max([float(np.linalg.norm(world.obj_pos(n, d)[:2] - p)) for n, p in before.items() if n != me
                 and not (task.get("stack") and n == task["ref"] and np.linalg.norm(world.obj_pos(n, d)[:2] - p) < 0.04)]
                or [0.0])
    met = goal_met(world, task, d, hs)
    problems = []
    if tilt > 25:
        problems.append(f"the {task['name']} tips over")
    if moved > 0.03:
        problems.append("it bumps into something")
    if task.get("stack") and not met:
        problems.append(f"it slides off the {task.get('ref_name', 'other object')}")
    elif not met and err > 0.06:
        problems.append(f"it lands {err * 100:.0f} cm off")
    text = ("it lands " + (f"{err * 100:.0f} cm from the spot" if err >= 0.01 else "on the spot") +
            (", stays upright" if tilt <= 25 else "")) if not problems else " and ".join(problems)
    return {"ok": not problems, "err": err, "tilt": tilt, "moved": moved, "text": text}


# ---------------- the plan as waypoints ----------------
def waypoints(world: World, task: dict, skill: dict, d=None, pos=None) -> list:
    """The move as 8 steps. `pos`: where the object is seen to be (x, y), instead of where the robot believes it is."""
    me = task["object"]
    o = world.obj_pos(me, d)
    if pos is not None:
        o = np.array([pos[0], pos[1], o[2]])
    h = world.half(me)
    goal = task["goal"]
    tallest = world.tallest()
    carry_max = world.carry_height()
    carry = float(min(tallest + h + skill["lift"], carry_max))
    off = world.grasp_offset(me)
    # pad tips low on the object; thin things need the tips right at the table
    grasp_z = float(max(o[2] - h + 0.004 + (skill["grip"] + 1.0) * h * 0.3 * min(1.0, h / 0.03), 0.004))
    release_z = float(max(h + skill["drop"] + 0.002, 0.013))
    if task.get("stack"):  # set it down on top of the other object
        ref = task["ref"]
        release_z = float(world.obj_pos(ref, d)[2] + world.half(ref) + h + skill["drop"] + 0.006)
        carry = float(min(max(carry, release_z + 0.05), carry_max))
    ox, oy = o[0] + off[0], o[1] + off[1]          # the tool stands beside the object for a single-jaw gripper
    gx, gy = goal[0] + off[0], goal[1] + off[1]
    return [("move", [ox, oy, carry]), ("move", [ox, oy, grasp_z]), ("grip", 1.0), ("move", [ox, oy, carry]),
            ("move", [gx, gy, carry]), ("move", [gx, gy, release_z]), ("grip", 0.0),
            ("move", [gx, gy, carry])]


# ---------------- the closed loop: look, check, retry ----------------
MAX_ATTEMPTS = 3          # the first try and two retries, then the robot asks to be shown
SEEN_MOVED = 0.015        # the object is this far from where it was believed: say so
LOOK_TICKS = 3            # hold still this long before looking, so the arm and the picture settle


def closed_loop_on(world, env=None) -> bool:
    """CLOSED_LOOP=on|off|auto: auto is on for a real arm (and the mock one), where the robot only believes where objects are."""
    v = ((os.environ if env is None else env).get("CLOSED_LOOP") or "auto").strip().lower()
    return v == "on" or (v == "auto" and getattr(world, "name", "sim") == "real")


def loop_start(world, task: dict, skill: dict, max_attempts: int = MAX_ATTEMPTS) -> tuple[list, dict]:
    """(first steps, loop state) of a move that looks before it grips: park the arm out of the way, then observe.
    Everything after `observe` is planned from what is seen."""
    park = [float(v) for v in world.observe_pose(world.obj_pos(task["object"])[:2])]
    lp = {"task": task, "skill": skill, "attempts": 0, "max": max_attempts, "park": park, "planned": None, "log": []}
    return _park_steps(world, lp) + [("observe", None)], lp


def _park_steps(world, lp) -> list:
    """Go to the parking spot (high, off to the side: see World.observe_pose)."""
    return [("move", lp["park"])]


def _hold(world, r):
    return np.array([0, 0, 0, 1.0 if world.hand.grip else 0.0, r.get("yaw", 0.0)])


def _say(r, text):
    r.setdefault("events", []).append(text)
    r["loop"]["log"].append(text)


def _end(r):
    r["i"] = len(r["wps"])


def _retry(world, r, why: str):
    """A check failed: open the jaws, lift, go back to look, or give up after the last attempt."""
    lp = r["loop"]
    lp["attempts"] += 1
    name = lp["task"]["name"]
    if lp["attempts"] >= lp["max"]:
        _say(r, f"I tried {lp['attempts']} times and the {name} still isn't where it should be.")
        return _end(r)
    _say(r, f"{why} Trying again ({lp['attempts'] + 1} of {lp['max']}).")
    here = world.hand_pos()
    up = [float(here[0]), float(here[1]), float(world.carry_height())]
    r["wps"] = r["wps"][:r["i"]] + [("grip", 0.0), ("move", up)] + _park_steps(world, lp) + [("observe", None)]
    r["wait"] = r["ticks"] = 0


def _observe(world, r):
    lp, me = r["loop"], r["loop"]["task"]["object"]
    task, name = lp["task"], lp["task"]["name"]
    r["wait"] = r.get("wait", 0) + 1
    if r["wait"] <= LOOK_TICKS:
        return _hold(world, r)
    r["wait"] = 0
    believed = lp["planned"] if lp["planned"] is not None else world.obj_pos(me)[:2].copy()   # before looking moves the belief
    seen = world.observe(me)
    if seen is None:
        _say(r, f"I can't see the {name}.")
        lp["lost"] = True
        return _end(r) or _hold(world, r)
    if np.linalg.norm(np.asarray(seen, float) - believed) > SEEN_MOVED and lp["attempts"] == 0:
        _say(r, f"The {name} isn't where I thought. Adjusting.")
    lp["planned"] = np.asarray(seen, float)
    w = waypoints(world, task, lp["skill"], pos=seen)
    r["wps"] = r["wps"][:r["i"]] + [w[0], w[1], w[2], ("check_grasp", None), w[3], ("check_lift", None), w[4], w[5], w[6], w[7],
                                     *_park_steps(world, lp), ("check_goal", None)]
    r["ticks"] = 0                                          # index i now points at the first step planned from what was seen
    return _hold(world, r)


CHECK_TICKS = 8           # a grasp may take a few ticks to show in the contacts: look this long before calling it a miss
GOAL_SETTLE = 20          # let the object stop moving before judging where it ended up


def _check(kind, world, r):
    lp = r["loop"]
    me, name = lp["task"]["object"], lp["task"]["name"]
    r["wait"] = r.get("wait", 0) + 1
    if kind in ("check_grasp", "check_lift"):
        lifted = kind == "check_lift" and world.obj_pos(me)[2] > world.half(me) + 0.03     # it is up in the air: carried, whatever the contacts say
        if world.hand.held == me or lifted:                  # held: on to the next step
            r["i"] += 1
            r["wait"] = 0
        elif r["wait"] >= CHECK_TICKS:                       # still not held after looking for a while
            _retry(world, r, f"I missed the {name}." if kind == "check_grasp" else f"I dropped the {name}.")
    elif kind == "check_goal":
        if r["wait"] >= GOAL_SETTLE:
            r["wait"] = 0
            world.observe(me)                                # look once more: where did it end up?
            if goal_met(world, lp["task"]) or _near_goal(world, lp["task"]):
                _end(r)
            else:
                _retry(world, r, f"The {name} isn't where it should be.")
    return _hold(world, r)


def _near_goal(world, task) -> bool:
    """Close enough not to disturb it again: within the 6 cm the move is judged by."""
    return float(np.linalg.norm(world.obj_pos(task["object"])[:2] - np.asarray(task["goal"], float))) < 0.06 and not task.get("stack")


def waypoint_action(world: World, r: dict, d=None, hs=None):
    """Next action along r["wps"]; advances r. Returns None when finished.
    With r["loop"] (see loop_start) the list also has observe and check steps, which look and may re-plan or retry."""
    hs = hs if hs is not None else world.hand
    if r.get("policy") is not None:                         # a learned policy drives this move (see policy.LearnedExecutor)
        return r["policy"].action(world, d, hs)
    if r["i"] >= len(r["wps"]):
        return None
    kind, arg = r["wps"][r["i"]]
    if kind == "observe":
        return _observe(world, r)
    if kind in ("check_grasp", "check_lift", "check_goal"):
        return _check(kind, world, r)
    if kind == "grip":
        r["wait"] = r.get("wait", 0) + 1
        if r["wait"] >= 4 and (world.grip_settled(d) or r["wait"] >= 25):
            r["i"], r["wait"] = r["i"] + 1, 0
        return np.array([0, 0, 0, arg, r.get("yaw", 0.0)])
    dv = np.array(arg) - world.hand_pos(d)
    n = np.linalg.norm(dv)
    r["ticks"] = r.get("ticks", 0) + 1
    if n < 0.003 or r["ticks"] > 80:  # reached, or unreachable: move on rather than hang
        r["i"] += 1
        r["ticks"] = 0
    v = dv * 5.0
    if np.linalg.norm(v) > r["speed"]:
        v *= r["speed"] / np.linalg.norm(v)
    return np.array([*v, 1.0 if hs.grip else 0.0, r.get("yaw", 0.0)])


def imagine(world: World, task: dict, wps: list, speed: float) -> dict:
    """Run the whole move on a copy of the world and judge the result."""
    twin = world.twin                        # always imagined on the simulation, whatever robot will do it
    d, hs = twin.clone()
    before = {n: twin.obj_pos(n)[:2].copy() for n in twin.things()}
    r = {"wps": wps, "i": 0, "speed": speed, "yaw": twin.grasp_yaw(task["object"])}
    path = []
    for _ in range(900):
        a = waypoint_action(twin, r, d, hs)
        if a is None:
            break
        twin.step(a, d, hs)
        path.append(twin.hand_pos(d))
    for _ in range(25):  # let it settle
        twin.step(np.array([0, 0, 0, 0.0, r["yaw"]]), d, hs)
    res = outcome(twin, task, before, d, hs)
    res["path"] = np.array(path)
    return res


# ---------------- learning per object type ----------------
def extract(ep: dict) -> dict | None:
    t = ep["task"]
    S = np.array([f["state"] for f in ep["frames"]])
    A = np.array([f["action"] for f in ep["frames"]])
    carrying = S[:, 9] > 0.5
    if not carrying.any():
        return None
    g = int(np.argmax(carrying))
    r = g + int(np.argmin(carrying[g:])) if not carrying[g:].all() else len(S)
    h = max(t.get("h", 0.02), 1e-3)
    hand, obj = S[:, 0:3], S[:, 4:7]
    sp = np.linalg.norm(A[:, :3], axis=1)
    moving = sp[sp > 0.03]
    return {
        "grip": float(np.clip((hand[g, 2] - obj[g, 2]) / h, -0.8, 0.8)),
        "lift": float(np.clip(hand[g:r, 2].max() - (t.get("tallest", 0.1) + h), 0.01, 0.15)),
        "drop": float(np.clip(hand[r - 1, 2] - h, 0.0, 0.05)),
        "speed": float(np.clip(np.percentile(moving, 75) if len(moving) else 0.15, 0.06, VMAX)),
        "dist": float(np.linalg.norm(obj[-1, :2] - obj[0, :2])),
    }


def _wmedian(values, weights) -> float:
    order = np.argsort(values)
    v, w = np.asarray(values, float)[order], np.asarray(weights, float)[order]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, 0.5 * c[-1])])


class PropSkills:
    def __init__(self):
        self.demos: list[dict] = []          # {grip, lift, drop, speed, dist, m}

    def fit(self, episodes: list[dict]):
        self.demos = []
        for ep in episodes:
            t = ep.get("task", {})
            if t.get("kind") != "prop" or not ep.get("success", True):
                continue
            k = extract(ep)
            if k:
                k["m"] = M.from_task(t)
                self.demos.append(k)

    def _near(self, m: dict) -> list[tuple[float, dict]]:
        return [(M.distance(m, x["m"]), x) for x in self.demos]

    def count(self, m: dict) -> int:
        """Demos of objects of a similar size."""
        return sum(d <= KNOWN for d, _ in self._near(m))

    def known(self, m: dict) -> bool:
        return self.count(m) >= MIN_PROP_DEMOS

    def counts(self) -> dict:
        """Demos per size class (for the dashboard)."""
        out = {c: 0 for c in M.CLASSES}
        for x in self.demos:
            out[M.size_class(x["m"])] += 1
        return out

    def plan(self, m: dict, dist: float) -> dict:
        near = self._near(m)
        if not near:
            return {**DEFAULTS, "uncertainty": 9.0, "parts": {"demos": 0}}
        w = np.array([np.exp(-(d / KNOWN) ** 2) for d, _ in near]) + 1e-6
        ks = [x for _, x in near]
        med = {k: _wmedian([x[k] for x in ks], w) for k in ("grip", "lift", "drop", "speed")}
        n_eff = float(w.sum())
        nearest = min(d for d, _ in near)
        novelty = max(min(abs(x["dist"] - dist) for x in ks) / 0.4, nearest / KNOWN)
        spread = float(np.sqrt(np.cov([x["drop"] for x in ks], aweights=w)) / 0.02) if len(ks) > 1 and n_eff > 1.01 else 0.3
        u = 0.15 + 0.4 / max(n_eff, 1.0) + 0.5 * novelty + 0.3 * spread
        return {**med, "uncertainty": u,
                "parts": {"demos": self.count(m), "novelty": round(novelty, 2), "spread": round(spread, 2)}}
