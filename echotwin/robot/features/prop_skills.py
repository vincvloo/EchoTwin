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

from ..world import CTRL_DT, STACK_CLEAR, VMAX, World
from . import measure as M
from . import move_things as MT

MIN_PROP_DEMOS = 1
ASK = 1.0
KNOWN = 1.0                  # a demo this close in size (see measure.distance) counts as "a similar object"
KIND_LABEL = M.CLASS_LABEL   # the dashboard cards: demos per size class
DEFAULTS = {"grip": 1.0, "lift": 0.05, "drop": 0.004, "speed": 0.2}


def make_task(world: World, plan: dict, heard: str) -> dict:
    """The description of one move that everything else works from: the robot, the demos, the experiments."""
    i = plan["prop"]
    me = f"prop_{i}"
    pr = world.layout.props[i]
    goal = MT.goal_xy(world, plan)
    o = world.obj_pos(me)
    shape = pr.get("shape", "box")
    task = {"kind": "prop", "plan": plan, "object": me, "target": f"{shape} things", "shape": shape,
            "name": pr["name"], "instruction": heard, "goal": [float(goal[0]), float(goal[1])],
            "ref": f"prop_{plan['goal'][1]}" if plan["goal"][0] == "near" else None,
            "h": float(world.half(me)), "tallest": float(world.tallest()), "m": M.measure(world, me),
            "start": [float(o[0]), float(o[1])]}
    if plan["goal"][0] == "near" and plan["goal"][2] == "on top of":
        task["stack"] = True
        task["ref_name"] = world.layout.props[plan["goal"][1]]["name"]
    return task


# ---------------- geometry of success ----------------
def tilt_deg(world: World, name: str, d=None) -> float:
    """How far it leans (degrees); 0 for a ball, which looks the same however it lies (World.has_up)."""
    return world.tilt(name, d) if world.has_up(name) else 0.0


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
    ox, oy = o[0] + off[0], o[1] + off[1]          # the tool stands beside the object for a single-jaw gripper
    gx, gy = goal[0] + off[0], goal[1] + off[1]
    if task.get("stack"):  # set it down on top of the other object
        ref = task["ref"]
        top = float(world.obj_pos(ref, d)[2] + world.half(ref))
        carry = float(min(max(carry, top + h + skill["drop"] + 0.056), carry_max, world.path_ceiling((ox, oy), (gx, gy), me)))
        # what hangs below the tool must clear the other object's top: under a low ceiling, hold it lower down
        bottom = float(o[2] - h)
        grasp_z = float(max(min(grasp_z, carry - STACK_CLEAR - top + bottom), bottom + 0.004, 0.004))
        hang = world.stack_hang(me, grasp_z - bottom)
        release_z = float(min(top + max(h, hang) + skill["drop"] + 0.006, carry))
    steps = [("move", [ox, oy, carry]), ("move", [ox, oy, grasp_z]), ("grip", 1.0), ("move", [ox, oy, carry]),
             ("move", [gx, gy, carry]), ("move", [gx, gy, release_z]), ("grip", 0.0),
             ("move", [gx, gy, carry])]
    return with_drives(world, steps, (ox, oy), (gx, gy), me) if getattr(world, "mobile", False) else steps


def with_drives(world: World, steps: list, pick, place, me: str) -> list:
    """A mobile base drives first when the object, or later the place, is out of the arm's reach from where it stands:
    drive, stop, then the arm moves as on a fixed base."""
    (_, at_pick), (_, at_place) = world.stands(pick, place, me)
    out = list(steps)
    if at_pick is not None:
        out.insert(0, ("drive", at_pick.tolist()))
    if at_place is not None:
        out.insert(out.index(("grip", 1.0)) + 2, ("drive", at_place.tolist()))     # after lifting it
    return out


# ---------------- the closed loop: look, check, retry ----------------
FAR_ABOVE = 0.01          # m: a step this far out of the arm's reach is done at the nearest reachable point
SETTLED = 0.003           # m/s: slower than this, for STILL_TICKS in a row, the tool and what it carries are at rest
STILL_TICKS = 15
MAX_ATTEMPTS = 3          # the first try and two retries, then the robot asks to be shown
SEEN_MOVED = 0.015        # the object is this far from where it was believed: say so
LOOK_TICKS = 3            # hold still this long before looking, so the arm and the picture settle


def closed_loop_on(world, env=None) -> bool:
    """CLOSED_LOOP=on|off|auto: auto is on for a real arm (and the mock one), where the robot only believes where objects are."""
    v = ((os.environ if env is None else env).get("CLOSED_LOOP") or "auto").strip().lower()
    return v == "on" or (v == "auto" and getattr(world, "name", "sim") == "real")


ALIGN_TOL = 0.002         # the seen gripper is within this of where it should be: lined up
ALIGN_SETTLE = 10         # ticks to let the arm come to rest before looking at it
ALIGN_ARRIVED = 0.008     # the arm counts as having arrived at a hover when it is this close to it
ALIGN_ARRIVE_TRIES = 4    # times a move is repeated because the arm is still on its way
ALIGN_LOOKS = 3           # looks at the gripper before going down
ALIGN_MAX = 0.03          # a bigger correction than this means something other than a small offset is wrong
ALIGN_SAY = 0.003         # say it when the gripper was this far off
ALIGN_CLEARANCE = 0.012   # only look at the gripper when the object fits the jaws this tightly (m per side): a loose fit does not need it


def align_on(world, env=None) -> bool:
    """ALIGN=on|off: look at the gripper and line it up before going down (only for a tight fit, only with a camera that sees the gripper).
    Off unless asked for: in our tests it helps a tall cylinder only when the arm is badly off (docs/RESULTS.md)."""
    v = ((os.environ if env is None else env).get("ALIGN") or "off").strip().lower()
    return v == "on" and hasattr(world, "see_tool")


def _tight_fit(world, me: str) -> bool:
    """Is the object nearly as wide as the jaws open? Then a centimetre of error puts a pad on its rim."""
    opening = float(world.twin.arm.max_opening or 0.08) if hasattr(world, "twin") else 0.08
    return (opening - world.grasp_width(me)) / 2 < ALIGN_CLEARANCE


def loop_start(world, task: dict, skill: dict, max_attempts: int = MAX_ATTEMPTS, align: bool | None = None) -> tuple[list, dict]:
    """(first steps, loop state) of a move that looks before it grips: park the arm out of the way, then observe.
    Everything after `observe` is planned from what is seen. With `align`, the gripper is looked at and lined up before the descent."""
    park = [float(v) for v in world.observe_pose(world.obj_pos(task["object"])[:2])]
    lp = {"task": task, "skill": skill, "attempts": 0, "max": max_attempts, "park": park, "planned": None, "log": [],
          "align": (align_on(world) and _tight_fit(world, task["object"])) if align is None else bool(align), "bias": np.zeros(2), "looks": 0}
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
    lp["bias"], lp["looks"], lp["stage"] = np.zeros(2), 0, 0
    _plan_pick(world, r)
    return _hold(world, r)


def _shift(step, bias):
    kind, arg = step
    return (kind, [arg[0] - bias[0], arg[1] - bias[1], arg[2]]) if kind == "move" else step


def _plan_pick(world, r, stage: int = 0, align: bool | None = None):
    """From the current step on: hover over the object, (look at the gripper,) a second, lower hover and look, go down, grip, check, lift,
    carry, put down, park, check. The picking moves are shifted by the measured error of the arm (`bias`), the placing moves are not: that
    error is local to the pose. The arm's error changes with its pose, so the gripper is looked at three times: high, just above the object, and at the grip height.
    `stage` is the first level still to be looked at (0 high, 1 just above, 2 at the grip)."""
    lp = r["loop"]
    w = waypoints(world, lp["task"], lp["skill"], pos=lp["planned"])
    b = lp["bias"]
    look = lp["align"] if align is None else align
    me = lp["task"]["object"]
    ox, oy, carry = w[0][1]
    low = max(float(world.obj_pos(me)[2] + world.half(me)) + 0.015, w[1][1][2] + 0.03)      # the pads clear the top of the object
    grasp_z = w[1][1][2]
    levels = [carry] + ([low] if low < carry - 0.03 else []) + ([grasp_z] if grasp_z < low - 0.01 else [])      # high, just above, at the grip
    steps = []
    for k, z in enumerate(levels):
        if stage <= k:
            steps += [_shift(("move", [ox, oy, z]), b)] + ([("align", None)] if look else [])
    r["wps"] = r["wps"][:r["i"]] + steps + [w[2], ("check_grasp", None), _shift(w[3], b), ("check_lift", None),
                                             w[4], w[5], w[6], w[7], *_park_steps(world, lp), ("check_goal", None)]
    r["ticks"] = 0                                          # index i now points at the first step planned from what was seen


def _align(world, r):
    """Hovering over the object: look at where the gripper really is, and move the pick by the difference."""
    lp = r["loop"]
    name = lp["task"]["name"]
    r["wait"] = r.get("wait", 0) + 1
    if r["wait"] <= ALIGN_SETTLE:                           # the arm is still creeping the last millimetres towards where it was sent
        return _hold(world, r)
    r["wait"] = 0
    want = np.asarray(waypoints(world, lp["task"], lp["skill"], pos=lp["planned"])[0][1][:2], float)      # the hover point, unshifted
    prev = r["wps"][r["i"] - 1] if r["i"] > 0 else None
    if prev is not None and prev[0] == "move" and lp.get("arrive", 0) < ALIGN_ARRIVE_TRIES             and np.linalg.norm(world.hand_pos()[:2] - np.asarray(prev[1][:2])) > ALIGN_ARRIVED:
        lp["arrive"] = lp.get("arrive", 0) + 1              # the move gave up before the arm got there (a long way at half speed): keep going
        r["wps"].insert(r["i"], prev)
        r["ticks"] = 0
        return _hold(world, r)
    lp["arrive"] = 0
    seen = world.see_tool()
    if seen is None:
        _say(r, "I can't see the gripper well enough to line it up. Going on without.")
        r["i"] += 1
        return _hold(world, r)
    resid = np.asarray(seen, float) - want
    lp["looks"] += 1
    if np.linalg.norm(resid) <= ALIGN_TOL:
        if lp["stage"] == 0 and (lp["looks"] > 1 or np.linalg.norm(lp["bias"]) > ALIGN_SAY):
            _say(r, f"Lined up with the {name}.")
        lp["stage"], lp["looks"] = lp["stage"] + 1, 0
        r["i"] += 1
        return _hold(world, r)
    lp["bias"] = lp["bias"] + resid
    if np.linalg.norm(lp["bias"]) > ALIGN_MAX or lp["looks"] >= ALIGN_LOOKS:
        bad = np.linalg.norm(lp["bias"]) > ALIGN_MAX
        if bad or np.linalg.norm(resid) > 5 * ALIGN_TOL:
            _say(r, f"I could not line the gripper up with the {name} (it looks {np.linalg.norm(lp['bias']) * 1000:.0f} mm off). Going on without.")
            lp["bias"] = np.zeros(2)                        # a correction I cannot trust is worse than none
            _plan_pick(world, r, stage=lp["stage"], align=False)
            return _hold(world, r)
        _plan_pick(world, r, stage=lp["stage"], align=False)       # close enough after the last look: go with the bias found so far
        return _hold(world, r)
    if np.linalg.norm(resid) > ALIGN_SAY and lp["stage"] == 0 and lp["looks"] == 1:
        _say(r, f"The gripper is {np.linalg.norm(resid) * 1000:.0f} mm off from where my joints say. Lining it up.")
    _plan_pick(world, r, stage=lp["stage"])                 # this hover again with the corrected bias, then look once more
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
    if kind == "drive":                                     # a mobile base drives there; the arm holds still meanwhile
        if hs.drive is None and r.get("driving"):
            r["i"], r["driving"] = r["i"] + 1, False
        elif not r.get("driving"):
            hs.drive, r["driving"] = np.asarray(arg, float), True
        return np.array([0, 0, 0, 1.0 if hs.grip else 0.0, r.get("yaw", 0.0)])
    if kind == "observe":
        return _observe(world, r)
    if kind == "align":
        return _align(world, r)
    if kind in ("check_grasp", "check_lift", "check_goal"):
        return _check(kind, world, r)
    if kind == "grip":
        r["wait"] = r.get("wait", 0) + 1
        if r["wait"] >= 4 and (world.grip_settled(d) or r["wait"] >= 25):
            r["i"], r["wait"] = r["i"] + 1, 0
            r["hold"] = arg > 0.5
        return np.array([0, 0, 0, arg, r.get("yaw", 0.0)])
    target = np.array(arg, float)
    if r.get("hold") and hs.held:                         # steer the object, not the tool point (see _held_off)
        target[:2] -= _held_off(world, d, hs)
    reachable = world._clamp(target, hs)                   # where the arm can really go
    clamped = bool(np.linalg.norm(reachable - target) > FAR_ABOVE)
    here = world.hand_pos(d)
    dv = target - here                                     # still aimed at the step (aiming higher slows the carry: less swing)
    n = np.linalg.norm(reachable - here if clamped else dv)  # but done once it is where it can get to
    r["ticks"] = r.get("ticks", 0) + 1
    # a step above the arm's ceiling is reached at the ceiling, once the tool and what it carries have come to rest (a tall
    # thing swings), instead of waiting out the tick limit
    moving = r.get("last") is not None and np.linalg.norm(here - r["last"]) / CTRL_DT > SETTLED
    if hs.held:
        o = world.obj_pos(hs.held, d)
        moving = moving or (r.get("last_obj") is not None and np.linalg.norm(o - r["last_obj"]) / CTRL_DT > SETTLED)
        r["last_obj"] = o
    r["last"] = here
    r["still"] = 0 if moving else r.get("still", 0) + 1
    if n < 0.003 and (not clamped or r["still"] >= STILL_TICKS) or r["ticks"] > 80:  # reached, or unreachable: move on
        r["i"] += 1
        r["ticks"] = 0
        r["last"] = r["last_obj"] = None
        r["still"] = 0
    v = dv * 5.0
    if np.linalg.norm(v) > r["speed"]:
        v *= r["speed"] / np.linalg.norm(v)
    return np.array([*v, 1.0 if hs.grip else 0.0, r.get("yaw", 0.0)])


def _held_off(world: World, d, hs) -> np.ndarray:
    """How far the held thing is from where the plan expects it, beside the tool point (x, y). A single moving jaw holds it
    off-centre against the fixed jaw: the plan puts the tool `grasp_offset` beside it, but the thing sits a few mm further
    out and swings around the tool point when the wrist turns on the way. The moves that carry it are corrected by this,
    so the thing, not the tool point, follows the plan. A parallel gripper holds it centred: no correction."""
    if world.arm.gripper["mode"] != "single" or not hs.held:
        return np.zeros(2)
    seen = world.obj_pos(hs.held, d)[:2] - world.hand_pos(d)[:2]
    return seen + world.grasp_offset(hs.held, hs.yaw)[:2]          # the side it was gripped from, not chosen again


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
