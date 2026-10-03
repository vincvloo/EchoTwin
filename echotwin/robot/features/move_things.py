"""Move everyday objects by name, approximately: "put the chocolate next to the glass", "move the case left".

Targets are places relative to the table or to another object; the robot picks the object at its centre,
lifts it clear of everything, carries it and puts it down. No learning here: this is the approximate mode.
"""
import re

import numpy as np

from .tasks import _norm

STOP = {"the", "a", "an", "of", "and", "with", "on", "in", "to", "it", "this", "that", "small", "big", "white"}
NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}
DIRS = {"left": (-1, 0), "right": (1, 0), "front": (0, -1), "back": (0, 1), "forward": (0, 1), "backward": (0, -1),
        "closer": (0, -1), "towards me": (0, -1), "away": (0, 1), "far": (0, 1)}
NEAR = r"\b(next to|beside|near|by|close to|against|on top of|on|onto|in front of|behind|left of|right of)\b"
MOVE_VERB = r"\b(put|move|place|bring|take|push|slide|shift|drop|set|carry|get|swap|give)\b"


def _mentions(text: str, props: list[dict]) -> list[tuple[int, int]]:
    """-> [(position in text, prop index)] for every prop the text refers to."""
    t = _norm(text)
    out = []
    for m in re.finditer(r"\b(?:object|number|thing|item)\s*(\d+|" + "|".join(NUM) + r")\b", t):
        k = int(m.group(1)) if m.group(1).isdigit() else NUM[m.group(1)]
        if 1 <= k <= len(props):
            out.append((m.start(), k - 1))
    for i, pr in enumerate(props):
        words = [w for w in _norm(pr["name"]).split() if w not in STOP and len(w) >= 3]
        pos = [m.start() for w in words for m in re.finditer(rf"\b{re.escape(w)}s?\b", t)]
        if pos:
            out.append((min(pos), i))
    # one mention per prop, first occurrence wins
    seen, uniq = set(), []
    for pos, i in sorted(out):
        if i not in seen:
            seen.add(i)
            uniq.append((pos, i))
    return uniq


def parse(text: str, props: list[dict]) -> dict | None:
    """-> {"prop": i, "goal": ("near", j, relation) | ("dir", (dx, dy), dist) | ("center",) | None} or None."""
    if not props:
        return None
    t = _norm(text)
    ment = _mentions(text, props)
    if not ment or not re.search(MOVE_VERB + "|" + NEAR + r"|\b(left|right|front|back|centre|center|middle)\b", t):
        return None
    mover = ment[0][1]
    ref = ment[1][1] if len(ment) > 1 else None
    rel = re.search(NEAR, t)
    if ref is not None:
        relation = rel.group(1) if rel else "next to"
        if relation in ("on", "onto", "on top of"):
            relation = "on top of"
        return {"prop": mover, "goal": ("near", ref, relation)}
    for word, d in DIRS.items():
        if re.search(rf"\b{word}\b", t):
            dist = 0.08 if re.search(r"\b(a bit|a little|slightly|little)\b", t) else 0.2
            return {"prop": mover, "goal": ("dir", d, dist)}
    if re.search(r"\b(centre|center|middle)\b", t):
        return {"prop": mover, "goal": ("center",)}
    return {"prop": mover, "goal": None}


def goal_xy(world, plan: dict) -> np.ndarray:
    """Where the moved object should end up (sim metres), kept on the table and off other objects."""
    i = plan["prop"]
    g = plan["goal"]
    if g and g[0] == "near" and g[2] == "on top of":  # stacking: right above the other object
        return world.obj_pos(f"prop_{g[1]}")[:2].copy()
    me = f"prop_{i}"
    p = world.obj_pos(me)[:2]
    r = world.radius(me)
    g = plan["goal"]
    if g[0] == "near":
        _, j, relation = g
        other = f"prop_{j}"
        q, rq = world.obj_pos(other)[:2], world.radius(other)
        gap = r + rq + 0.025
        if relation in ("left of",):
            d = np.array([-1.0, 0])
        elif relation in ("right of",):
            d = np.array([1.0, 0])
        elif relation == "in front of":
            d = np.array([0, -1.0])
        elif relation == "behind":
            d = np.array([0, 1.0])
        else:  # next to: on the side it comes from
            d = p - q
            d = d / (np.linalg.norm(d) + 1e-9)
        target = q + d * gap
    elif g[0] == "dir":
        _, d, dist = g
        target = p + np.array(d, float) * dist
    else:
        target = np.zeros(2)
    lim = np.array([world.layout.table_half[0] - 0.06, world.layout.table_half[1] - 0.06])
    target = np.clip(target, -lim, lim)
    # nudge away from anything else it would land on
    for n in world.things():
        if n == me:
            continue
        q = world.obj_pos(n)[:2]
        need = r + world.radius(n) + 0.01
        dv = target - q
        if np.linalg.norm(dv) < need:
            target = q + dv / (np.linalg.norm(dv) + 1e-9) * need
    for ob in world.layout.obstacles:  # and off fixed furniture
        q = np.array(ob["pos"][:2])
        need = r + max(ob["size"][0], ob["size"][1]) + 0.01
        dv = target - q
        if np.linalg.norm(dv) < need:
            target = q + dv / (np.linalg.norm(dv) + 1e-9) * need
    return np.clip(target, -lim, lim)


def describe_goal(world, plan: dict) -> str:
    g = plan["goal"]
    if g[0] == "near":
        return f"{g[2]} the {world.layout.props[g[1]]['name']}"
    if g[0] == "dir":
        names = {(-1, 0): "left", (1, 0): "right", (0, -1): "front", (0, 1): "back"}
        return f"to the {names.get(tuple(g[1]), 'side')}"
    return "to the middle"
