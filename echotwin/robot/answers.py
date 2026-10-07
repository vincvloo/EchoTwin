"""What the robot says when asked about itself or the table: answers from its own state, no AI."""
import numpy as np

from . import config
from .features import measure as M
from .features import move_things as MT
from .features import prop_skills as PS
from .features import tasks as T
from .features.everyday import listing

NO_TABLE = "Scan your table first, so I know what is on it."


def _on(world) -> str:
    """Where the things are, for sentences: 'the table', 'the floor', 'the surface'."""
    kind = (getattr(world.layout, "surface", None) or {}).get("kind", "table")
    return {"table": "the table", "floor": "the floor"}.get(kind, "the surface")


def about_robot(sim, n: str) -> str:
    """n: why_stop | doing | sure | learned | who | help (the names in router.ROBOT_Q)."""
    if n == "why_stop":
        if sim.halted:
            return {"you said stop": "You told me to stop.", "shake": "You shook the phone.",
                    "button": "You pressed stop."}.get(sim.halt_reason, f"Because {sim.halt_reason}.")
        return "I haven't stopped."
    if n == "doing":
        if sim.halted:
            return "Nothing. I'm stopped. Say continue when ready."
        t = sim.task
        return {
            "move": f"Moving the {t['name']}." if t else "Working.",
            "teach": f"Watching you. You're showing me: {t['instruction']}." if t else "Watching you.",
            "review": "Waiting for you to keep or discard the demo.",
            "replay": "Replaying your video in my twin.",
            "practice": "Practising in my twin.",
        }.get(sim.mode, "Waiting for an instruction.")
    props = sim.world.layout.props
    if not props:
        return NO_TABLE
    names = [p["name"] for p in props]
    if n == "sure":
        t = sim.task
        if sim.plan and t:
            return (f"About {T.percent_sure(sim.plan['uncertainty'])} percent. "
                    f"{sim.plan.get('why', '').split(' · ')[0]}.")
        if t:
            c = sim.skills.count(M.from_task(t))
            return f"Not sure yet. I have {c} demo{'s' if c != 1 else ''} of something this size."
        return "Tell me what to move, and I'll tell you how sure I am."
    if n == "learned":
        known = [f"{PS.KIND_LABEL[k]} from {c} demo{'s' if c != 1 else ''}" for k, c in sim.skills.counts().items() if c]
        if not known:
            return "Nothing on this table yet. Tell me to move something and show me once."
        return "I know how to move " + " and ".join(known) + "."
    if n == "who":
        return f"I'm {config.ROBOT_NAME}. Tell me what to move on your table, and where."
    if n == "help":
        other = names[1] if len(names) > 1 else "the table's middle"
        return f"Try: put the {names[0]} next to the {other}. Or say stop, anytime."
    return "I'm not sure."


def where_prop(world, i: int) -> str:
    name = world.layout.props[i]["name"]
    me = f"prop_{i}"
    if world.hand.attached == me:
        return f"I'm holding the {name}."
    p = world.obj_pos(me)
    lr = "left" if p[0] < -0.1 else "right" if p[0] > 0.1 else "middle"
    fb = "front" if p[1] < -0.08 else "back" if p[1] > 0.08 else "centre"
    near = [(np.linalg.norm(world.obj_pos(f"prop_{j}")[:2] - p[:2]) - world.radius(me) - world.radius(f"prop_{j}"), j)
            for j in range(len(world.layout.props)) if j != i]
    close = min(near) if near else None
    extra = f", next to the {world.layout.props[close[1]]['name']}" if close and close[0] < 0.06 else ""
    spot = f"at the {fb} {lr}" if lr != "middle" else f"in the {fb} middle"
    return f"The {name} is {spot} of {_on(world)}{extra}."


def about_table(world, n: str, data: dict) -> str:
    """n: which | count | where | see (the names in router.SCENE_Q); data: {"index": k} or {"text": what was said}."""
    props = world.layout.props
    if n == "which":
        k = data.get("index")
        if not props:
            return "I haven't mapped any objects. Snap a photo of your table first."
        if k is None or not 1 <= k <= len(props):
            return f"I have {len(props)} objects, numbered 1 to {len(props)}."
        return f"Object {k} is {props[k - 1]['name']}."
    if not props:
        return "I don't see anything yet. Snap a photo of your table first."
    if n == "count":
        return f"I see {len(props)} thing{'s' if len(props) != 1 else ''} on {_on(world)}."
    if n == "where":
        ment = MT._mentions(data.get("text", ""), props)
        if ment:
            return where_prop(world, ment[0][1])
        return " ".join(where_prop(world, i) for i in range(len(props)))
    where = "your table" if _on(world) == "the table" else _on(world)
    return f"On {where} I see {listing([p['name'] for p in props])}."
