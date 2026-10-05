"""What was said or typed: practice, a move, a question for the AI, or a plain command for the robot."""
import asyncio
import re

from .features import move_things as MT
from .hub import brain, emit, latest_photo, sim
from .router import Intent, route

PRACTISE = r"\b(practi[cs]e|train (yourself|in (the )?sim)|generate (data|demos))\b"
RELATIONS = ("next to", "left of", "right of", "in front of", "behind", "near", "beside")


async def handle_text(text: str, source: str):
    text = text.strip()
    if not text:
        return
    emit({"t": "log", "who": "user", "text": text, "source": source})
    it = route(text)
    props = sim.world.layout.props
    if props and re.search(PRACTISE, text.lower()):
        ment = MT._mentions(text, props)
        sim.submit(sim.say, "Practising in my twin. This takes a few seconds.")
        if ment:
            sim.submit(sim.practice, None, 3, True, ment[0][1])
        else:
            sim.submit(sim.practice)
        return
    if props and it.kind not in ("safety", "control", "robot_q") and it.name not in ("which", "see"):
        plan = MT.parse(text, props)
        if plan is None and sim.pending_move is not None:  # answer to "where should I put it?"
            again = MT.parse(f"move object {sim.pending_move['prop'] + 1} {text}", props)
            plan = again if again and again["goal"] else None
        if plan:
            sim.submit(sim.handle_prop_task, plan, text)
            return
        if it.kind == "other":
            await _understand_with_ai(text, [p["name"] for p in props])
            return
    if not props and it.kind == "other":
        reply = await brain.chat(text) if brain.enabled else None
        sim.submit(sim.say, reply or "Scan your table first, so I know what is on it.")
        return
    sim.submit(sim.handle_intent, it)
    if it.kind == "scene_q" and it.name == "see" and brain.enabled and latest_photo["jpeg"]:
        asyncio.ensure_future(describe_photo(text))


async def _understand_with_ai(text: str, names: list[str]):
    """An odd phrasing: ask the AI, using the real object names; without AI, say what can be moved."""
    if not brain.enabled:
        sim.submit(sim.say, f"I can move the {names[0]}" + (f" or the {names[1]}" if len(names) > 1 else "")
                   + ". Tell me which one and where.")
        return
    emit({"t": "thinking", "on": True})
    parsed = await brain.parse_everyday(text, names)
    emit({"t": "thinking", "on": False})
    plan = _everyday_plan(parsed, len(names))
    if plan:
        emit({"t": "log", "who": "system", "text": f"AI understood: move the {names[plan['prop']]}"})
        sim.submit(sim.handle_prop_task, plan, text)
    elif parsed and parsed.get("clarify"):
        sim.submit(sim.say, str(parsed["clarify"])[:120])
    else:
        reply = await brain.chat(text, scene=", ".join(names))
        sim.submit(sim.say, reply or "I can move the things on your table. Tell me which one and where.")


def _everyday_plan(parsed: dict | None, n: int) -> dict | None:
    """AI answer {object, relation, reference} -> move plan, or None."""
    if not parsed or "object" not in parsed:
        return None
    try:
        i = int(parsed["object"]) - 1
    except (TypeError, ValueError):
        return None
    if not 0 <= i < n:
        return None
    rel = str(parsed.get("relation", "")).lower()
    ref = parsed.get("reference")
    if ref is not None and rel in RELATIONS:
        try:
            j = int(ref) - 1
        except (TypeError, ValueError):
            return None
        if 0 <= j < n and j != i:
            return {"prop": i, "goal": ("near", j, "next to" if rel in ("near", "beside") else rel)}
    if rel in ("middle", "center", "centre"):
        return {"prop": i, "goal": ("center",)}
    if rel in MT.DIRS:
        return {"prop": i, "goal": ("dir", MT.DIRS[rel], 0.2)}
    return {"prop": i, "goal": None}


async def describe_photo(question: str):
    emit({"t": "thinking", "on": True})
    desc = await brain.describe(latest_photo["jpeg"], question)
    emit({"t": "thinking", "on": False})
    if desc:
        sim.submit(sim.say, "From the photo: " + desc)
