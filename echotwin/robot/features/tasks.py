"""F1 instruction parsing and F2 'done, do, or teach me' decision (pure functions)."""
import re

from ..scene import OBJECT_NAMES, ZONE_LABEL

VERBS = r"(put|move|place|bring|take|drop|carry|get|set|stick|shift|push|give)"
PREPS = r"\b(in|into|inside|on|onto|to|at|over|in to|on to|towards?)\b"

OBJECT_WORDS = {
    "red": ["red", "rouge", "crimson"],
    "blue": ["blue", "navy"],
    "yellow": ["yellow", "gold", "golden"],
}
ZONE_WORDS = {
    "tray": ["blue tray", "tray", "blue zone", "blue note", "blue area", "blue square", "blue sticky", "blue one",
             "blue", "left zone", "left one"],
    "green": ["green zone", "green area", "green note", "green square", "green sticky", "green one", "green",
              "right zone", "right one"],
}


def _norm(text: str) -> str:
    t = text.lower().replace("-", " ")
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def find_object(text: str) -> str | None:
    t = _norm(text)
    for name, words in OBJECT_WORDS.items():
        if any(re.search(rf"\b{w}\b", t) for w in words):
            return name
    return None


def find_zone(text: str) -> str | None:
    t = _norm(text)
    for name, words in ZONE_WORDS.items():
        for w in words:
            if re.search(rf"\b{w}\b", t):
                return name
    return None


def parse_instruction(text: str) -> dict | None:
    """-> {'object', 'target', 'instruction'} with None for missing slots, {'cleanup': True}, or None."""
    t = _norm(text)
    if re.search(r"\b(clean|tidy) (it )?up\b|\bclear the table\b", t):
        return {"cleanup": True, "instruction": text.strip()}
    if not re.search(rf"\b{VERBS}\b", t) and not re.search(PREPS, t):
        return None
    m = re.search(PREPS, t)
    left, right = (t[:m.start()], t[m.end():]) if m else (t, "")
    obj = find_object(left)
    target = find_zone(right) if right else None
    if obj is None and target is None:
        # "put it in the tray" / "move the block": still a task, just missing slots
        if re.search(rf"\b{VERBS}\b", t) and re.search(r"\b(block|cube|one|it|brick|thing)\b", t):
            return {"object": None, "target": None, "instruction": text.strip()}
        return None
    if obj is None and not re.search(rf"\b{VERBS}\b", t):
        return None
    return {"object": obj, "target": target, "instruction": text.strip()}


def canonical(obj: str, target: str) -> str:
    return f"put the {obj} block in the {ZONE_LABEL[target]}"


def fill_slots(pending: dict, text: str) -> dict:
    """Answer to a clarifying question ('the red one', 'blue tray')."""
    out = dict(pending)
    if out.get("object") is None:
        out["object"] = find_object(text)
    if out.get("target") is None:
        z = find_zone(text)
        # 'blue' alone answers the object question, not the zone one
        if z and not (z == "tray" and _norm(text) in ("blue", "the blue one", "blue one") and pending.get("object") is None):
            out["target"] = z
    return out


def clarify_question(task: dict) -> str | None:
    if task.get("object") is None:
        return "Which block should I move: red, blue or yellow?"
    if task.get("target") is None:
        return "Where to: the green zone or the blue tray?"
    return None


def confidence_words(u: float) -> str:
    if u < 0.45:
        return "I'm confident about this one."
    if u < 0.6:
        return "I'm fairly sure about this one."
    return "I'm not very sure, but I'll try. Say stop if I go wrong."


def percent_sure(u: float) -> int:
    return int(round(max(5, min(99, 100 * (1 - u / 1.5)))))


__all__ = ["parse_instruction", "fill_slots", "clarify_question", "canonical", "find_object", "find_zone",
           "confidence_words", "percent_sure", "OBJECT_NAMES"]
