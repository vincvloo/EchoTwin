"""One router for every utterance (plan section 5). Order matters: safety is always first."""
import re
from dataclasses import dataclass, field

from .features.tasks import _norm


@dataclass
class Intent:
    kind: str                 # safety | control | robot_q | scene_q | other
    name: str = ""
    data: dict = field(default_factory=dict)
    text: str = ""


SAFETY = r"\b(stop|halt|freeze|wait|pause|hold on|emergency|no no)\b"
CONTROL = [
    ("continue", r"\b(continue|go on|resume|carry on|keep going|go ahead|proceed)\b"),
    ("arm_robot", r"^(arm|arm the robot|arm the arm|enable the arm|let the arm move)$"),
    ("keep", r"\b(keep it|keep that|keep this|save it|yes keep|keep)\b"),
    ("discard", r"\b(discard|throw it away|delete it|bin it|drop that one|don t keep)\b"),
    ("robot_turn", r"\b(your turn|you do it|you try|try it|try again|try now|show me what you learned|over to you)\b"),
    ("human_turn", r"\b(my turn|i ll take over|let me|take over|i ll do it|i ll show you)\b"),
    ("grip", r"^(grip|grab|close|pick( it)? up)$"),
    ("release", r"^(release|open|let go|drop it)$"),
    ("reset", r"^(reset|reset scene|back to the scan|restore)$"),
    ("home", r"\b(go home|home position|go back)\b"),
    ("yes", r"^(yes|yeah|yep|sure|ok|okay|do it)$"),
    ("no", r"^(no|nope|nah)$"),
]
ROBOT_Q = [
    ("why_stop", r"\bwhy (did|have) you stop"),
    ("doing", r"\bwhat are you doing\b|\bwhat s happening\b|\bstatus\b"),
    ("sure", r"\bhow (sure|confident|certain)\b|\bare you sure\b"),
    ("learned", r"\bwhat (have|did) you learn|\bwhat do you know\b|\bwhat can you do\b"),
    ("who", r"\bwho are you\b|\bwhat s your name\b|\byour name\b|\bintroduce yourself\b"),
    ("help", r"^help$|\bwhat can i say\b"),
]
NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "first": 1,
       "second": 2, "third": 3, "fourth": 4, "fifth": 5}
WHICH = (r"\b(identify|what is|what s|which is|name)\b.*\b(object|number|thing|item)\s*(?:number\s*)?(\d+|"
         + "|".join(NUM) + r")\b")

SCENE_Q = [
    ("see", r"\bwhat (do|can) you see\b|\bdescribe\b|\blook (at|around)\b|\bwhat s on the table\b"),
    ("where", r"\bwhere (is|s|are)\b"),
    ("count", r"\bhow many\b"),
]


def route(text: str) -> Intent:
    t = _norm(text)
    if not t:
        return Intent("other", text=text)
    if re.search(SAFETY, t) and not re.search(r"\bwhy (did|have) you stop", t):
        return Intent("safety", "stop", text=text)
    for name, pat in CONTROL:
        if re.search(pat, t):
            return Intent("control", name, text=text)
    for name, pat in ROBOT_Q:
        if re.search(pat, t):
            return Intent("robot_q", name, text=text)
    m = re.search(WHICH, t)
    if m:
        k = m.group(3)
        return Intent("scene_q", "which", {"index": int(k) if k.isdigit() else NUM[k]}, text=text)
    for name, pat in SCENE_Q:
        if re.search(pat, t):
            return Intent("scene_q", name, text=text)
    return Intent("other", text=text)
