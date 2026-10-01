"""Small helpers for what the robot says: text normalising and confidence wording."""
import re


def _norm(text: str) -> str:
    t = text.lower().replace("-", " ")
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def confidence_words(u: float) -> str:
    if u < 0.45:
        return "I'm confident about this one."
    if u < 0.6:
        return "I'm fairly sure about this one."
    return "I'm not very sure, but I'll try. Say stop if I go wrong."


def percent_sure(u: float) -> int:
    return int(round(max(5, min(99, 100 * (1 - u / 1.5)))))


__all__ = ["_norm", "confidence_words", "percent_sure"]
