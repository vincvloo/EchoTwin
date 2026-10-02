"""Review of detections by a vision model (NVIDIA build.nvidia.com, OpenAI-compatible API).

YOLO names every blob with one of 80 words and is sometimes wrong (a sofa merged with a chair, a lamp called a
plant). A vision model looks at each object once more, in the photo, and answers: what is it really, what shape,
can a person pick it up, any grasp traits (fragile, hollow, soft), or is it not an object at all.

This file holds the parts that need no images: the key and model settings, the prompt, reading the answer,
voting over several photos, and applying the verdict to a scene. No key means no review: the detector's
names stay and nothing breaks. Standard library only; the HTTP call can be replaced for tests.
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from . import catalog, schema

URL = "https://integrate.api.nvidia.com/v1/chat/completions"
TRAITS = ("fragile", "hollow", "soft")
REPO_ROOT = Path(__file__).resolve().parents[2]
MIN_GAP = 1.5            # seconds between calls: the free tier allows about 40 a minute

PROMPT = """The photo shows a room. Yellow boxes with numbers mark things an object detector found.
The detector's guess for each number:
{guesses}
For every number say what it really is. Reply with JSON only:
{{"objects": [{{"id": 1, "name": "armchair", "shape": "box", "movable": false, "traits": [], "skip": false}}]}}
name: 1 to 4 plain words. shape: box, cylinder, flat (paper, card, phone lying down) or round (ball, egg-shaped).
movable: true for a loose object a small robot arm could carry (mug, book, bottle, small plant pot); false for furniture,
big plants and anything built in. traits: any of fragile, hollow, soft.
A box may show only part of an object (leaves of a plant, the back of a chair): name the whole object it belongs to.
skip: true only if the box is not an object at all (wall, floor, window, shadow, reflection)."""


# ---------------- settings ----------------
def load_env(path: Path | None = None):
    """Fill missing environment variables from the repo's .env (no extra package needed)."""
    path = path or REPO_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^\s*([A-Z][A-Z0-9_]*)\s*=\s*(.*?)\s*$", line)
        if m and not line.lstrip().startswith("#") and m.group(2) and m.group(1) not in os.environ:
            os.environ[m.group(1)] = m.group(2)


def config() -> dict:
    load_env()
    models = [os.environ.get("AI_VISION_MODEL", "meta/muse-glimmer-30b")]
    models += [m.strip() for m in os.environ.get("AI_FALLBACK_MODELS", "meta/llama-3.2-11b-vision-instruct").split(",")
               if m.strip()]
    return {"key": os.environ.get("AI_API_KEY", "").strip(), "models": list(dict.fromkeys(models)),
            "provider": os.environ.get("AI_PROVIDER", "nvidia").lower()}


# ---------------- the call ----------------
def _post(url: str, headers: dict, payload: dict, timeout: float = 45.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


_last = [0.0]
_down: set[str] = set()      # models that answered 404 or 410: not asked again in this run


def ask_json(jpeg: bytes, prompt: str, cfg: dict | None = None, post=_post, sleep=time.sleep) -> dict | None:
    """Send one photo and a question to the vision model; return the parsed JSON answer or None."""
    cfg = cfg or config()
    if not cfg["key"] or cfg["provider"] != "nvidia":
        return None
    b64 = base64.b64encode(jpeg).decode()
    for model in cfg["models"]:
        if model in _down:
            continue
        wait = MIN_GAP - (time.time() - _last[0])
        if wait > 0:
            sleep(wait)
        _last[0] = time.time()
        payload = {"model": model, "max_tokens": 1600, "temperature": 0.2,
                   "chat_template_kwargs": {"enable_thinking": False},
                   "messages": [{"role": "user", "content": [
                       {"type": "text", "text": prompt},
                       {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]}
        try:
            out = post(URL, {"Authorization": f"Bearer {cfg['key']}", "Content-Type": "application/json"}, payload)
            text = out["choices"][0]["message"].get("content") or ""
        except (urllib.error.URLError, OSError, KeyError, IndexError, ValueError) as e:
            print(f"[review] {model}: {type(e).__name__} {e}")
            if isinstance(e, urllib.error.HTTPError) and e.code in (404, 410):
                _down.add(model)
            continue
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        m = re.search(r"\{.*\}", text, re.S)
        try:
            return json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            continue
    return None


# ---------------- prompt and answer ----------------
def build_prompt(numbered: list[tuple[int, str]]) -> str:
    return PROMPT.format(guesses="\n".join(f"{n}: {name}" for n, name in numbered))


def parse_answer(ans: dict | None) -> dict[int, dict]:
    """The model's JSON -> {number: {name, shape, movable, traits, skip}}. Anything unusable is dropped."""
    out = {}
    for o in (ans or {}).get("objects", []) if isinstance(ans, dict) else []:
        try:
            n = int(o["id"])
        except (KeyError, TypeError, ValueError):
            continue
        name = " ".join(str(o.get("name") or "").split())[:40]
        shape = o.get("shape") if o.get("shape") in catalog.SHAPES else None
        movable = o.get("movable") if isinstance(o.get("movable"), bool) else None
        traits = [t for t in (o.get("traits") or []) if t in TRAITS] if isinstance(o.get("traits"), list) else []
        out[n] = {"name": name or None, "shape": shape, "movable": movable, "traits": traits,
                  "skip": o.get("skip") is True}
    return out


def vote(entries: list[dict]) -> dict:
    """Combine what several photos said about one object.

    One photo alone never changes anything: a name or shape needs two photos that agree (and a clear winner),
    removal needs two skip votes and a majority. A weak model that invents an answer for one photo is ignored.
    """
    n = len(entries)
    skips = sum(e["skip"] for e in entries)

    def agreed(values: list):
        top = Counter(values).most_common(2)
        if top and top[0][1] >= 2 and (len(top) == 1 or top[0][1] > top[1][1]):
            return top[0][0]
        return None

    movs = [e["movable"] for e in entries if e["movable"] is not None]
    traits = Counter(t for e in entries for t in set(e["traits"]))
    return {"skip": skips >= 2 and skips * 2 > n,
            "name": agreed([e["name"].lower() for e in entries if e["name"]]),
            "shape": agreed([e["shape"] for e in entries if e["shape"]]),
            "movable": (sum(movs) * 2 > len(movs)) if len(movs) >= 2 and sum(movs) * 2 != len(movs) else None,
            "traits": sorted(t for t, c in traits.items() if c >= 2 and c * 2 >= n), "votes": n}


# ---------------- applying the verdict ----------------
def apply_review(scene: dict, verdicts: dict[str, dict], model: str | None = None) -> dict:
    """Update a scene in place. verdicts: {object id: voted entry}. Returns a short report.

    The reviewer can rename an object, change its shape, mark it unmovable, add traits or remove it.
    It cannot make something big movable: the catalog's size limits still apply.
    """
    kept, removed, renamed = [], [], []
    for o in scene["objects"]:
        v = verdicts.get(o["id"])
        if v is None:
            kept.append(o)
            continue
        if v["skip"]:
            removed.append({"id": o["id"], "class": o["class"]})
            continue
        name = v["name"] or o["label"]
        if name.lower() != o["label"].lower():
            renamed.append((o["label"], name))
        info = catalog.classify(name, o["size_x"], o["size_y"], o.get("height"))
        if not info["known"]:                                  # a name the catalog does not know: keep what we had
            info = catalog.classify(o["class"], o["size_x"], o["size_y"], o.get("height"))
        o["label"] = name
        o["shape"] = v["shape"] or info["shape"]
        o["surface"] = info["surface"]
        o["movable"] = bool(info["movable"] and v["movable"] is not False)
        o["traits"] = v["traits"]
        o["source"] = "nvidia"
        kept.append(o)
    scene["objects"] = kept
    schema._assign_support(kept)
    report = {"model": model, "reviewed": len(verdicts), "renamed": [list(r) for r in renamed], "removed": removed}
    scene["review"] = report
    schema.validate(scene)
    return report
