"""A record of every move the robot made, by back-end, so sim and real can be compared on the same kind of task.

One JSON line per move in `data/robot/runs.jsonl` (next to the demos): which back-end ran it, what the object measured,
whether it worked and why not. `python -m echotwin.robot.runlog` prints the success rate per back-end and size class.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from . import config
from .features import measure as M

PATH = config.DATA / "runs.jsonl"


def record(backend: str, task: dict, res: dict, path: Path = PATH, seconds: float | None = None,
           attempts: int | None = None) -> dict:
    """Append one move. `res` is what `prop_skills.outcome` returned ({ok, text, err, tilt, moved})."""
    m = M.from_task(task)
    row = {"time": round(time.time()), "backend": backend, "object": task.get("name"), "class": M.size_class(m),
           "width": round(m["width"], 3), "height": round(m["height"], 3), "length": round(m["length"], 3),
           "instruction": task.get("instruction"), "ok": bool(res.get("ok")), "why": "" if res.get("ok") else res.get("text", ""),
           "err_cm": round(100 * float(res.get("err", 0.0)), 1), "seconds": None if seconds is None else round(seconds, 1), "attempts": attempts}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return row


def read(path: Path = PATH) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    return rows


def summary(rows: list[dict]) -> dict:
    """{backend: {class or 'all': (moves, successes)}}"""
    out: dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        for key in (r["class"], "all"):
            c = out[r["backend"]][key]
            c[0] += 1
            c[1] += int(r["ok"])
    return {b: {k: tuple(v) for k, v in d.items()} for b, d in out.items()}


def table(rows: list[dict]) -> str:
    s = summary(rows)
    if not s:
        return "No moves recorded yet."
    classes = [c for c in (*M.CLASSES, "all") if any(c in d for d in s.values())]
    lines = ["| Back-end | " + " | ".join(classes) + " |", "|---|" + "---|" * len(classes)]
    for b, d in sorted(s.items()):
        lines.append(f"| {b} | " + " | ".join(f"{d[c][1]}/{d[c][0]}" if c in d else "-" for c in classes) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    print(table(read()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
