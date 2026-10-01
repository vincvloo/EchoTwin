r"""Clear taught demos before a run-through: keeps the green-zone seed demos, deletes human and correction ones.

    python -m echotwin.robot.reset          (keep seeds)
    python -m echotwin.robot.reset --all    (delete everything; seeds are re-created on start)
"""
import json
import sys

from .config import EPISODES

if __name__ == "__main__":
    wipe_all = "--all" in sys.argv
    n = 0
    for p in EPISODES.glob("ep_*.json"):
        if wipe_all or json.loads(p.read_text(encoding="utf-8")).get("source") != "seed":
            p.unlink()
            n += 1
    print(f"removed {n} episode(s). Restart the server.")
