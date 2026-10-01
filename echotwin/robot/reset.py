"""Clear taught demos before a run-through.

    python -m echotwin.robot.reset
"""
from .config import EPISODES

if __name__ == "__main__":
    n = 0
    for p in EPISODES.glob("ep_*.json"):
        p.unlink()
        n += 1
    print(f"removed {n} episode(s). Restart the server.")
