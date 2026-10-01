"""Language-conditioned demonstration dataset (LeRobot-style fields, stored as JSON).

Each episode: {episode_index, task:{instruction, object, target, goal, shape...}, source: human|practice|video,
frames:[{timestamp, state, action}], success, score, quality, duration}.
observation.state = [hand x,y,z, grip, object x,y,z, target x,y, carrying]
action            = [vx, vy, vz, grip]
"""
import json
import time
from pathlib import Path

import numpy as np

from .config import EPISODES
from .world import CTRL_DT, World

STATE_NAMES = ["hand.x", "hand.y", "hand.z", "grip", "obj.x", "obj.y", "obj.z", "goal.x", "goal.y", "carrying"]
ACTION_NAMES = ["vx", "vy", "vz", "grip"]
TELEOP_CHF_PER_HOUR = 60.0


def state_vector(world: World, obj: str, goal, d=None, hs=None) -> list[float]:
    hs = hs if hs is not None else world.hand
    h = world.hand_pos(d)
    o = world.obj_pos(obj, d)
    t = np.asarray(goal, dtype=float)
    return [*map(float, h), float(hs.grip), *map(float, o), float(t[0]), float(t[1]),
            1.0 if hs.attached == obj else 0.0]


def score_episode(frames: list[dict], success: bool) -> tuple[int, str]:
    if not success or len(frames) < 3:
        return 0, "failed"
    dur = len(frames) * CTRL_DT
    v = np.array([f["action"][:3] for f in frames])
    jerk = float(np.abs(np.diff(v, 2, axis=0)).mean()) if len(v) > 3 else 0.0
    speed = max(0.0, 1.0 - max(0.0, dur - 6.0) / 24.0)
    smooth = max(0.0, 1.0 - jerk / 0.03)
    score = int(round(55 + 25 * speed + 20 * smooth))
    return score, ("great" if score >= 85 else "good" if score >= 70 else "ok")


class Dataset:
    def __init__(self, root: Path = EPISODES):
        self.root = root
        self.episodes: list[dict] = []
        for p in sorted(root.glob("ep_*.json")):
            try:
                self.episodes.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                pass

    def next_index(self) -> int:
        return max((e["episode_index"] for e in self.episodes), default=-1) + 1

    def add(self, task: dict, frames: list[dict], source: str, success: bool) -> dict:
        score, quality = score_episode(frames, success)
        ep = {"episode_index": self.next_index(), "task": task, "source": source,
              "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "success": success, "score": score,
              "quality": quality, "duration": round(len(frames) * CTRL_DT, 2),
              "features": {"observation.state": STATE_NAMES, "action": ACTION_NAMES}, "frames": frames}
        (self.root / f"ep_{ep['episode_index']:06d}.json").write_text(json.dumps(ep), encoding="utf-8")
        self.episodes.append(ep)
        return ep

    def stats(self) -> dict:
        per_target, per_source = {}, {}
        for e in self.episodes:
            t = e["task"]["target"]
            per_target[t] = per_target.get(t, 0) + 1
            per_source[e["source"]] = per_source.get(e["source"], 0) + 1
        human = [e for e in self.episodes if e["source"] in ("human", "correction", "video")]
        mins = sum(e["duration"] for e in human) / 60
        return {
            "episodes": len(self.episodes), "per_target": per_target, "per_source": per_source,
            "frames": sum(len(e["frames"]) for e in self.episodes),
            "human_minutes": round(mins, 2),
            "teleop_cost_chf": round(mins / 60 * TELEOP_CHF_PER_HOUR, 2),
            "recent": [{"i": e["episode_index"], "instruction": e["task"]["instruction"], "source": e["source"],
                        "score": e["score"], "quality": e["quality"], "kind": e["task"].get("kind", "prop")}
                       for e in self.episodes[-30:]][::-1],
        }
