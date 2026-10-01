"""Task-conditioned policy learned from demonstrations (keypoint imitation).

From every demo we extract object- and target-relative keypoints:
  approach: hover height, grasp offset (hand - object at grip)        pooled over all targets
  carry:    carry height, release offset (hand - target at release)  per target: must be taught
  style:    transit speed
The robot replays those keypoints on the current layout with a smooth controller.

Uncertainty (0 = sure, >= ASK_THRESHOLD = lost) combines:
  novelty    how far this layout (object-to-target distance) is from the demonstrated ones
  spread     how inconsistent the demos were at the release point
  evidence   fewer demos -> less sure
and at run time: missed grasps, dropped blocks, no progress.
"""
from dataclasses import dataclass, field

import numpy as np

from .world import VMAX, World

ASK_THRESHOLD = 1.0
NOVELTY_SCALE = 0.4
MIN_APPROACH_Z = 0.06


@dataclass
class Keypoints:
    hover_z: float
    grasp_off: np.ndarray        # hand - object when the grip closed
    carry_z: float = 0.12
    release_off: np.ndarray = field(default_factory=lambda: np.zeros(3))  # hand - target at release
    speed: float = 0.2
    reach: float = 0.2           # object-to-target distance in that demo


def extract(frames: list[dict]) -> Keypoints | None:
    """Keypoints from one recorded episode, or None if it has no grasp."""
    S = np.array([f["state"] for f in frames])
    A = np.array([f["action"] for f in frames])
    carrying = S[:, 9] > 0.5
    if not carrying.any():
        return None
    g = int(np.argmax(carrying))                                  # first carrying frame
    r = g + int(np.argmin(carrying[g:])) if not carrying[g:].all() else len(S)  # first frame after carry
    hand, obj = S[:, 0:3], S[:, 4:7]
    tgt = np.c_[S[:, 7:9], np.zeros(len(S))]
    lateral = np.linalg.norm(hand[:g, :2] - obj[:g, :2], axis=1) if g else np.zeros(0)
    far = hand[:g][lateral > 0.05] if g else np.zeros((0, 3))
    hover = float(np.median(far[:, 2])) if len(far) else float(hand[max(g - 1, 0), 2] + 0.08)
    kp = Keypoints(hover_z=hover, grasp_off=hand[g] - obj[g], reach=float(np.linalg.norm(obj[0, :2] - tgt[0, :2])))
    carry = hand[g:r]
    kp.carry_z = float(np.percentile(carry[:, 2], 90)) if len(carry) else 0.12
    kp.release_off = hand[r - 1] - tgt[r - 1]
    speeds = np.linalg.norm(A[:, :3], axis=1)
    moving = speeds[speeds > 0.03]
    kp.speed = float(np.percentile(moving, 75)) if len(moving) else 0.15
    return kp


class Policy:
    def __init__(self):
        self.approach: list[Keypoints] = []
        self.carry: dict[str, list[Keypoints]] = {}
        self.demo_counts: dict[str, int] = {}

    def fit(self, episodes: list[dict]):
        self.approach, self.carry, self.demo_counts = [], {}, {}
        for ep in episodes:
            if not ep.get("success", True) or ep["task"].get("kind") == "prop":
                continue
            kp = extract(ep["frames"])
            if kp is None:
                continue
            tgt = ep["task"]["target"]
            if ep.get("source") != "correction":
                self.approach.append(kp)
                self.demo_counts[tgt] = self.demo_counts.get(tgt, 0) + 1
            self.carry.setdefault(tgt, []).append(kp)

    def knows(self, target: str) -> bool:
        return bool(self.carry.get(target))

    # ---------- the plan for one task ----------
    def plan(self, world: World, obj: str, target: str) -> dict | None:
        """Blend the demos into one set of keypoints for this layout (nearest demos weigh most)."""
        if not self.approach or not self.knows(target):
            return None
        o, t = world.obj_pos(obj), world.zone_pos(target)
        reach = float(np.linalg.norm(o[:2] - t))
        kps = self.carry[target]
        dr = np.array([abs(k.reach - reach) for k in kps])
        w = 1.0 / (dr + 0.05)
        w /= w.sum()
        h = world.half(obj)
        zone_half = np.array(world.layout.zones[target]["half"])
        rel = np.array([k.release_off for k in kps])
        release = (rel * w[:, None]).sum(0)
        release[:2] = np.clip(release[:2], -zone_half * 0.6, zone_half * 0.6)
        release[2] = float(np.clip(release[2], 0.0, 0.05))
        spread = float(np.linalg.norm(rel[:, :2].std(0)) / zone_half.min()) if len(kps) > 1 else 0.5
        novelty = float(dr.min() / NOVELTY_SCALE)
        evidence = 0.4 / len(kps)
        uncertainty = 0.15 + 0.5 * novelty + 0.5 * spread + evidence
        grasp = np.median([k.grasp_off for k in self.approach], axis=0)
        grasp[:2] = np.clip(grasp[:2], -h * 0.5, h * 0.5)
        grasp[2] = float(np.clip(grasp[2], -h * 0.3, h * 0.4))
        return {
            "hover_z": float(max(MIN_APPROACH_Z, np.median([k.hover_z for k in self.approach]))),
            "grasp": grasp,
            "carry_z": float(max(2 * h + 0.04, (np.array([k.carry_z for k in kps]) * w).sum())),
            "release": release,
            "speed": float(np.clip(np.median([k.speed for k in self.approach + kps]), 0.08, VMAX)),
            "uncertainty": uncertainty,
            "parts": {"novelty": round(novelty, 2), "spread": round(spread, 2), "demos": len(kps)},
        }

    def act(self, world: World, obj: str, target: str, plan: dict, d=None, hs=None):
        """-> (action(4), done). A small waypoint controller through the learned keypoints."""
        hs = hs if hs is not None else world.hand
        hand = world.hand_pos(d)
        spd = plan["speed"]

        def go(goal, grip, gain=5.0):
            v = (np.asarray(goal) - hand) * gain
            n = np.linalg.norm(v)
            if n > spd:
                v *= spd / n
            return np.array([*v, grip])

        if hs.attached == obj:
            t = world.zone_pos(target)
            goal = np.array([t[0], t[1], 0]) + plan["release"]
            goal[2] = max(goal[2], world.half(obj) + 0.004)
            dxy = np.linalg.norm(hand[:2] - goal[:2])
            if dxy > 0.012:
                if hand[2] < plan["carry_z"] - 0.025:
                    return go([hand[0], hand[1], plan["carry_z"]], 1), False
                return go([goal[0], goal[1], plan["carry_z"]], 1), False
            if hand[2] - goal[2] > 0.006:
                return go(goal, 1), False
            return np.array([0, 0, 0, 0.0]), False
        if world.in_zone(obj, target, d, hs):
            return np.array([0, 0, 0.15, 0.0]), True
        if hs.attached:
            return np.array([0, 0, 0, 0.0]), False  # wrong object: let go
        o = world.obj_pos(obj, d)
        goal = o + plan["grasp"]
        dxy = np.linalg.norm(hand[:2] - goal[:2])
        if hs.grip:                       # missed grasp: open and retry
            return np.array([0, 0, 0.05, 0.0]), False
        if dxy > 0.008:
            if dxy > 0.03 and hand[2] < plan["hover_z"] - 0.02:
                return go([hand[0], hand[1], plan["hover_z"]], 0), False
            return go([goal[0], goal[1], plan["hover_z"] if dxy > 0.03 else hand[2]], 0), False
        if hand[2] - goal[2] > 0.005:
            return go(goal, 0), False
        return np.array([0, 0, 0, 1.0]), False

    def rollout(self, world: World, obj: str, target: str, plan: dict, max_ticks: int = 400):
        """Imagine the run on a copy of the world: ghost path and predicted success."""
        d, hs = world.clone()
        path, success = [], False
        for _ in range(max_ticks):
            a, done = self.act(world, obj, target, plan, d, hs)
            if done:
                success = True
                break
            world.step(a, d, hs)
            path.append(world.hand_pos(d))
        return {"path": np.array(path).reshape(-1, 3), "success": success, "ticks": len(path)}


# ---------------- scripted expert (seed demos only) ----------------

class Expert:
    """Hand-written controller that creates the seed demonstrations for the green zone."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.gain = rng.uniform(4.0, 6.0)
        self.hover = rng.uniform(0.09, 0.13)
        self.drop = rng.uniform(-0.02, 0.02, 2)

    def act(self, world: World, obj: str, target: str):
        hs = world.hand
        hand = world.hand_pos()
        noise = self.rng.normal(0, 0.008, 3)
        clip = lambda v: np.clip(v, -VMAX, VMAX)
        if hs.attached == obj:
            t = world.zone_pos(target) + self.drop
            goal = np.array([t[0], t[1], self.hover])
            if np.linalg.norm(hand[:2] - goal[:2]) > 0.01:
                if hand[2] < self.hover - 0.03:
                    goal = np.array([hand[0], hand[1], self.hover])
                return np.array([*(clip((goal - hand) * self.gain) + noise), 1.0]), False
            low = np.array([t[0], t[1], world.half(obj) + 0.01])
            if hand[2] > low[2] + 0.006:
                return np.array([*clip((low - hand) * self.gain), 1.0]), False
            return np.array([0, 0, 0, 0.0]), False
        if world.in_zone(obj, target):
            return np.array([0, 0, 0.1, 0.0]), True
        o = world.obj_pos(obj)
        dxy = np.linalg.norm(hand[:2] - o[:2])
        if dxy > 0.008:
            goal = np.array([o[0], o[1], self.hover])
            return np.array([*(clip((goal - hand) * self.gain) + noise), 0.0]), False
        if hand[2] - o[2] > 0.005:
            return np.array([*clip((o - hand) * self.gain), 0.0]), False
        return np.array([0, 0, 0, 1.0]), False
