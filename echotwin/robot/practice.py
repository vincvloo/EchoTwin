"""Practising in the twin: varied tries on a fresh copy of the scanned table, and only the ones that work are kept as demos.

`Practice.step()` is called once per control tick while the robot's mode is "practice"; it returns the action to take.
"""
import numpy as np

from .dataset import state_vector
from .features import measure as M
from .features import move_things as MT
from .features import prop_skills as PS
from .world import CTRL_DT


class Practice:
    def __init__(self, sim, kinds: list[str], per_kind: int, prop: int | None):
        self.sim = sim
        self.kinds = kinds          # the size classes to practise, one after the other
        self.per = per_kind         # demos to keep per class
        self.prop = prop            # practise only this object, or None
        self.k = 0                  # index into kinds
        self.kept = 0
        self.tries = 0
        self.report: list[str] = []
        self.cur: dict | None = None

    def overlay(self) -> tuple[str, str | None]:
        """The two lines drawn over the video: progress, and what this try is."""
        kind = self.kinds[min(self.k, len(self.kinds) - 1)]
        label = f"PRACTICE {min(self.kept + 1, self.per)}/{self.per} · {PS.KIND_LABEL[kind]} · try {self.tries}"
        return label, self.cur["text"] if self.cur else None

    def _new_try(self):
        """Set up the next try on a fresh copy of the scanned table."""
        sim, rng = self.sim, self.sim.rng
        kind = self.kinds[self.k]
        props = sim.base_layout.props
        if self.prop is not None:
            movers = [self.prop]
        else:
            movers = [i for i in range(len(props)) if M.size_class(M.measure(sim.world, f"prop_{i}")) == kind]
        sim._rebuild(sim.base_layout)
        sim.world.settle(10)
        w = sim.world
        i = int(rng.choice(movers))
        others = [j for j in range(len(props)) if j != i]
        r = rng.random()
        if r < 0.55:
            goal = ("near", int(rng.choice(others)), "next to")
        elif r < 0.75:
            goal = ("near", int(rng.choice(others)), "on top of")
        else:
            goal = ("dir", [(-1, 0), (1, 0), (0, -1), (0, 1)][int(rng.integers(4))], 0.15)
        plan = {"prop": i, "goal": goal}
        text = f"put the {props[i]['name']} {MT.describe_goal(w, plan)}"
        task = PS.make_task(w, plan, text)
        style = {"grip": float(rng.uniform(-0.3, 0.3)), "lift": float(rng.uniform(0.03, 0.09)),
                 "drop": float(rng.uniform(0.002, 0.012)), "speed": float(rng.uniform(0.14, 0.25))}
        wps = PS.waypoints(w, task, style)
        self.tries += 1
        self.cur = {"task": task, "r": {"wps": wps, "i": 0, "speed": style["speed"], "yaw": w.grasp_yaw(task["object"])},
                    "frames": [], "settle": 0, "before": {k: w.obj_pos(k)[:2].copy() for k in w.things()}, "text": text}
        sim.task = task
        sim.ghost = np.array([wp[1] for wp in wps if wp[0] == "move"])
        sim.log("system", f"Practice {PS.KIND_LABEL[kind]} {self.kept + 1}/{self.per} (try {self.tries}): {text}")

    def step(self):
        sim, w = self.sim, self.sim.world
        idle = np.array([0, 0, 0, 0.0])
        if self.cur is None:
            self._new_try()
            return idle
        cur = self.cur
        a = PS.waypoint_action(w, cur["r"])
        if a is not None:
            cur["frames"].append({"timestamp": round(len(cur["frames"]) * CTRL_DT, 3),
                                  "state": state_vector(w, cur["task"]["object"], cur["task"]["goal"]),
                                  "action": [float(x) for x in a]})
            return a
        cur["settle"] += 1
        if cur["settle"] < 20:
            return idle
        # judge the try
        res = PS.outcome(w, cur["task"], cur["before"])
        ok = res["ok"] and PS.goal_met(w, cur["task"])
        if ok:
            sim.dataset.add(cur["task"], cur["frames"], "practice", True)
            self.kept += 1
        sim.log("system", "✓ kept" if ok else f"✗ {res['text']}")
        self.cur = None
        sim.ghost = None
        kind = self.kinds[self.k]
        if self.kept >= self.per or self.tries >= self.per * 4:
            self.report.append(f"{PS.KIND_LABEL[kind]}: {self.kept} of {self.tries} worked")
            self.k += 1
            self.kept = self.tries = 0
            if self.k >= len(self.kinds):
                sim.mode, sim.authority = "idle", "human"
                sim._rebuild(sim.base_layout)
                sim._refit()
                sim.say("I practised in my twin. " + "; ".join(self.report) + ".")
        return idle
