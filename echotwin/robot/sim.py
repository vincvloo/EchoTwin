"""The robot: sim loop, shared control, recording, and the 'done, do, or teach me' behaviour.

Everything that touches MuJoCo runs in this one thread (OpenGL contexts are thread-bound).
Other threads talk to it through submit().
"""
import queue
import threading
import time
import traceback

import cv2
import mujoco
import numpy as np

from . import backend as BK
from . import config
from . import runlog
from .dataset import Dataset, score_episode, state_vector
from .features import prop_skills as PS
from .features import measure as M
from .features import tasks as T
from .router import Intent
from .scene import Layout

from .voice import Voice
from .world import CTRL_DT, VMAX, World

W, H = 800, 600
HUMAN_TIMEOUT = 0.5


class Sim:
    def __init__(self, emit):
        self.emit = emit
        self.voice = Voice(emit)
        self.world, self.backend_note = BK.make()
        self.base_layout = self.world.layout.copy()
        self.rng = np.random.default_rng()
        self.dataset = Dataset()
        self.skills = PS.PropSkills()
        self.skills.fit(self.dataset.episodes)
        self._stats = self.dataset.stats()

        self.cmds: queue.Queue = queue.Queue()
        self.human_v = np.zeros(3)
        self.human_t = 0.0
        self.human_grip = False
        self.authority = "human"      # who drives: human | robot
        self.mode = "idle"            # idle | teach | review | move | replay | practice
        self.halted = False
        self.halt_reason = ""
        self.task: dict | None = None
        self.plan: dict | None = None
        self.ghost: np.ndarray | None = None
        self.frames: list[dict] = []
        self.review: dict | None = None
        self.settled = 0
        self.robot_runs = {"runs": 0, "success": 0, "asked": 0}
        self.scan: dict | None = None
        self.replay: dict | None = None
        self.pending_move: dict | None = None
        self.jpeg: bytes | None = None
        self.frame_id = 0
        self.tick = 0
        self._renderer = None

    # ---------------- plumbing ----------------
    def submit(self, fn, *args):
        self.cmds.put((fn, args))

    def start(self):
        threading.Thread(target=self._run, daemon=True, name="sim").start()

    def _run(self):
        self._renderer = mujoco.Renderer(self.world.model, H, W)
        nxt = time.perf_counter()
        while True:
            while not self.cmds.empty():
                fn, args = self.cmds.get()
                try:
                    fn(*args)
                except Exception:
                    traceback.print_exc()
            try:
                self._tick()
                if self.tick % 2 == 0:
                    self._render()
                if self.tick % 4 == 0:
                    self._emit_state()
            except Exception:
                traceback.print_exc()
            self.tick += 1
            nxt += CTRL_DT
            delay = nxt - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.5:
                nxt = time.perf_counter()

    def say(self, text: str):
        self.voice.say(text)

    def log(self, who: str, text: str):
        self.emit({"t": "log", "who": who, "text": text})

    def haptic(self, pattern):
        self.emit({"t": "haptic", "pattern": pattern, "to": "phone"})

    # ---------------- world changes ----------------
    def _rebuild(self, layout: Layout):
        self.world.build(layout)
        self.world.settle(5)
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = mujoco.Renderer(self.world.view.model, H, W)
        self.human_grip = False
        self.ghost = None

    def _abort(self):
        self.mode, self.frames, self.review, self.plan, self.ghost = "idle", [], None, None, None
        self.authority = "human"

    def reset_scene(self, announce=True):
        self._abort()
        self._rebuild(self.base_layout)
        if announce:
            self.say("Back to your table." if self.scan else "Scene reset.")

    def apply_scan(self, layout: Layout, summary: dict):
        self._abort()
        self.halted = False
        try:
            self._rebuild(layout)
        except ValueError as e:  # e.g. a mesh MuJoCo cannot read: keep the current twin
            self.log("system", f"Could not load that twin: {e}")
            return self.say("I couldn't load that twin. The current table stays as it is.")
        self.base_layout = layout.copy()
        self.scan = summary
        self.say(summary["greeting"])

    def name_props(self, props: list[dict], line: str):
        """The AI named the everyday objects (and their shapes) after the twin was already shown."""
        lay = self.world.layout.copy()
        lay.props = [dict(p) for p in props]
        self.base_layout = lay.copy()
        self._rebuild(lay)
        self.say(line)

    # ---------------- human input ----------------
    def set_human(self, vx, vy, vz):
        v = np.array([vx, vy, vz], dtype=float)
        n = np.linalg.norm(v)
        self.human_v = v if n <= 1 else v / n
        self.human_t = time.time()

    def set_grip(self, on: bool):
        self.human_grip = bool(on)

    # ---------------- safety ----------------
    def halt(self, reason: str):
        self.halted = True
        self.halt_reason = reason
        self.world.stop()                                   # a real arm: torque off
        self.voice.hush()
        self.say("Stopped.")
        self.haptic([400])
        self.log("system", f"STOPPED ({reason})")

    def resume(self):
        if not self.halted:
            self.say("I'm not stopped.")
            return
        self.halted = False
        self.world.resume()
        self.say("Continuing.")

    # ---------------- the tick ----------------
    def _tick(self):
        w = self.world
        action = np.array([0, 0, 0, 1.0 if w.hand.grip else 0.0])
        recording = False
        if self.halted:
            pass
        elif self.authority == "robot" and self.mode == "practice":
            action = self._practice_step()
        elif self.authority == "robot" and self.mode in ("replay", "move"):
            action = self._replay_step()
            recording = self.mode == "replay"
        elif self.authority == "human":
            fresh = time.time() - self.human_t < HUMAN_TIMEOUT
            action = np.array([*(self.human_v * VMAX if fresh else np.zeros(3)), 1.0 if self.human_grip else 0.0,
                               w.auto_yaw(w.hand_pos())])  # the wrist turns to the nearest object by itself
            if self.mode == "teach":
                moving = np.linalg.norm(action[:3]) > 0.01 or action[3] != float(w.hand.grip)
                recording = bool(self.frames) or moving
        if recording:
            t = self.task
            self.frames.append({"timestamp": round(len(self.frames) * CTRL_DT, 3),
                                "state": state_vector(w, t["object"], t["goal"]),
                                "action": [float(x) for x in action]})
        w.step(action)
        if self.mode == "teach" and not self.halted:
            if PS.goal_met(w, self.task) and not w.hand.grip:
                self.settled += 1
                if self.settled >= 6:
                    self._demo_done()
            else:
                self.settled = 0

    def _demo_done(self):
        self.settled = 0
        frames, task = self.frames, self.task
        self.frames = []
        score, quality = score_episode(frames, True)
        self.review = {"task": task, "frames": frames, "score": score, "quality": quality}
        self.mode = "review"
        self.say(f"Nice. That scored {score}. Keep this demo?")
        self.haptic([60, 60, 60])

    def keep(self, keep: bool):
        if self.mode != "review" or not self.review:
            self.say("There's no demo to keep." if keep else "Nothing to discard.")
            return
        rv, self.review = self.review, None
        self.mode = "idle"
        if not keep:
            self.say("Discarded.")
            self._rebuild(self.base_layout)
            return
        self.dataset.add(rv["task"], rv["frames"], rv.get("source", "human"), True)
        self._refit()
        self._rebuild(self.base_layout)
        self.say(f"Thanks. Now I know how to handle {M.CLASS_LABEL[M.size_class(M.from_task(rv['task']))]}. Try me again.")

    def _refit(self):
        self.skills.fit(self.dataset.episodes)
        self._stats = self.dataset.stats()
        self.emit({"t": "dataset", "stats": self._stats})

    # ---------------- replaying a move ----------------
    def _replay_step(self):
        a = PS.waypoint_action(self.world, self.replay)
        if a is None:
            if self.mode == "move":  # let the object settle before judging
                self.replay["settle"] = self.replay.get("settle", 0) + 1
                if self.replay["settle"] < 20:
                    return np.array([0, 0, 0, 0.0])
            self._replay_done()
            return np.array([0, 0, 0, 0.0])
        return a

    # ---------------- everyday objects: done, do, or teach me (per object type) ----------------
    def handle_prop_task(self, plan: dict, heard: str):
        from .features import move_things as MT
        w = self.world
        props = w.layout.props
        i = plan["prop"]
        name, shape = props[i]["name"], props[i].get("shape", "box")
        if plan["goal"] is None:
            self.pending_move = plan
            return self.say(f"Where should I put the {name}? For example: next to another object, or to the left.")
        self.pending_move = None
        self.halted = False
        self._abort()
        task, goal, o = self._make_prop_task(plan, heard)
        me = task["object"]
        self.task = task
        where = MT.describe_goal(w, plan)
        # 1. already done?
        if PS.goal_met(w, task):
            self.emit({"t": "decision", "kind": "done", "task": task, "why": "checked the table: it is already there"})
            return self.say(f"Already done: the {name} is {where}.")
        if not w.arm_ready():
            return self.say("The real arm is waiting. Say or press 'arm the robot' to let it move, then ask again.")
        # 1b. can my arm do it at all?
        why_not = w.refusal(me, goal)
        if why_not:
            self.emit({"t": "decision", "kind": "refuse", "task": task, "why": why_not})
            return self.say(f"I can't move the {name}: {why_not}.")
        # 2. do I know how to handle this kind of object?
        m = task["m"]
        n = self.skills.count(m)
        if n < PS.MIN_PROP_DEMOS:
            return self._start_teach(task, reason="new_kind", why=f"no demos with something this size yet",
                                     line=f"I've never moved something this size: {M.describe(m)}. Can you show me? "
                                          "Drive me with the keyboard, upload a video, or say practice.")
        # 3. plan from what I learned, then imagine it on a copy of the world
        skill = self.skills.plan(m, float(np.linalg.norm(goal - o[:2])))
        wps = PS.waypoints(w, task, skill)
        im = PS.imagine(w, task, wps, skill["speed"])
        u = skill["uncertainty"] + (0 if im["ok"] else 0.8)
        why = f"{n} demo{'s' if n != 1 else ''} of a similar size · imagined: {im['text']}"
        if not im["ok"] or u >= PS.ASK:
            return self._start_teach(task, reason="unsure", why=why,
                                     line=f"I tried it in my head: {im['text']}. Can you show me how?")
        self.plan = {"uncertainty": u, "parts": skill["parts"], "why": why}
        self.ghost = im["path"]
        self.replay = {"wps": wps, "i": 0, "speed": skill["speed"], "task": task, "yaw": w.grasp_yaw(task["object"]),
                       "before": {k: w.obj_pos(k)[:2].copy() for k in w.things()}}
        self.mode, self.authority = "move", "robot"
        self.emit({"t": "decision", "kind": "do", "task": task, "uncertainty": round(u, 2), "sure": T.percent_sure(u),
                   "parts": skill["parts"], "why": why})
        self.say(f"I'll move the {name} {where}. {T.confidence_words(u)}")

    move_prop = handle_prop_task

    def practice(self, shape: str | None = None, per_kind: int = 3, live: bool = True, prop: int | None = None):
        """Create demos by practising in the twin: varied tries, only successes are kept.
        live=True runs in real time on screen, try by try; live=False runs instantly (tests)."""
        self._abort()
        props = self.base_layout.props
        if len(props) < 2:
            return self.say("I need at least two things on the table to practise.")
        classes = [M.size_class(M.measure(self.world, f"prop_{i}")) for i in range(len(props))]
        kinds = [shape] if shape else [c for c in M.CLASSES if c in classes]
        if prop is not None:
            kinds = [classes[prop]]
        self.practice_state = {"kinds": kinds, "k": 0, "kept": 0, "tries": 0, "per": per_kind, "report": [],
                               "cur": None, "prop": prop}
        self.halted = False
        self.mode, self.authority = "practice", "robot"
        if not live:
            while self.mode == "practice":
                self.world.step(self._practice_step())

    def _practice_new_try(self):
        """Set up the next try on a fresh copy of the scanned table."""
        from .features import move_things as MT
        st, rng = self.practice_state, self.rng
        kind = st["kinds"][st["k"]]
        props = self.base_layout.props
        movers = [st["prop"]] if st.get("prop") is not None else             [i for i in range(len(props)) if M.size_class(M.measure(self.world, f"prop_{i}")) == kind]
        self._rebuild(self.base_layout)
        self.world.settle(10)
        w = self.world
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
        task, _, _ = self._make_prop_task(plan, text)
        style = {"grip": float(rng.uniform(-0.3, 0.3)), "lift": float(rng.uniform(0.03, 0.09)),
                 "drop": float(rng.uniform(0.002, 0.012)), "speed": float(rng.uniform(0.14, 0.25))}
        wps = PS.waypoints(w, task, style)
        st["tries"] += 1
        st["cur"] = {"task": task, "r": {"wps": wps, "i": 0, "speed": style["speed"], "yaw": w.grasp_yaw(task["object"])}, "frames": [], "settle": 0,
                     "before": {k: w.obj_pos(k)[:2].copy() for k in w.things()}, "text": text}
        self.task = task
        self.ghost = np.array([wp[1] for wp in wps if wp[0] == "move"])
        self.log("system", f"Practice {PS.KIND_LABEL[kind]} {st['kept'] + 1}/{st['per']} (try {st['tries']}): {text}")

    def _practice_step(self):
        st = self.practice_state
        w = self.world
        if st["cur"] is None:
            self._practice_new_try()
            return np.array([0, 0, 0, 0.0])
        cur = st["cur"]
        a = PS.waypoint_action(w, cur["r"])
        if a is not None:
            cur["frames"].append({"timestamp": round(len(cur["frames"]) * CTRL_DT, 3),
                                  "state": state_vector(w, cur["task"]["object"], cur["task"]["goal"]),
                                  "action": [float(x) for x in a]})
            return a
        cur["settle"] += 1
        if cur["settle"] < 20:
            return np.array([0, 0, 0, 0.0])
        # judge the try
        res = PS.outcome(w, cur["task"], cur["before"])
        ok = res["ok"] and PS.goal_met(w, cur["task"])
        if ok:
            self.dataset.add(cur["task"], cur["frames"], "practice", True)
            st["kept"] += 1
        self.log("system", "✓ kept" if ok else f"✗ {res['text']}")
        st["cur"] = None
        self.ghost = None
        kind = st["kinds"][st["k"]]
        if st["kept"] >= st["per"] or st["tries"] >= st["per"] * 4:
            st["report"].append(f"{PS.KIND_LABEL[kind]}: {st['kept']} of {st['tries']} worked")
            st["k"] += 1
            st["kept"] = st["tries"] = 0
            if st["k"] >= len(st["kinds"]):
                self.mode, self.authority = "idle", "human"
                self._rebuild(self.base_layout)
                self._refit()
                self.say("I practised in my twin. " + "; ".join(st["report"]) + ".")
        return np.array([0, 0, 0, 0.0])

    def _make_prop_task(self, plan: dict, heard: str):
        from .features import move_things as MT
        w = self.world
        i = plan["prop"]
        pr = w.layout.props[i]
        me = f"prop_{i}"
        goal = MT.goal_xy(w, plan)
        o = w.obj_pos(me)
        shape = pr.get("shape", "box")
        task = {"kind": "prop", "plan": plan, "object": me, "target": f"{shape} things", "shape": shape,
                "name": pr["name"], "instruction": heard, "goal": [float(goal[0]), float(goal[1])],
                "ref": f"prop_{plan['goal'][1]}" if plan["goal"][0] == "near" else None,
                "h": float(w.half(me)), "tallest": float(w.tallest()), "m": M.measure(w, me),
                "start": [float(o[0]), float(o[1])]}
        if plan["goal"][0] == "near" and plan["goal"][2] == "on top of":
            task["stack"] = True
            task["ref_name"] = w.layout.props[plan["goal"][1]]["name"]
        return task, goal, o

    def play_prop_video(self, plan: dict, heard: str):
        """Replay a move seen in a video of your hand, with the robot, in the twin; record it as a demo."""
        from .features import move_things as MT
        self.halted = False
        self._abort()
        name = self.world.layout.props[plan["prop"]]["name"]
        task, goal, o = self._make_prop_task(plan, f"put the {name} {MT.describe_goal(self.world, plan)}")
        self.task = task
        wps = PS.waypoints(self.world, task, dict(PS.DEFAULTS))
        self.replay = {"wps": wps, "i": 0, "speed": 0.15, "task": task, "yaw": self.world.grasp_yaw(task["object"])}
        self.mode, self.authority, self.frames = "replay", "robot", []
        self.emit({"t": "decision", "kind": "teach", "task": task, "reason": "video",
                   "why": f"learning {M.CLASS_LABEL[M.size_class(task['m'])]} from your video"})
        self.say(heard)

    def _replay_done(self):
        if self.mode == "move":
            r, w = self.replay, self.world
            t = r["task"]
            self.mode, self.authority, self.ghost = "idle", "human", None
            res = PS.outcome(w, t, r["before"])
            runlog.record(w.name, t, res, self.dataset.root.parent / "runs.jsonl")
            self.robot_runs["runs"] += 1
            if res["ok"]:
                self.robot_runs["success"] += 1
                return self.say("Done.")
            # it went wrong for real: ask to be shown, from where things are now
            self.robot_runs["asked"] += 1
            return self._start_teach(t, reason="failed", why=f"real run: {res['text']}",
                                     line=f"That didn't go well: {res['text']}. Can you show me how?")
        t = self.task
        frames, self.frames = self.frames, []
        ok = PS.goal_met(self.world, t)
        if not ok:
            self.mode, self.authority = "idle", "human"
            return self.say("I couldn't reproduce your video in my twin. Can you show me with the phone?")
        score, quality = score_episode(frames, True)
        self.review = {"task": t, "frames": frames, "score": score, "quality": quality, "source": "video"}
        self.mode, self.authority = "review", "human"
        self.say(f"That worked in my twin. Keep this demo?")
        self.haptic([60, 60, 60])

    # ---------------- done, do, or teach me ----------------
    def _start_teach(self, task, reason="new", announce=True, line=None, why=None):
        self.task = task
        self.mode, self.authority = "teach", "human"
        self.frames, self.ghost, self.plan = [], None, None
        self.settled = 0
        self.human_grip = self.world.hand.grip
        self.emit({"t": "decision", "kind": "teach", "task": task, "reason": reason, "why": why})
        if announce and line:
            self.say(line)
        elif announce:
            self.say({
                "new": "I haven't learned that yet. Can you show me?",
                "more": "I've only seen that once. Can you show me again?",
                "unsure": "I'm not sure about this layout. Can you show me?",
            }[reason])
        self.haptic([80, 40, 80])

    # ---------------- utterances ----------------
    def handle_intent(self, it: Intent):
        k, n = it.kind, it.name
        if k == "safety":
            return self.halt("you said stop")
        if k == "control":
            return self._control(n)
        if k == "robot_q":
            return self.say(self.answer_robot(n))
        if k == "scene_q":
            return self.say(self.answer_scene(n, {**it.data, "text": it.text}))
        self.say("Tell me what to move on the table, like: put the glass next to the chocolate.")

    def _control(self, n):
        w = self.world
        if n == "continue":
            return self.resume()
        if n == "arm_robot":
            self.world.enable()
            return self.say("The arm may move now." if self.world.name == "real" else "There is no real arm to arm.")
        if n in ("keep", "yes") and self.mode == "review":
            return self.keep(True)
        if n in ("discard", "no") and self.mode == "review":
            return self.keep(False)
        if n in ("keep", "discard", "yes", "no"):
            if self.mode == "teach" and n in ("discard", "no"):
                self._abort()
                return self.say("Okay, I stopped recording.")
            return self.say("Okay.")
        if n == "robot_turn":
            if self.task:
                return self.handle_prop_task(self.task["plan"], self.task["instruction"])
            return self.say("Tell me what to do first.")
        if n == "human_turn":
            self.authority = "human"
            self.human_grip = w.hand.grip
            return self.say("Your turn.")
        if n == "grip":
            self.human_grip = True
            return self.say("Gripping.")
        if n == "release":
            self.human_grip = False
            return self.say("Releasing.")
        if n == "reset":
            return self.reset_scene()
        if n == "home":
            if not w.hand.held:
                w.go_rest()
            return self.say("Home.")

    # ---------------- answers from state (no AI) ----------------
    def answer_robot(self, n) -> str:
        name = config.ROBOT_NAME
        if n == "why_stop":
            if self.halted:
                return {"you said stop": "You told me to stop.", "shake": "You shook the phone.",
                        "button": "You pressed stop."}.get(self.halt_reason, f"Because {self.halt_reason}.")
            return "I haven't stopped."
        if n == "doing":
            if self.halted:
                return "Nothing. I'm stopped. Say continue when ready."
            t = self.task
            return {
                "move": f"Moving the {t['name']}." if t else "Working.",
                "teach": f"Watching you. You're showing me: {t['instruction']}." if t else "Watching you.",
                "review": "Waiting for you to keep or discard the demo.",
                "replay": "Replaying your video in my twin.",
                "practice": "Practising in my twin.",
            }.get(self.mode, "Waiting for an instruction.")
        props = self.world.layout.props
        if not props:
            return "Scan your table first, so I know what is on it."
        names = [p["name"] for p in props]
        if n == "sure":
            t = self.task
            if self.plan and t:
                return (f"About {T.percent_sure(self.plan['uncertainty'])} percent. "
                        f"{self.plan.get('why', '').split(' · ')[0]}.")
            if t:
                c = self.skills.count(M.from_task(t))
                return f"Not sure yet. I have {c} demo{'s' if c != 1 else ''} of something this size."
            return "Tell me what to move, and I'll tell you how sure I am."
        if n == "learned":
            known = [f"{PS.KIND_LABEL[k]} from {c} demo{'s' if c != 1 else ''}" for k, c in self.skills.counts().items() if c]
            if not known:
                return "Nothing on this table yet. Tell me to move something and show me once."
            return "I know how to move " + " and ".join(known) + "."
        if n == "who":
            return f"I'm {name}. Tell me what to move on your table, and where."
        if n == "help":
            other = names[1] if len(names) > 1 else "the table's middle"
            return f"Try: put the {names[0]} next to the {other}. Or say stop, anytime."
        return "I'm not sure."

    def where_prop(self, i: int) -> str:
        w = self.world
        name = w.layout.props[i]["name"]
        me = f"prop_{i}"
        if w.hand.attached == me:
            return f"I'm holding the {name}."
        p = w.obj_pos(me)
        lr = "left" if p[0] < -0.1 else "right" if p[0] > 0.1 else "middle"
        fb = "front" if p[1] < -0.08 else "back" if p[1] > 0.08 else "centre"
        near = [(np.linalg.norm(w.obj_pos(f"prop_{j}")[:2] - p[:2]) - w.radius(me) - w.radius(f"prop_{j}"), j)
                for j in range(len(w.layout.props)) if j != i]
        close = min(near) if near else None
        extra = f", next to the {w.layout.props[close[1]]['name']}" if close and close[0] < 0.06 else ""
        spot = f"at the {fb} {lr}" if lr != "middle" else f"in the {fb} middle"
        return f"The {name} is {spot} of the table{extra}."

    def answer_scene(self, n, data) -> str:
        w = self.world
        props = w.layout.props
        if n == "which":
            k = data.get("index")
            if not props:
                return "I haven't mapped any objects. Snap a photo of your table first."
            if k is None or not 1 <= k <= len(props):
                return f"I have {len(props)} objects, numbered 1 to {len(props)}."
            return f"Object {k} is {props[k - 1]['name']}."
        if not props:
            return "I don't see anything yet. Snap a photo of your table first."
        if n == "count":
            return f"I see {len(props)} thing{'s' if len(props) != 1 else ''} on the table."
        if n == "where":
            from .features import move_things as MT
            ment = MT._mentions(data.get("text", ""), props)
            if ment:
                return self.where_prop(ment[0][1])
            return " ".join(self.where_prop(i) for i in range(len(props)))
        return self.scene_summary()

    def scene_summary(self) -> str:
        things = [p["name"] for p in self.world.layout.props]
        if not things:
            return "I don't see anything on the table yet."
        from .features.everyday import listing
        return f"On your table I see {listing(things)}."

    # ---------------- output ----------------
    def _render(self):
        r = self._renderer
        view = self.world.view
        r.update_scene(view.data, camera="photo" if view.layout.view else "main")
        if self.ghost is not None and len(self.ghost):
            scn = r.scene
            u = self.plan["uncertainty"] if self.plan else 0.5
            rgba = np.array([0.2, 0.9, 0.5, 0.55] if u < 0.45 else [1.0, 0.75, 0.2, 0.55] if u < 0.75
                            else [1.0, 0.35, 0.3, 0.55], dtype=np.float32)
            for p in self.ghost[::3]:
                if scn.ngeom >= scn.maxgeom:
                    break
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                    np.array([0.005, 0, 0]), p.astype(np.float64), np.eye(3).flatten(), rgba)
                scn.ngeom += 1
        img = r.render()
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if self.halted:
            cv2.rectangle(bgr, (0, 0), (W - 1, H - 1), (40, 40, 230), 10)
            cv2.putText(bgr, f"STOPPED ({self.halt_reason})", (24, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                        (40, 40, 230), 3, cv2.LINE_AA)
        elif self.mode == "practice" and getattr(self, "practice_state", None):
            st = self.practice_state
            kind = st["kinds"][min(st["k"], len(st["kinds"]) - 1)]
            label = f"PRACTICE {min(st['kept'] + 1, st['per'])}/{st['per']} · {PS.KIND_LABEL[kind]} · try {st['tries']}"
            cv2.putText(bgr, label, (20, H - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(bgr, label, (20, H - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 120), 2, cv2.LINE_AA)
            cur = st.get("cur")
            if cur:
                cv2.putText(bgr, cur["text"], (20, H - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(bgr, cur["text"], (20, H - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        elif self.mode == "teach" and self.frames:
            cv2.circle(bgr, (30, 34), 11, (40, 40, 230), -1)
            cv2.putText(bgr, "REC", (50, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 230), 2, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            self.jpeg = buf.tobytes()
            self.frame_id += 1

    def _emit_state(self):
        w = self.world
        t = self.task
        self.emit({
            "t": "state",
            "mode": self.mode, "authority": self.authority, "halted": self.halted, "halt_reason": self.halt_reason,
            "task": t,
            "grip": w.hand.grip, "holding": w.hand.attached, "hand": [round(float(x), 4) for x in w.hand_pos()],
            "recording": len(self.frames) if self.mode in ("teach", "replay") else 0,
            "review": {"score": self.review["score"], "quality": self.review["quality"]} if self.review else None,
            "plan": {"uncertainty": round(self.plan["uncertainty"], 2), "sure": T.percent_sure(self.plan["uncertainty"]),
                     "parts": self.plan["parts"], "why": self.plan.get("why")} if self.plan else None,
            "skills": self.skills.counts() if w.layout.props else None,
            "props": [{"name": pr["name"], "shape": pr.get("shape"), "pos": [round(float(v), 4) for v in w.obj_pos(f"prop_{i}")],
                       "h": pr["size"][2]} for i, pr in enumerate(w.layout.props)],
            "skill_labels": PS.KIND_LABEL,
            "demos": self.skills.counts(), "min_demos": PS.MIN_PROP_DEMOS,
            "backend": {"name": w.name, "note": self.backend_note, "ready": w.arm_ready()},
            "robot_runs": self.robot_runs, "scanned": bool(self.scan), "name": config.ROBOT_NAME,
        })
