"""The robot: sim loop, shared control, recording, and the 'done, do, or teach me' behaviour.

Everything that touches MuJoCo runs in this one thread (OpenGL contexts are thread-bound).
Other threads talk to it through submit().
"""
import queue
import threading
import time
import traceback

import numpy as np

from . import answers
from . import arm as A
from . import robots as RB
from . import backend as BK
from . import policy as POL
from . import config
from . import runlog
from .dataset import Dataset, score_episode, state_vector
from .features import everyday as EV
from .features import measure as M
from .features import move_things as MT
from .features import prop_skills as PS
from .features import tasks as T
from .practice import Practice
from .render import FrameRenderer
from .router import Intent
from .scene import Layout
from .voice import Voice
from .world import CTRL_DT, VMAX

HUMAN_TIMEOUT = 0.5


class Sim:
    def __init__(self, emit):
        self.emit = emit
        self.voice = Voice(emit)
        self.world, self.backend_note = BK.make()
        self.policy, self.policy_note = POL.from_env()
        self.base_layout = self.world.layout.copy()
        self.rng = np.random.default_rng()
        self.dataset = Dataset()
        self.skills = PS.PropSkills()
        self.skills.fit(self.dataset.episodes)
        self._stats = self.dataset.stats()

        self.cmds: queue.Queue = queue.Queue()
        self.human_v = np.zeros(3)
        self.human_t = 0.0
        self.base_cmd = np.zeros(3)       # driving a mobile base by hand: forward, sideways, turn (-1..1)
        self.base_t = 0.0
        self.base_blocked_t = 0.0
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
        self._renderer: FrameRenderer | None = None
        self._practice: Practice | None = None

    # ---------------- plumbing ----------------
    def submit(self, fn, *args):
        self.cmds.put((fn, args))

    def start(self):
        threading.Thread(target=self._run, daemon=True, name="sim").start()

    def _run(self):
        self._renderer = FrameRenderer(self.world.model)
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
            self._renderer.reopen(self.world.view.model)
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

    robot_choice: str | None = None     # a robot the user picked; None: the one that fits the surface (robots.for_surface)

    def _robot_for(self, layout: Layout):
        """Put the robot that fits where the things are, unless the user chose one (simulation only)."""
        if self.world.name != "sim" or self.robot_choice is not None:
            return
        name = RB.for_surface(layout.surface)
        if self.world.robot.name != name:
            self.world.robot = RB.load(name)

    def apply_scan(self, layout: Layout, summary: dict):
        self._abort()
        self.halted = False
        old = self.world.robot
        self._robot_for(layout)
        try:
            self._rebuild(layout)
        except ValueError as e:  # e.g. a mesh MuJoCo cannot read: keep the current twin
            self.world.robot = old
            self.log("system", f"Could not load that twin: {e}")
            return self.say("I couldn't load that twin. The current table stays as it is.")
        self.base_layout = layout.copy()
        self.scan = summary
        self.say(summary["greeting"])

    def set_scale(self, k: float, line: str | None = None):
        """The twin k times bigger or smaller (the arm stays): when the sizes the photo gave are wrong, or to make Pip
        look bigger (k below 1) or smaller (k above 1) next to the table."""
        if self.world.name != "sim":
            return self.say("I can only rescale the simulation, not the real arm's twin.")
        why = self.base_layout.scale_problem(k)        # the result must be plausible, whatever the factor
        if why:
            return self.say(why)
        self._abort()
        self.base_layout = self.base_layout.scaled(k)
        self._rebuild(self.base_layout)
        self.say(line or f"Sizes changed by {k:.2f}. The table is now {self.base_layout.meta['scale']:.2f} times what the photo said.")

    ARM_SIZE_RANGE = (0.75, 4.0)    # tested: the built-in arm picks and places at every size in it (docs/ARMS.md)

    def set_arm(self, name: str | None = None, size: float | None = None):
        """Change the robot arm (a descriptor in echotwin/robot/arms or a path) and/or its size (1.0 = as described).
        The table stays as it is. Without a name the arm stays, without a size the size stays."""
        if self.world.name != "sim":
            return self.say("I can only change the arm of the simulation.")
        old = self.world.arm
        name, size = name or old.name, float(old.scale if size is None else size)
        lo, hi = self.ARM_SIZE_RANGE
        if not lo <= size <= hi:
            return self.say(f"I can be between {lo} and {hi} times my size.")
        try:
            self.world.arm = A.load(name).sized(size)
            self._abort()
            self._rebuild(self.base_layout)
        except (A.ArmError, ValueError) as e:
            self.world.arm = old
            self.log("system", f"Could not use that arm: {e}")
            return self.say("I couldn't use that arm. I keep the one I have.")
        a, ws = self.world.arm, self.world.workspace
        size_txt = "" if abs(size - 1) < 1e-6 else f" at {size:.2f} times its size"
        self.say(f"Now I have the {a.name} arm{size_txt}: my jaws open {a.max_opening * 100:.0f} cm "
                 f"and I reach {ws.r_min * 100:.0f} to {ws.r_max * 100:.0f} cm from my base.")

    def set_surface(self, kind: str, height: float | None = None):
        """What the things stand on (the guess was wrong): table, floor or other (height in m). A one-photo twin without the
        marker is resized too, because its sizes came from how high the phone was assumed to be above that surface."""
        from .scene import SURFACES
        if self.world.name != "sim":
            return self.say("I can only change that in the simulation.")
        if kind not in SURFACES:
            return self.say("The things are on a table, on the floor, or on something else.")
        h = 0.0 if kind == "floor" else float(0.75 if height is None and kind == "table" else (0.45 if height is None else height))
        if kind != "floor" and not 0.05 <= h <= 2.0:
            return self.say("A surface is between 5 cm and 2 m high.")
        new = {"kind": kind, "height": h}
        scan = self.scan or {}
        estimated = scan.get("mode") == "everyday" and (scan.get("calibration") or {}).get("source") != "marker"
        before = EV.phone_height(self.base_layout.surface)
        lay = EV.on_surface(self.base_layout, new) if estimated else self.base_layout.copy()
        lay.surface = dict(new)
        self._abort()
        self.base_layout = lay
        self._robot_for(lay)
        self._rebuild(lay)
        if scan:
            scan["surface"] = dict(new)
        where = {"table": "on a table", "floor": "on the floor"}.get(kind, f"on something {h * 100:.0f} cm high")
        k = EV.phone_height(new) / before if estimated else 1.0
        resized = "" if abs(k - 1) < 0.02 else (f" Seen from higher up, they are bigger than I thought: {k:.1f} times. I resized them."
                                                if k > 1 else f" Seen from closer, they are smaller than I thought: {k:.1f} times. I resized them.")
        self.say(f"Got it: the things are {where}.{resized}")

    def set_robot(self, name: str | None):
        """Use this robot (robots/<name>.json); None or "auto": the one that fits the surface. The arm and its size stay."""
        if self.world.name != "sim":
            return self.say("I can only change the robot in the simulation.")
        old, old_choice = self.world.robot, self.robot_choice
        try:
            self.robot_choice = None if name in (None, "auto") else name
            self.world.robot = RB.load(self.robot_choice or RB.for_surface(self.base_layout.surface))
            self._abort()
            self._rebuild(self.base_layout)
        except (RB.RobotError, A.ArmError, ValueError) as e:
            self.world.robot, self.robot_choice = old, old_choice
            self.log("system", f"Could not use that robot: {e}")
            return self.say("I couldn't use that robot. I keep the one I have.")
        r = self.world.robot
        what = "an arm on a base that drives up to things" if r.mobile else "an arm fixed at the edge of the table"
        self.say(f"Now I am {what}.")

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

    def set_drive(self, forward, sideways, turn):
        """The phone or the dashboard drives the mobile base by hand (each -1..1); stops by itself when they stop sending."""
        self.base_cmd = np.clip(np.array([forward, sideways, turn], dtype=float), -1, 1)
        self.base_t = time.time()

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
            action = self._practice.step()
        elif self.authority == "robot" and self.mode in ("replay", "move"):
            action = self._replay_step()
            recording = self.mode == "replay"
        elif self.authority == "human":
            if w.mobile and time.time() - self.base_t < HUMAN_TIMEOUT and np.any(self.base_cmd):
                if not w.drive_by(*self.base_cmd) and time.time() - self.base_blocked_t > 1.0:
                    self.base_blocked_t = time.time()       # something is in the way: tell the hand on the phone, once a second
                    self.haptic([60, 40, 60])
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
        for line in self.replay.pop("events", []):         # what the closed loop noticed: said as it happens
            self.say(line)
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
        w = self.world
        name = w.layout.props[plan["prop"]]["name"]
        if plan["goal"] is None:
            self.pending_move = plan
            return self.say(f"Where should I put the {name}? For example: next to another object, or to the left.")
        self.pending_move = None
        self.halted = False
        self._abort()
        task = PS.make_task(w, plan, heard)
        me = task["object"]
        goal, o = np.array(task["goal"]), w.obj_pos(me)
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
            return self._start_teach(task, reason="new_kind", why="no demos with something this size yet",
                                     line=f"I've never moved something this size: {M.describe(m)}. Can you show me? "
                                          "Drive me with the keyboard, upload a video, or say practice.")
        # 3. plan from what I learned, then imagine it on a copy of the world
        skill = self.skills.plan(m, float(np.linalg.norm(goal - o[:2])))
        wps = PS.waypoints(w, task, skill)
        im = PS.imagine(w, task, wps, skill["speed"])
        looped = PS.closed_loop_on(w) and self.policy is None      # a learned policy has no steps to check
        u = skill["uncertainty"] + (0 if im["ok"] else 0.8)
        why = f"{n} demo{'s' if n != 1 else ''} of a similar size · imagined: {im['text']}"
        if not im["ok"] or u >= PS.ASK:
            return self._start_teach(task, reason="unsure", why=why,
                                     line=f"I tried it in my head: {im['text']}. Can you show me how?")
        self.plan = {"uncertainty": u, "parts": skill["parts"], "why": why}
        self.ghost = im["path"]
        loop = None
        if looped:                                          # look first, check each step, retry: see prop_skills.loop_start
            wps, loop = PS.loop_start(w, task, skill)
        self.replay = {"wps": wps, "i": 0, "speed": skill["speed"], "task": task, "yaw": w.grasp_yaw(task["object"]), "loop": loop,
                       "before": {k: w.obj_pos(k)[:2].copy() for k in w.things()}}
        if self.policy is not None:
            self.replay["policy"] = POL.LearnedExecutor(self.policy, task, world=self.world)
        self.mode, self.authority = "move", "robot"
        self.emit({"t": "decision", "kind": "do", "task": task, "uncertainty": round(u, 2), "sure": T.percent_sure(u),
                   "parts": skill["parts"], "why": why})
        self.say(f"I'll move the {name} {where}. {T.confidence_words(u)}")

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
        self._practice = Practice(self, kinds, per_kind, prop)
        self.halted = False
        self.mode, self.authority = "practice", "robot"
        if not live:
            while self.mode == "practice":
                self.world.step(self._practice.step())

    def play_prop_video(self, plan: dict, heard: str):
        """Replay a move seen in a video of your hand, with the robot, in the twin; record it as a demo."""
        self.halted = False
        self._abort()
        name = self.world.layout.props[plan["prop"]]["name"]
        task = PS.make_task(self.world, plan, f"put the {name} {MT.describe_goal(self.world, plan)}")
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
            attempts = (r.get("loop") or {}).get("attempts")
            runlog.record(w.name, t, res, self.dataset.root.parent / "runs.jsonl", attempts=attempts)
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
    def _start_teach(self, task, reason: str, why: str, line: str):
        self.task = task
        self.mode, self.authority = "teach", "human"
        self.frames, self.ghost, self.plan = [], None, None
        self.settled = 0
        self.human_grip = self.world.hand.grip
        self.emit({"t": "decision", "kind": "teach", "task": task, "reason": reason, "why": why})
        self.say(line)
        self.haptic([80, 40, 80])

    # ---------------- utterances ----------------
    def handle_intent(self, it: Intent):
        k, n = it.kind, it.name
        if k == "safety":
            return self.halt("you said stop")
        if k == "control":
            return self._control(n)
        if k == "robot_q":
            return self.say(answers.about_robot(self, n))
        if k == "scene_q":
            return self.say(answers.about_table(self.world, n, {**it.data, "text": it.text}))
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

    # ---------------- output ----------------
    def _render(self):
        if self.halted:
            overlay = ("halted", self.halt_reason)
        elif self.mode == "practice" and self._practice:
            overlay = ("practice", *self._practice.overlay())
        elif self.mode == "teach" and self.frames:
            overlay = ("rec",)
        else:
            overlay = None
        jpeg = self._renderer.jpeg(self.world.view, self.ghost, self.plan["uncertainty"] if self.plan else 0.5, overlay)
        if jpeg:
            self.jpeg = jpeg
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
            "surface": dict(w.layout.surface),
            "robot": {"name": w.robot.name, "mobile": w.robot.mobile, "auto": self.robot_choice is None} if hasattr(w, "robot") else None,
            "arm": w.arm.name if hasattr(w, "arm") else None, "arm_size": float(w.arm.scale) if hasattr(w, "arm") else 1.0, "scale": float(self.base_layout.meta.get("scale", 1.0)),
            "backend": {"name": w.name, "note": self.backend_note, "ready": w.arm_ready(), "policy": self.policy is not None},
            "attempt": ((self.replay or {}).get("loop") or {}).get("attempts", 0) + 1 if self.mode == "move" and (self.replay or {}).get("loop") else None,
            "robot_runs": self.robot_runs, "scanned": bool(self.scan), "name": config.ROBOT_NAME,
        })
