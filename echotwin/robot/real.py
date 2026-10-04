"""The real-arm back-end: servos move the arm, the simulation runs beside it as its twin (see backend.py)."""
from __future__ import annotations

import numpy as np

from .drivers import Driver, closure
from .world import CTRL_DT, World

REAL_SPEED = 0.5            # a real arm moves at half the simulation's tool speed
MAX_LEAD = 0.02             # the tool target is never more than this far ahead of the measured tool (m)
SETTLED = 0.02              # the gripper closure changes less than this per tick: it has stopped
CLOSED_ALL_THE_WAY = 0.95   # a closure above this means the jaws met nothing (or the object slipped out)
DISAGREE = 0.25             # the real jaws and the twin's jaws are this far apart (fraction of the travel): they do not hold the same thing


class TruthObserver:
    """Reads object poses from a world that is the truth (the mock arm's plant): a perfect camera. For tests and for the
    sim-vs-mock comparison. A real camera loop would replace it (PR12)."""

    def __init__(self, plant: World):
        self.plant = plant

    def obj_pos(self, name):
        return self.plant.obj_pos(name)

    def tilt(self, name):
        return self.plant.tilt(name)


def mock_camera(driver) -> "CameraObserver":
    """A top-down camera over the mock arm's own world."""
    from .observe import CameraObserver, RenderSource
    src = RenderSource(lambda: driver.plant)
    return CameraObserver(src, lambda: src.plane(driver.plant))


class RealBackend:
    """An arm moved by servos, with the simulation running beside it as its twin.

    Each tick: the twin's controller turns the wanted tool motion into joint setpoints, the driver sends them and lets
    time pass, the measured joints are read back and fed to the twin's physics. The tool position comes from the measured
    joints. Objects are where the twin believes they are (it reproduces contacts, so a held object moves with the tool)
    unless an `observer` says otherwise.
    """
    name = "real"

    def __init__(self, twin: World, driver: Driver, observer=None, speed: float = REAL_SPEED, armed: bool = True,
                 camera=None):
        """`observer`: a source of object truth (the mock's own world: for judging in tests). `camera`: a CameraObserver the
        robot looks through (`observe`); without one it can only believe the twin."""
        self.twin, self.driver, self.observer, self.camera, self.speed = twin, driver, observer, camera, speed
        self.stopped, self.armed = False, armed
        self.q_meas, self.closure, self._closure_prev = np.zeros(5), 0.0, 0.0
        driver.connect(twin.hand.q)
        self._read()
        twin.hand.q = self.q_meas.copy()

    # everything that is not about moving or observing the arm is the twin's
    def __getattr__(self, name):
        if name in ("twin", "driver"):
            raise AttributeError(name)
        return getattr(self.twin, name)

    # ---------- the arm
    def _read(self):
        self._closure_prev = self.closure
        self.q_meas, self.closure = self.driver.read()

    def arm_ready(self) -> bool:
        return self.armed

    def enable(self):
        self.armed = True

    def step(self, action, d=None, hs=None):
        if d is not None or hs is not None:                    # a copy made by clone(): imagination, on the twin
            return self.twin.step(action, d, hs)
        if self.stopped or not self.armed:
            return
        a = np.array(action, dtype=float)
        a[:3] *= self.speed
        tw = self.twin
        q = tw.command(a, tw.hand)
        self.driver.command(q, tw.hand.grip)
        self.driver.advance(CTRL_DT)
        self._read()
        tw.hand.q = self.q_meas.copy()                          # the next setpoint starts from where the arm is
        err = tw.hand.target - self.hand_pos()                  # the arm lags behind its target: do not let the target run away
        n = float(np.linalg.norm(err))                          # from it (it would keep sinking after the move is "done")
        if n > MAX_LEAD:
            tw.hand.target = self.hand_pos() + err * (MAX_LEAD / n)
        tw.advance(tw.data, tw.hand, q=self.q_meas)
        if tw.hand.held and (self.closure > CLOSED_ALL_THE_WAY or abs(self.closure - closure(tw, tw.data)) > DISAGREE):
            tw.hand.held = None          # the twin's jaws stopped on an object; the real jaws did not (they went further): nothing is held

    def hand_pos(self, d=None):
        if d is not None:
            return self.twin.hand_pos(d)
        return self.twin.ik.pose(self.q_meas)[0]

    def grip_settled(self, d=None) -> bool:
        if d is not None:
            return self.twin.grip_settled(d)
        return abs(self.closure - self._closure_prev) < SETTLED

    def go_rest(self, d=None, hs=None, teleport: bool = False):
        if d is not None or hs is not None or teleport:
            return self.twin.go_rest(d, hs, teleport)
        self.twin.go_rest()                                     # a new target: the next steps move the arm there

    def settle(self, ticks: int = 20):
        for _ in range(ticks):
            self.step((0, 0, 0, 1 if self.twin.hand.grip else 0))

    def build(self, layout):
        self.twin.build(layout)
        self.driver.on_layout(self.twin.layout)
        self.driver.connect(self.twin.hand.q)
        self._read()
        self.twin.hand.q = self.q_meas.copy()

    def reset(self, layout=None):
        self.build(layout or self.twin.layout)

    # ---------- the scene
    def observe(self, name):
        """Look at an object and move the twin's belief to where it is seen. The position, or None if it is not seen."""
        tw = self.twin
        if self.camera is None:
            return self.obj_pos(name)[:2].copy()
        others = [(*tw.obj_pos(n)[:2], tw.half(n)) for n in tw.things() if n != name]
        hand = self.hand_pos()
        others += [(tw.base[0], tw.base[1], 0.04), (hand[0], hand[1], hand[2])]      # the arm's base and tool are not objects
        est = self.camera.observe(name, tw.obj_pos(name)[:2], 2 * tw.radius(name), tw.half(name), others)
        if est is not None and not tw.hand.held:
            tw.set_obj_pose(name, est)
        return est

    def obj_pos(self, name, d=None):
        if d is None and self.observer is not None:
            return self.observer.obj_pos(name)
        return self.twin.obj_pos(name, d)

    def tilt(self, name, d=None):
        if d is None and self.observer is not None:
            return self.observer.tilt(name)
        return self.twin.tilt(name, d)

    @property
    def view(self) -> World:
        """Draw the mock arm's own world when there is one (it is the 'real' one), else the twin."""
        return self.driver.plant or self.twin

    # ---------- safety
    def stop(self):
        self.stopped = True
        self.driver.stop()

    def resume(self):
        self.stopped = False
        self.driver.start()

    def close(self):
        self.driver.stop()
        self.driver.close()
