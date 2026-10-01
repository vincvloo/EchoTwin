"""Run the robot in the true world while the particle filter localises it in the scanned map."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..perception.gridmap import GridMap
from .mcl import MCL
from .robot import Robot, Wander, wrap
from .sonar import SonarRig


def add_clutter(world: GridMap, n: int, rng: np.random.Generator, size=(0.25, 0.5), clearance=0.5) -> GridMap:
    """World with `n` box obstacles the scan does not contain (bags, chairs moved after the scan).

    Boxes go in free space at least `clearance` m from mapped obstacles so doorways stay passable.
    Only for the simulator's world: the filter must keep using the scanned map.
    """
    fx, fy = world.free_cells(clearance)
    extra = np.zeros_like(world.occ)
    for i in rng.choice(len(fx), min(n, len(fx)), replace=False):
        w, h = rng.uniform(*size, 2)
        r0, c0 = world.to_cell(np.array([fx[i] - w / 2]), np.array([fy[i] - h / 2]))
        r1, c1 = world.to_cell(np.array([fx[i] + w / 2]), np.array([fy[i] + h / 2]))
        extra[max(r0[0], 0):r1[0] + 1, max(c0[0], 0):c1[0] + 1] = True
    return world.with_extra(extra & world.known)


@dataclass
class Episode:
    t: list = field(default_factory=list)
    true: list = field(default_factory=list)
    est: list = field(default_factory=list)
    z: list = field(default_factory=list)
    spread: list = field(default_factory=list)
    n_particles: list = field(default_factory=list)
    snapshots: list = field(default_factory=list)   # (step, particles) for animation
    dist: float = 0.0

    def errors(self):
        tr, es = np.array(self.true), np.array(self.est)
        pos = np.hypot(tr[:, 0] - es[:, 0], tr[:, 1] - es[:, 1])
        ang = np.abs(wrap(tr[:, 2] - es[:, 2]))
        return pos, ang

    def convergence(self, pos_tol=0.3, ang_tol=np.deg2rad(20), hold=20):
        """First step after which the estimate stays within tolerance for `hold` steps."""
        pos, ang = self.errors()
        ok = (pos < pos_tol) & (ang < ang_tol)
        for i in range(len(ok) - hold):
            if ok[i:i + hold].all():
                return i
        return None


def run_episode(gmap: GridMap, world: GridMap, rig: SonarRig, seed=0, steps=400, dt=0.25,
                start=None, kidnap_at=None, snapshot_every=0, n_max=3000, drive: GridMap | None = None):
    """`drive` (optional) limits where the robot can go without being visible to the sonars,
    e.g. the edge of a raised platform (a cliff sensor stops the robot, sonar sees nothing).
    Defaults to `world`."""
    rng = np.random.default_rng(seed)
    drive = world if drive is None else drive
    if start is None:
        fx, fy = drive.free_cells(0.35)
        i = rng.integers(len(fx))
        start = np.array([fx[i], fy[i], rng.uniform(-np.pi, np.pi)])
    robot = Robot(pose=np.array(start, float))
    ctrl = Wander(rng)
    pf = MCL(gmap, rig, rng, n_max=n_max)
    ep = Episode()
    z = rig.measure(world, robot.pose, rng)
    turn = 0                                                  # bumper / cliff reflex: steps left to turn in place
    for k in range(steps):
        if kidnap_at is not None and k == kidnap_at:          # pick the robot up and move it
            fx, fy = drive.free_cells(0.35)
            i = rng.integers(len(fx))
            robot.pose = np.array([fx[i], fy[i], rng.uniform(-np.pi, np.pi)])
        v, w = ctrl(z)
        if turn:
            v, w, turn = 0.0, turn_w, turn - 1
        before = robot.pose.copy()
        odo = robot.step(drive, v, w, dt, rng)
        if v > 0 and np.allclose(robot.pose[:2], before[:2]):   # blocked: turn away for ~1-2 s
            turn, turn_w = int(rng.integers(4, 9)), float(rng.choice([-1.2, 1.2]))
        ep.dist += np.hypot(*(robot.pose[:2] - before[:2]))
        z = rig.measure(world, robot.pose, rng)
        pf.predict(odo)
        pf.update(z)
        est, _ = pf.estimate()
        pf.resample()
        ep.t.append(k * dt); ep.true.append(robot.pose.copy()); ep.est.append(est)
        ep.z.append(z.copy()); ep.spread.append(pf.spread()); ep.n_particles.append(len(pf.p))
        if snapshot_every and k % snapshot_every == 0:
            ep.snapshots.append((k, pf.p.copy()))
    return ep
