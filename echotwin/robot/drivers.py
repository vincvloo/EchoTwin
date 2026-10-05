"""Drivers: how joint targets reach an arm, and how its joints are read back.

A driver moves five joints and a gripper and says where they are. `RealBackend` does everything else (kinematics,
safety, the twin). Joint values are the simulator's: radians, in the order of the arm descriptor; the gripper is a
closure from 0 (open) to 1 (closed all the way).

- `MockDriver`: an arm that is not the simulation. It moves a second, perturbed MuJoCo world (heavier objects, other
  friction, servo lag, encoder offsets and noise) and reports its joints like encoders would. It is the stand-in for
  hardware in tests, and a stress test for skills: what they lose when reality is not the simulation.
- `FeetechDriver` (`feetech.py`): the SO-ARM100 / SO-101 servo bus. Not tested on hardware.
"""
from __future__ import annotations

import numpy as np

from .world import CTRL_DT, World


class DriverError(RuntimeError):
    pass


class Driver:
    """The interface `RealBackend` needs. Subclasses override all of it."""
    plant = None                    # a simulated truth world, only for the mock

    def connect(self, q0: np.ndarray) -> None:
        """Open the connection. `q0`: the joint values the twin starts from (a driver may ignore it)."""

    def command(self, q: np.ndarray, closed: bool) -> None:
        raise NotImplementedError

    def advance(self, dt: float) -> None:
        """Let `dt` seconds pass (a real arm: sleep; the mock: step its world)."""
        raise NotImplementedError

    def read(self) -> tuple[np.ndarray, float]:
        """(five joint values in radians, gripper closure 0..1)."""
        raise NotImplementedError

    def stop(self) -> None:
        """Hold still / torque off."""

    def start(self) -> None:
        """Undo `stop`."""

    def close(self) -> None:
        pass

    def on_layout(self, layout) -> None:
        """The table was rebuilt (a new scan, or a reset): the mock rebuilds its world too."""


def closure(world: World, d) -> float:
    """How far the gripper of `world` is closed, 0 (open) to 1 (the commanded closed value)."""
    m = world.model
    vals = []
    for a, lo, hi in zip(world.gact, world.arm.gripper["open"], world.arm.gripper["closed"]):
        q = float(d.qpos[m.jnt_qposadr[m.actuator_trnid[a][0]]])
        vals.append(np.clip((q - lo) / (hi - lo + 1e-9), 0.0, 1.0))
    return float(np.mean(vals))


class MockDriver(Driver):
    """A second world as the 'real' arm: joints lag behind their targets, encoders are a little off and noisy, objects are
    heavier and the pads are less grippy than in the twin."""

    def __init__(self, layout, arm=None, lag=0.45, mass_scale=1.4, friction_scale=0.8, encoder_offset=0.012,
                 encoder_noise=0.002, seed=0):
        self.arm, self.layout = arm, layout
        self.lag, self.mass_scale, self.friction_scale = lag, mass_scale, friction_scale
        rng = np.random.default_rng(seed)
        self.offset = rng.normal(0.0, encoder_offset, 5)          # what each encoder reads over the true angle
        self.noise, self.rng = encoder_noise, rng
        self.stopped = False
        self.plant: World | None = None
        self._build()

    # what a randomised arm may be like: (low, high) of each parameter
    RANGES = {"lag": (0.25, 0.65), "mass_scale": (1.0, 1.8), "friction_scale": (0.6, 1.0), "encoder_offset": (0.0, 0.03),
              "encoder_noise": (0.0005, 0.003)}

    @classmethod
    def randomized(cls, layout, arm=None, rng=None, scale: float = 1.0) -> "MockDriver":
        """A mock arm with parameters drawn at random: the training data for a policy that must work on an arm that is not the simulation
        (domain randomisation). `scale` widens (>1) or narrows (<1) every range around its middle."""
        rng = rng or np.random.default_rng()
        p = {}
        for k, (lo, hi) in cls.RANGES.items():
            mid, half = (lo + hi) / 2, (hi - lo) / 2 * scale
            p[k] = float(max(1e-4, rng.uniform(mid - half, mid + half)))
        return cls(layout, arm, seed=int(rng.integers(1 << 31)), **p)

    def _build(self):
        self.plant = World(self.layout, self.arm)
        m = self.plant.model
        m.geom_friction[:, 0] *= self.friction_scale
        for b in self.plant.obj_body.values():
            m.body_mass[b] *= self.mass_scale
            m.body_inertia[b] *= self.mass_scale
        self.q_cmd = self._true_q().copy()
        self.q_filt = self.q_cmd.copy()
        self.closed = False

    def _true_q(self) -> np.ndarray:
        return np.array([self.plant.data.qpos[a] for a in self.plant.ik.qadr])

    def on_layout(self, layout) -> None:
        self.layout = layout
        self._build()

    def connect(self, q0) -> None:
        """Start the mock arm where the twin is."""
        p = self.plant
        for adr, val in zip(p.ik.qadr, np.asarray(q0, float) - self.offset):      # so that it reads q0
            p.data.qpos[adr] = val
        for act, val in zip(p.act, np.asarray(q0, float) - self.offset):
            p.data.ctrl[act] = val
        import mujoco
        mujoco.mj_forward(p.model, p.data)
        self.q_cmd = self._true_q().copy()
        self.q_filt = self.q_cmd.copy()

    def command(self, q, closed) -> None:
        if self.stopped:
            return
        self.q_cmd = np.asarray(q, float) - self.offset          # the controller aims at the reading it wants
        self.closed = bool(closed)

    def advance(self, dt: float) -> None:
        p = self.plant
        if not self.stopped:
            self.q_filt = self.q_filt + self.lag * (self.q_cmd - self.q_filt)
        p.advance(p.data, p.hand, q=self.q_filt, grip=self.closed)

    def read(self):
        q = self._true_q() + self.offset + self.rng.normal(0.0, self.noise, 5)
        return q, closure(self.plant, self.plant.data)

    def stop(self) -> None:
        self.stopped = True
        self.q_filt = self._true_q().copy()                       # holds where it is
        self.q_cmd = self.q_filt.copy()

    def start(self) -> None:
        self.stopped = False
