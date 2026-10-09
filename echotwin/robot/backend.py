"""The robot contract: what a skill needs from whatever executes it.

Skills, practice, replay and the dashboard talk to a *back-end*. Two exist:

- `sim` (`World`): MuJoCo is the robot. It knows everything (object ground truth, contacts).
- `real` (`RealBackend`, below): servos move a real arm. The same MuJoCo model runs next to it as a *twin*: it
  gives geometry and inverse kinematics, imagines moves before they are made, and shadows the real arm.

What a skill may rely on, and what a real arm must therefore provide:

  Command    step(action)           one control tick (20 Hz): action = (vx, vy, vz, grip, yaw), tool velocity in m/s in
                                    table coordinates, grip 0 or 1, yaw of the jaws in radians (optional)
  Robot      hand_pos()             where the tool is (m)         real: forward kinematics of the measured joints
             hand.grip, hand.held   commanded grip; what the pads hold   real: the twin's contacts, checked against the gripper
             grip_settled()         has the gripper stopped moving      real: from successive gripper readings
             go_rest()              park the arm
             stop() / resume()      emergency stop (torque off on a real arm)
  Scene      obj_pos(name), tilt(name)   where an object is, how far it leans   real: the twin's belief; a camera later (PR12)
             things(), half(), radius(), tallest(), layout        sizes and places from the scan (the twin)
  Geometry   reachable(xy), can_grasp(name), refusal(name, goal, on), grasp_offset(name), grasp_yaw(name),
             carry_height(), path_ceiling(a, b), stack_hang(name),
             workspace_sample(rng)                                from the arm descriptor and the twin's IK
  Planning   twin                   the World used to imagine (for the sim back-end: itself); clone() copies its state

`d` and `hs` arguments (MjData and HandState) select a copy made by `clone()`: they always refer to the twin.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

CONTRACT = ("step", "hand_pos", "grip_settled", "go_rest", "stop", "resume", "close", "obj_pos", "tilt", "things", "half",
            "radius", "tallest", "reachable", "can_grasp", "refusal", "grasp_offset", "grasp_yaw", "carry_height",
            "path_ceiling", "stack_hang", "workspace_sample", "clone", "settle", "arm_ready", "observe", "observe_pose",
            "set_obj_pose", "see_tool")


@runtime_checkable
class Backend(Protocol):
    name: str
    hand: object          # .grip (commanded), .held (name of the object the pads hold, or None), .target, .yaw
    layout: object
    twin: object

    def step(self, action, d=None, hs=None) -> None: ...
    def hand_pos(self, d=None) -> np.ndarray: ...
    def grip_settled(self, d=None) -> bool: ...
    def go_rest(self, d=None, hs=None, teleport: bool = False) -> None: ...
    def stop(self) -> None: ...
    def resume(self) -> None: ...
    def close(self) -> None: ...
    def arm_ready(self) -> bool: ...
    def obj_pos(self, name: str, d=None) -> np.ndarray: ...
    def observe(self, name: str): ...
    def observe_pose(self) -> np.ndarray: ...
    def see_tool(self): ...
    def set_obj_pose(self, name: str, xy, d=None) -> None: ...
    def tilt(self, name: str, d=None) -> float: ...
    def things(self) -> list: ...
    def half(self, name: str) -> float: ...
    def radius(self, name: str) -> float: ...
    def tallest(self) -> float: ...
    def reachable(self, xy, z: float = 0.02) -> bool: ...
    def can_grasp(self, name: str) -> tuple: ...
    def refusal(self, name: str, goal_xy=None, on: str | None = None) -> str: ...
    def grasp_offset(self, name: str) -> np.ndarray: ...
    def grasp_yaw(self, name: str) -> float: ...
    def carry_height(self) -> float: ...
    def workspace_sample(self, rng) -> tuple: ...
    def clone(self) -> tuple: ...
    def settle(self, ticks: int = 20) -> None: ...


def missing(obj) -> list[str]:
    """The names of the contract that `obj` does not provide (empty when it is a complete back-end)."""
    return [n for n in CONTRACT if not callable(getattr(obj, n, None))]


# ---------------------------------------------------------------- choosing one
def make(layout=None, env=None) -> tuple["Backend", str]:
    """The back-end named by BACKEND in the environment, and a sentence that says what was chosen.

    BACKEND=sim (default): the simulation is the robot.
    BACKEND=real: servos move the arm on REAL_PORT. Without a port the mock arm is used (an arm that is not the
    simulation); a port that cannot be opened falls back to the simulation. The real arm waits for "arm the robot"
    before its first move unless REAL_REQUIRE_GO=0."""
    import os
    from .arm import ArmError
    from .drivers import DriverError, MockDriver
    from .real import RealBackend, mock_camera
    from .world import World
    env = os.environ if env is None else env
    kind = (env.get("BACKEND") or "sim").strip().lower()
    if kind not in ("sim", "real"):
        return World(layout), f"BACKEND={kind!r} is not sim or real: using the simulation."
    if kind == "sim":
        return World(layout), "Simulation."
    twin = World(layout)
    port = (env.get("REAL_PORT") or "").strip()
    go = (env.get("REAL_REQUIRE_GO") or "1").strip() != "0"
    if not port:
        drv = MockDriver(twin.layout, twin.arm)
        return (RealBackend(twin, drv, camera=mock_camera(drv), armed=True),
                "No REAL_PORT: running the mock arm (a simulated arm that is not the simulation: lag, encoder offsets, "
                "heavier objects).")
    try:
        from .feetech import FeetechDriver
        drv = FeetechDriver(twin.arm, port)
        return RealBackend(twin, drv, armed=not go), f"Real arm on {port}."
    except (DriverError, ArmError, ImportError, OSError) as e:
        return twin, f"Could not use the real arm on {port} ({e}). Using the simulation."
