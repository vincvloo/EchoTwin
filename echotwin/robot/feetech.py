"""Driver for the SO-ARM100 / SO-101: Feetech STS3215 servos on one serial bus.

NOT TESTED ON HARDWARE. It is written from the servo protocol and the arm's layout; the calibration maths and the command
flow are tested against a fake bus (`tests/robot/test_feetech.py`), nothing else. Expect to fix things on the first
connection, and read docs/REAL_ARM.md before you power the arm.

It needs the vendor SDK (`pip install feetech-servo-sdk`, module `scservo_sdk`) and a `real` block in the arm's descriptor:

    "real": {"baudrate": 1000000, "ids": [1, 2, 3, 4, 5], "gripper_id": 6,
             "zero_ticks": [2048, 2048, 2048, 2048, 2048],      # servo reading when each joint is at 0 rad in the sim
             "direction": [1, 1, 1, 1, 1],                      # +1 or -1: does a positive sim angle raise the reading?
             "ticks_per_rad": 651.9,                            # 4096 / 2 pi
             "gripper_ticks": {"open": 2048, "closed": 1500},
             "max_speed": 600, "acceleration": 50, "calibrated": false}

`"calibrated": false` makes the driver refuse to move anything. Set it to true only after the zeros and directions have
been checked with `python -m echotwin.robot.feetech --read` (arm held in its home pose, torque off).
"""
from __future__ import annotations

import sys
import time

import numpy as np

from .drivers import Driver, DriverError

TICKS = 4096


class FeetechDriver(Driver):
    def __init__(self, arm, port: str, bus=None):
        r = arm.real
        if not r:
            raise DriverError(f"the arm '{arm.name}' has no 'real' block (servo ids and calibration) in its descriptor")
        if not r.get("calibrated"):
            raise DriverError(f"the 'real' block of '{arm.name}' is not calibrated: fill in zero_ticks and direction, "
                              "then set \"calibrated\": true (docs/REAL_ARM.md)")
        self.arm, self.port, self.cfg = arm, port, r
        self.ids = list(r["ids"])
        self.gid = r["gripper_id"]
        self.zero = np.array(r["zero_ticks"], float)
        self.dir = np.array(r["direction"], float)
        self.tpr = float(r.get("ticks_per_rad", TICKS / (2 * np.pi)))
        self.g_open, self.g_closed = float(r["gripper_ticks"]["open"]), float(r["gripper_ticks"]["closed"])
        self.speed, self.acc = int(r.get("max_speed", 600)), int(r.get("acceleration", 50))
        lo, hi = arm_limits(arm)
        self.lo, self.hi = lo, hi
        self.bus = bus or open_bus(port, int(r.get("baudrate", 1_000_000)))

    # ---- calibration: simulator radians <-> servo ticks
    def to_ticks(self, q) -> np.ndarray:
        q = np.clip(np.asarray(q, float), self.lo, self.hi)             # never ask for more than the sim's joint limits
        return np.clip(np.round(self.zero + self.dir * q * self.tpr), 0, TICKS - 1).astype(int)

    def to_rad(self, ticks) -> np.ndarray:
        return (np.asarray(ticks, float) - self.zero) / (self.dir * self.tpr)

    def gripper_ticks(self, closed: bool) -> int:
        return int(round(self.g_closed if closed else self.g_open))

    def gripper_closure(self, ticks: float) -> float:
        return float(np.clip((ticks - self.g_open) / (self.g_closed - self.g_open), 0.0, 1.0))

    # ---- Driver
    def connect(self, q0) -> None:
        for i in (*self.ids, self.gid):
            self.bus.torque(i, True)

    def command(self, q, closed) -> None:
        for i, t in zip(self.ids, self.to_ticks(q)):
            self.bus.write_pos(i, int(t), self.speed, self.acc)
        self.bus.write_pos(self.gid, self.gripper_ticks(bool(closed)), self.speed, self.acc)

    def advance(self, dt: float) -> None:
        time.sleep(dt)

    def read(self):
        q = self.to_rad([self.bus.read_pos(i) for i in self.ids])
        return q, self.gripper_closure(self.bus.read_pos(self.gid))

    def stop(self) -> None:
        for i in (*self.ids, self.gid):
            self.bus.torque(i, False)

    def start(self) -> None:
        for i in (*self.ids, self.gid):
            self.bus.torque(i, True)

    def close(self) -> None:
        self.bus.close()


def arm_limits(arm):
    """Joint limits of the arm's MJCF, the same ones the simulator's IK respects."""
    from .scene import Layout, build_xml
    import mujoco
    from . import arm as A
    spec = mujoco.MjSpec.from_string(build_xml(Layout()))
    A.compose(spec, arm, (0.0, 0.0))
    m = spec.compile()
    ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, arm.prefix + n) for n in arm.joints]
    return m.jnt_range[ids, 0].copy(), m.jnt_range[ids, 1].copy()


class SdkBus:
    """The vendor SDK behind four calls, so the driver can be tested with a fake."""

    def __init__(self, port: str, baudrate: int):
        try:
            import scservo_sdk as sdk
        except ImportError as e:
            raise DriverError("the servo SDK is not installed (pip install feetech-servo-sdk)") from e
        self.sdk = sdk
        self.port = sdk.PortHandler(port)
        if not self.port.openPort():
            raise DriverError(f"cannot open {port}")
        self.port.setBaudRate(baudrate)
        self.h = sdk.sms_sts(self.port)

    def write_pos(self, i: int, ticks: int, speed: int, acc: int) -> None:
        self.h.WritePosEx(i, ticks, speed, acc)

    def read_pos(self, i: int) -> int:
        pos, result, error = self.h.ReadPos(i)
        if result != self.sdk.COMM_SUCCESS:
            raise DriverError(f"servo {i} did not answer")
        return int(pos)

    def torque(self, i: int, on: bool) -> None:
        self.h.write1ByteTxRx(i, 40, 1 if on else 0)        # STS3215 torque-enable register

    def close(self) -> None:
        self.port.closePort()


def open_bus(port: str, baudrate: int):
    return SdkBus(port, baudrate)


def main(argv=None) -> int:
    """`--read`: print what every servo reads now (torque off, arm held in its home pose), to fill in zero_ticks."""
    import argparse
    from . import arm as A
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--arm", default="so_arm100")
    ap.add_argument("--port", required=True)
    ap.add_argument("--read", action="store_true")
    a = ap.parse_args(argv)
    arm = A.load(a.arm)
    if not arm.real:
        print(f"{arm.name} has no 'real' block")
        return 1
    bus = open_bus(a.port, int(arm.real.get("baudrate", 1_000_000)))
    for name, i in zip((*arm.joints, "gripper"), (*arm.real["ids"], arm.real["gripper_id"])):
        bus.torque(i, False)
        print(f"  {name:12s} id {i}  ticks {bus.read_pos(i)}")
    bus.close()
    print("Put these in the 'real' block: zero_ticks for the joints at their home pose; check each direction by moving a joint.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
