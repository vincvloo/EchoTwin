"""Tilted grasps: an arm may lean its tool outward from straight down ("tilts_deg"), to reach further and higher."""
import json

import numpy as np
import pytest

from echotwin.robot import arm as A
from echotwin.robot.scene import Layout
from echotwin.robot.world import World


def _desc(name):
    return json.loads((A.ARMS_DIR / f"{name}.json").read_text(encoding="utf-8"))


def test_tilts_are_read_in_degrees_and_always_include_straight_down():
    assert A._tilts([40, 20]) == pytest.approx((0.0, np.radians(20), np.radians(40)))
    assert A._tilts([0]) == (0.0,)
    for bad in ([-10], [90], ["x"], None):
        with pytest.raises(A.ArmError):
            A._tilts(bad)
    assert A.load("builtin").tilts == (0.0,)                                    # the built-in arm does not tilt


def test_a_tilt_leans_the_tool_outward_and_keeps_the_jaws_square_to_it():
    w = World(Layout())
    ik = w.ik
    target = ik.base + np.array([0.0, 0.25, 0.05])
    point0, close0 = ik._aim(target, 0.3, 0.0)
    assert point0.tolist() == [0.0, 0.0, -1.0] and close0[2] == 0.0             # no tilt: as before
    point, close = ik._aim(target, 0.3, np.radians(30))
    assert np.linalg.norm(point) == pytest.approx(1.0) and point[2] == pytest.approx(-np.cos(np.radians(30)))
    assert point[1] > 0 and abs(point[0]) < 1e-9                                # leans away from the base (+y here)
    assert close @ point == pytest.approx(0.0, abs=1e-9) and np.linalg.norm(close) == pytest.approx(1.0)


def test_the_built_in_arm_never_tilts():
    w = World(Layout())
    for _ in range(10):
        w.step((0.25, 0.25, 0.25, 0))                                          # pushed towards the edge of its reach
    assert w.hand.tilt == 0.0 and w.workspace.tilts == (0.0,)


so100 = pytest.mark.skipif(A.load("so_arm100").missing_files() != [], reason="so_arm100 is not downloaded")


@so100
def test_tilting_lifts_the_so_arm100s_ceiling_far_from_its_base():
    straight = World(Layout(), A.parse({**_desc("so_arm100"), "tilts_deg": [0]}, A.ARMS_DIR / "so_arm100.json"))
    tilted = World(Layout(), A.load("so_arm100"))
    assert tilted.arm.tilts == pytest.approx(tuple(np.radians([0, 20, 40])))
    far = (0.0, 0.33)                                                            # 33 cm in front of the base
    assert not straight.workspace.reachable(far, 0.05) and tilted.workspace.reachable(far, 0.05)
    assert tilted.workspace.z_max(0.33) > straight.workspace.z_max(0.33)
    assert tilted.workspace.tilt_for((0.0, 0.2), 0.02) == 0.0                   # near: still straight down
    assert tilted.workspace.tilt_for(far, 0.05) > 0.0                           # far and up: leaning out
