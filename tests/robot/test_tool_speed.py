"""An arm's tool_speed caps how fast it moves its tool (a single jaw carries slower)."""
import numpy as np
import pytest

from echotwin.robot import arm as A
from echotwin.robot.scene import Layout
from echotwin.robot.world import CTRL_DT, World


def _speed(w):
    w.command((0.0, 0.0, 0.0, 0))                          # the target settles where the arm can reach first
    before = w.hand.target.copy()
    w.command((0.0, 0.25, 0.0, 0))
    return float(np.linalg.norm(w.hand.target - before)) / CTRL_DT


def test_an_arm_without_tool_speed_moves_as_before():
    w = World(Layout())
    assert w.arm.tool_speed is None and _speed(w) == pytest.approx(0.25)


@pytest.mark.skipif(A.load("so_arm100").missing_files() != [], reason="so_arm100 is not downloaded")
def test_the_so_arm100_carries_at_its_own_speed_and_it_grows_with_the_arm():
    so = A.load("so_arm100")
    assert so.tool_speed == pytest.approx(0.1)
    assert _speed(World(Layout(), so)) == pytest.approx(0.1)
    assert so.sized(so.scale * 2).tool_speed == pytest.approx(0.2)
