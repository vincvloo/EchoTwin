"""The reach ceiling: pointing straight down it is found between the grid's rows; leaning out it stays on them."""
import numpy as np
import pytest

from echotwin.robot import arm as A
from echotwin.robot.scene import Layout
from echotwin.robot.world import World


def _ws(arm=None):
    return World(Layout(), arm).workspace


def test_the_straight_ceiling_lies_between_the_row_reached_and_the_next():
    ws = _ws()
    for i in range(len(ws.RADII)):
        rows = np.nonzero(ws.oks[0, i])[0]
        if not len(rows):
            assert ws.tops[0, i] == 0.0
            continue
        assert ws.HEIGHTS[rows.max()] <= ws.tops[0, i] + 1e-9
        above = ws.HEIGHTS[ws.HEIGHTS > ws.HEIGHTS[rows.max()]]
        if len(above):
            assert ws.tops[0, i] < above[0]


def test_the_arm_really_reaches_its_straight_ceiling():
    w = World(Layout())
    ws, ik = w.workspace, w.ik
    for r in (0.14, 0.22, 0.30):
        i = int(np.argmin(abs(ws.RADII - r)))
        _, ep, _ = ik.solve(ik.base + np.array([0.0, ws.RADII[i], ws.tops[0, i]]), 0.0, ik.q_down, iters=200)
        assert ep < 2e-3


so100 = pytest.mark.skipif(A.load("so_arm100").missing_files() != [], reason="so_arm100 is not downloaded")


@so100
def test_leaning_ceilings_stay_on_the_rows_and_straight_down_needs_no_lean():
    ws = _ws(A.load("so_arm100"))
    for k in range(1, len(ws.tilts)):
        for i in range(len(ws.RADII)):
            assert ws.tops[k, i] == 0.0 or ws.tops[k, i] in ws.HEIGHTS          # not refined
    i = int(np.argmin(abs(ws.RADII - 0.26)))
    assert ws.tops[0, i] > 0.08                                                  # the grid said 5 cm: its 9 cm row was missed
    assert ws.tilt_for((0.0, ws.RADII[i]), ws.tops[0, i] - 0.002) == 0.0       # up to there, straight down


@so100
def test_one_lean_per_distance_from_the_table_to_the_ceiling():
    """Going down to a thing and up from it the lean must not change: the open jaw swung over a flat thing and flipped it."""
    ws = _ws(A.load("so_arm100"))
    for r in (0.20, 0.26, 0.34):
        i = int(np.argmin(abs(ws.RADII - r)))
        leans = {ws.tilt_for((0.0, ws.RADII[i]), z) for z in np.arange(0.006, ws.z_max(ws.RADII[i]), 0.005)}
        assert len(leans) == 1
    assert ws.tilt_for((0.0, 0.20), 0.05) == 0.0 and ws.tilt_for((0.0, 0.34), 0.01) > 0.0   # far out it leans all the way
