"""A mobile base stands where the grip and the carry work, not only where the object is in reach."""
import numpy as np
import pytest

from echotwin.robot import arm as A
from echotwin.robot.features import prop_skills as PS
from echotwin.robot.scene import Layout
from echotwin.robot.world import World

pytestmark = pytest.mark.skipif(A.load("so_arm100").missing_files() != [], reason="so_arm100 is not downloaded")


def _floor(*props):
    lay = Layout(table_half=(1.0, 0.8), surface={"kind": "floor", "height": 0.0})
    lay.props = [dict(p) for p in props]
    return lay


def _thing(name, x, y, half, shape="box"):
    return {"name": name, "shape": shape, "pos": (x, y), "yaw": 0.0, "size": half, "rgb": (0.8, 0.3, 0.3)}


def test_it_picks_from_a_spot_where_its_single_jaw_gets_around_the_thing():
    w = World(_floor(_thing("bar", 0.5, 0.4, (0.05, 0.02, 0.02))), A.load("so_arm100"), "mobile_arm")
    pose = w.pick_pose("prop_0")
    assert w.grips_from("prop_0", pose) and w.refusal("prop_0") == ""
    assert not np.allclose(pose, w.hand.base)                 # it is far away: it drives there first


def test_it_sets_a_thing_on_another_from_where_it_can_lift_it_high_enough():
    w = World(_floor(_thing("ball", 0.0, 0.1, (0.025, 0.025, 0.025), "round"), _thing("box", 0.5, 0.5, (0.025, 0.025, 0.025))),
              A.load("so_arm100"), "mobile_arm")
    task = PS.make_task(w, {"prop": 0, "goal": ("near", 1, "on top of")}, "put the ball on the box")
    pick = w.obj_pos("prop_0")[:2] + w.grasp_offset("prop_0")[:2]
    place = np.asarray(task["goal"]) + w.grasp_offset("prop_0")[:2]
    _, (pose, at_place) = w.stands(pick, place, "prop_0", "prop_1")
    top = w.obj_pos("prop_1")[2] + w.half("prop_1")
    assert at_place is not None
    assert w.ceiling(place, pose) - w.stack_hang("prop_0") - top >= w.stack_clear("prop_0")
    assert w.refusal("prop_0", task["goal"], on="prop_1") == ""
