"""Skills keyed by measurements: width, height, length, mass; size classes; old demos."""
import pytest

from echotwin.robot.features import measure as M
from echotwin.robot.scene import Layout
from echotwin.robot.world import World


def _world(*props):
    lay = Layout()
    lay.props = [{"name": n, "shape": s, "pos": (0.0, -0.05 + 0.1 * i), "yaw": 0.0, "size": z, "rgb": (0.8, 0.3, 0.3)}
                 for i, (n, s, z) in enumerate(props)]
    return World(lay)


def test_measure_uses_the_narrow_side_height_and_length():
    w = _world(("bar", "flat", (0.05, 0.03, 0.01)), ("glass", "cylinder", (0.035, 0.035, 0.05)))
    m = M.measure(w, "prop_0")
    assert m["width"] == pytest.approx(0.06) and m["height"] == pytest.approx(0.02) and m["length"] == pytest.approx(0.10)
    assert m["mass"] > 0
    g = M.measure(w, "prop_1")
    assert g["width"] == pytest.approx(0.07) and g["height"] == pytest.approx(0.10)


def test_distance_and_size_classes():
    a = {"width": 0.05, "height": 0.04, "length": 0.05}
    b = {"width": 0.06, "height": 0.045, "length": 0.06}                 # 1 cm wider, 5 mm taller
    c = {"width": 0.06, "height": 0.02, "length": 0.10}                  # a flat bar
    assert M.distance(a, a) == 0 and M.distance(a, b) < 0.7 < 1.0 < M.distance(a, c)
    assert [M.size_class(x) for x in (a, c, {"width": 0.03, "height": 0.04, "length": 0.03},
                                      {"width": 0.07, "height": 0.10, "length": 0.07})] == ["medium", "flat", "small", "tall"]
    assert "6 cm wide and 10 cm tall" in M.describe({"width": 0.06, "height": 0.10, "length": 0.06})


def test_old_task_without_measurement_gets_a_typical_one():
    m = M.from_task({"shape": "cylinder", "h": 0.05})
    assert m["legacy"] and m["height"] == pytest.approx(0.10) and m["width"] == pytest.approx(0.065)
    assert M.from_task({"m": {"width": 1, "height": 2, "length": 3, "mass": 4}})["width"] == 1


# ---------------- skills on measurements ----------------
def _demo(m, drop=0.004, dist=0.2):
    return {"grip": 0.5, "lift": 0.05, "drop": drop, "speed": 0.2, "dist": dist, "m": m}


BOX5 = {"width": 0.05, "height": 0.04, "length": 0.05, "mass": 0.1}


def test_a_demo_counts_for_similar_sizes_only():
    from echotwin.robot.features.prop_skills import PropSkills
    s = PropSkills()
    s.demos = [_demo(BOX5)]
    box6 = {"width": 0.06, "height": 0.045, "length": 0.06, "mass": 0.15}
    bar = {"width": 0.06, "height": 0.02, "length": 0.10, "mass": 0.04}
    assert s.known(box6) and not s.known(bar)
    assert s.counts() == {"flat": 0, "small": 0, "medium": 1, "tall": 0}


def test_uncertainty_grows_with_distance_in_size():
    from echotwin.robot.features.prop_skills import PropSkills
    s = PropSkills()
    s.demos = [_demo(BOX5), _demo(BOX5, 0.006)]
    near = s.plan(BOX5, 0.2)["uncertainty"]
    far = s.plan({"width": 0.06, "height": 0.07, "length": 0.06, "mass": 0.1}, 0.2)["uncertainty"]
    assert far > near + 0.3


def test_nearest_demos_decide_the_parameters():
    from echotwin.robot.features.prop_skills import PropSkills
    s = PropSkills()
    tall = {"width": 0.07, "height": 0.10, "length": 0.07, "mass": 0.3}
    s.demos = [{**_demo(BOX5), "lift": 0.03}, {**_demo(tall), "lift": 0.12}]
    assert s.plan(BOX5, 0.2)["lift"] == pytest.approx(0.03)
    assert s.plan(tall, 0.2)["lift"] == pytest.approx(0.12)


def test_old_episodes_without_measurements_still_load():
    from echotwin.robot.features.prop_skills import PropSkills
    import numpy as np
    frames = []
    for i in range(10):                                   # hand goes down, grips, lifts, releases
        carrying = 1.0 if 3 <= i < 8 else 0.0
        frames.append({"state": [0, 0, 0.1, carrying, 0.1 * i, 0, 0.02, 0, 0, carrying], "action": [0.1, 0, 0, carrying]})
    ep = {"success": True, "task": {"kind": "prop", "h": 0.02, "tallest": 0.1}, "frames": frames}   # no shape, no m
    s = PropSkills()
    s.fit([ep, {"success": True, "task": {"kind": "other"}, "frames": []}])
    assert len(s.demos) == 1 and s.demos[0]["m"]["legacy"]
