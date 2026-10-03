"""The arm adapter, inverse kinematics, workspace, contact grasp and refusals (built-in arm, no download)."""
import json

import numpy as np
import pytest

from echotwin.robot import arm as A
from echotwin.robot.features import prop_skills as PS
from echotwin.robot.scene import Layout
from echotwin.robot.world import World


def _layout(*props):
    lay = Layout()
    lay.props = [{"name": n, "shape": s, "pos": p, "yaw": 0.0, "size": z, "rgb": (0.8, 0.3, 0.3)} for n, s, p, z in props]
    return lay


def _move(w, name="prop_0", goal=(0.13, 0.0)):
    h = w.half(name)
    task = {"object": name, "name": name, "goal": np.array(goal), "h": h}
    d, hs = w.clone()
    r = {"wps": PS.waypoints(w, task, dict(PS.DEFAULTS)), "i": 0, "speed": PS.DEFAULTS["speed"], "yaw": w.grasp_yaw(name)}
    for _ in range(1200):
        a = PS.waypoint_action(w, r, d, hs)
        if a is None:
            break
        w.step(a, d, hs)
    for _ in range(30):
        w.step(np.array([0, 0, 0, 0.0, r["yaw"]]), d, hs)
    return w.obj_pos(name, d)


def test_ik_reaches_random_workspace_points():
    w = World(_layout(("a", "box", (0.0, 0.0), (0.02, 0.02, 0.02))))
    rng = np.random.default_rng(1)
    for _ in range(8):
        x, y = w.workspace_sample(rng)
        q, ep, er = w.ik.solve(np.array([x, y, 0.02]), 0.0, w.ik.q_down, iters=300)
        assert ep < 0.005, (x, y, ep)
    assert not w.reachable((0.0, -0.29))                 # right on top of the base
    assert not w.reachable((0.0, 0.29))                  # beyond the arm's reach


@pytest.mark.parametrize("shape,half", [("box", (0.025, 0.02, 0.025)), ("cylinder", (0.03, 0.03, 0.04))])
def test_grasp_lifts_and_places(shape, half):
    w = World(_layout(("thing", shape, (0.0, -0.05), half), ("mark", "box", (0.2, 0.0), (0.02, 0.02, 0.02))))
    w.settle(20)
    end = _move(w)
    assert np.linalg.norm(end[:2] - np.array([0.13, 0.0])) < 0.04, end


def test_wide_box_is_refused_with_a_reason():
    w = World(_layout(("brick", "box", (0.0, -0.05), (0.06, 0.06, 0.03))))
    ok, why = w.can_grasp("prop_0")
    assert not ok and "12 cm wide" in why and "opens" in why
    assert "12 cm" in w.refusal("prop_0", (0.1, 0.0))


def test_out_of_reach_is_refused():
    w = World(_layout(("a", "box", (0.0, -0.22), (0.02, 0.02, 0.02))))     # next to the base: too close
    assert w.refusal("prop_0") == "it is out of my reach"
    w2 = World(_layout(("a", "box", (0.0, 0.0), (0.02, 0.02, 0.02))))
    assert w2.refusal("prop_0", (0.38, 0.28)) == "the spot is out of my reach"
    assert w2.refusal("prop_0", (0.1, 0.0)) == ""


def test_clone_reproduces_a_run():
    w = World(_layout(("thing", "box", (0.0, -0.05), (0.025, 0.02, 0.025))))
    w.settle(20)
    a = _move(w)
    b = _move(w)
    assert np.allclose(a, b, atol=1e-6)


def test_mass_comes_from_size():
    from echotwin.robot.scene import prop_mass
    assert prop_mass({"size": (0.01, 0.01, 0.01)}) == pytest.approx(0.02)      # floor
    assert prop_mass({"size": (0.05, 0.05, 0.05), "shape": "box"}) == pytest.approx(0.3)
    assert prop_mass({"size": (0.2, 0.2, 0.2)}) == pytest.approx(0.4)          # ceiling


# ---------------- the descriptor ----------------
def test_descriptor_errors_are_plain():
    good = json.loads((A.ARMS_DIR / "builtin.json").read_text())
    for key in ("joints", "gripper", "tool"):
        bad = {k: v for k, v in good.items() if k != key}
        with pytest.raises(A.ArmError, match=key):
            A.parse(bad)
    with pytest.raises(A.ArmError, match="5 joints"):
        A.parse({**good, "joints": good["joints"][:4]})
    with pytest.raises(A.ArmError, match="No arm descriptor"):
        A.load("nope")


def test_missing_joint_name_gives_a_clear_error():
    good = json.loads((A.ARMS_DIR / "builtin.json").read_text())
    bad = A.parse({**good, "joints": ["pan", "lift", "elbow", "wflex", "no_such_joint"]})
    with pytest.raises(A.ArmError, match="no_such_joint"):
        World(_layout(("a", "box", (0.0, 0.0), (0.02, 0.02, 0.02))), bad)


def test_a_second_arm_with_other_names_and_order_works(tmp_path):
    """The adapter goes by names from the descriptor, not by the order inside the MJCF."""
    renames = {"pan": "j_base", "lift": "j_shoulder", "elbow": "j_elbow", "wflex": "j_wrist", "roll": "j_roll",
               "pad_l": "jaw_a", "pad_r": "jaw_b"}
    xml = A.BUILTIN_MJCF
    for old, new in renames.items():
        xml = xml.replace(f'"{old}"', f'"{new}"')
    a, b = xml.index("<actuator>"), xml.index("</actuator>")
    lines = [ln for ln in xml[a + len("<actuator>"):b].strip().splitlines()]
    xml = xml[:a] + "<actuator>\n" + "\n".join(reversed(lines)) + "\n  " + xml[b:]    # actuators declared backwards
    (tmp_path / "other.xml").write_text(xml, encoding="utf-8")
    d = json.loads((A.ARMS_DIR / "builtin.json").read_text())
    d.update(name="other", mjcf=str(tmp_path / "other.xml"), joints=["j_base", "j_shoulder", "j_elbow", "j_wrist", "j_roll"],
             actuators=["j_base", "j_shoulder", "j_elbow", "j_wrist", "j_roll"], pads=["jaw_"])
    arm = A.parse(d)
    w = World(_layout(("thing", "box", (0.0, -0.05), (0.025, 0.02, 0.025)), ("m", "box", (0.2, 0.0), (0.02, 0.02, 0.02))), arm)
    w.settle(20)
    end = _move(w)
    assert np.linalg.norm(end[:2] - np.array([0.13, 0.0])) < 0.04, end
