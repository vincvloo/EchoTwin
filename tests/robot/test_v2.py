"""Robot tests: router, everyday objects, done / do / teach per object type, practice, video demos."""
import cv2
import numpy as np
import pytest

from echotwin.robot.router import route


def test_router_order():
    assert route("stop").kind == "safety"
    assert route("wait, stop moving the glass").kind == "safety"  # stop word never goes elsewhere
    assert route("why did you stop?").kind == "robot_q"
    assert route("continue").name == "continue"
    assert route("what do you see?").kind == "scene_q"
    assert route("how sure are you").name == "sure"
    assert route("what is object 2").name == "which"
    assert route("tell me a joke").kind == "other"


@pytest.fixture(scope="module")
def sim():
    import tempfile
    from pathlib import Path

    from echotwin.robot import dataset as D
    from echotwin.robot.sim import Sim
    tmp = Path(tempfile.mkdtemp())
    orig = D.EPISODES
    D.Dataset.__init__.__defaults__ = (tmp,)
    s = Sim(lambda m: None)
    yield s
    D.Dataset.__init__.__defaults__ = (orig,)



# ---------------- everyday objects ----------------
def test_everyday_objects_on_plain_table():
    from echotwin.robot.features import everyday as E
    rng = np.random.default_rng(0)
    img = np.full((960, 540, 3), 225, np.uint8)
    img[:] += np.linspace(0, 20, 960, dtype=np.uint8)[:, None, None]      # lighting gradient
    img = cv2.add(img, rng.integers(0, 6, img.shape, dtype=np.uint8))
    img[:120] = (60, 70, 90)                                                # background behind the table
    cv2.ellipse(img, (330, 560), (30, 45), 20, 0, 360, (30, 30, 30), -1)   # black case
    cv2.rectangle(img, (120, 620), (200, 700), (245, 245, 250), -1)        # white packet with print
    cv2.putText(img, "Zurich", (125, 665), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 60, 30), 2)
    res = E.analyse(img, 50)
    assert len(res["items"]) == 2, [i["box"] for i in res["items"]]
    E.apply_names(res, {"objects": [{"id": 1, "name": "black case", "shape": "box"},
                                    {"id": 2, "name": "chocolate", "shape": "flat"}]})
    assert E.listing([i["name"] for i in res["items"]]) in ("a black case and a chocolate", "a chocolate and a black case")


# ---------------- everyday objects: done, do, or teach me per object type ----------------
def _everyday_layout():
    from echotwin.robot.scene import Layout
    lay = Layout()
    lay.props = [
        {"name": "glass of water", "pos": (-0.1, 0.05), "size": (0.035, 0.035, 0.07), "rgb": (0.8, 0.8, 0.8), "shape": "cylinder"},
        {"name": "black earbud case", "pos": (0.2, -0.05), "size": (0.035, 0.03, 0.025), "rgb": (0.1, 0.1, 0.1), "shape": "box"},
        {"name": "chocolate bar", "pos": (-0.25, -0.05), "size": (0.06, 0.035, 0.01), "rgb": (0.9, 0.9, 0.9), "shape": "flat"},
    ]
    return lay


def _human_demo(sim, task):
    """A scripted 'human' drives the gripper with the same inputs the phone sends."""
    w = sim.world

    def drive(target, grip, ticks=300):
        for _ in range(ticks):
            d = np.array(target) - w.hand_pos()
            if np.linalg.norm(d) < 0.005:
                break
            sim.set_human(*(d * 12))
            sim.human_grip = grip
            sim._tick()

    o = w.obj_pos(task["object"])
    g, h = task["goal"], task["h"]
    drive([o[0], o[1], 0.085], False)
    drive([o[0], o[1], 0.004], False)
    for _ in range(15):
        sim.human_grip = True
        sim._tick()
    drive([o[0], o[1], 0.085], True)
    drive([g[0], g[1], 0.085], True)
    drive([g[0], g[1], h + 0.006], True)
    for _ in range(30):
        sim.set_human(0, 0, 0)
        sim.human_grip = False
        sim._tick()
        if sim.mode == "review":
            break


def test_everyday_teach_per_object_type(sim):
    from echotwin.robot.features import move_things as MT
    sim._rebuild(_everyday_layout())
    sim.base_layout = sim.world.layout.copy()
    sim.world.settle(20)
    sim.skills.demos = {}
    out = []
    sim.emit = out.append
    sim.voice.emit = sim.emit
    decisions = lambda: [m for m in out if m.get("t") == "decision"]
    said = lambda: [m["text"] for m in out if m.get("t") == "log"]
    text = "put the chocolate next to the glass"

    # 1. never moved a flat thing -> teach me
    sim.handle_prop_task(MT.parse(text, sim.world.layout.props), text)
    assert decisions()[-1]["kind"] == "teach" and decisions()[-1]["reason"] == "new_kind", said()[-2:]
    # 2. the human shows it once; keep
    _human_demo(sim, sim.task)
    assert sim.mode == "review", said()[-3:]
    sim.keep(True)
    assert sim.skills.count("flat") == 1
    # 3. same instruction on the reset table: plan, imagine, do it
    sim.handle_prop_task(MT.parse(text, sim.world.layout.props), text)
    assert decisions()[-1]["kind"] == "do", (decisions()[-1], said()[-2:])
    assert "imagined" in decisions()[-1]["why"]
    for _ in range(900):
        if sim.mode != "move":
            break
        sim._tick()
    assert said()[-1] == "Done.", said()[-3:]
    # 4. asked again: already done
    sim.handle_prop_task(MT.parse(text, sim.world.layout.props), text)
    assert decisions()[-1]["kind"] == "done", said()[-2:]
    # 5. the glass is another kind of object: teach me again
    t2 = "move the glass closer"
    sim.handle_prop_task(MT.parse(t2, sim.world.layout.props), t2)
    assert decisions()[-1]["kind"] == "teach" and decisions()[-1]["reason"] == "new_kind", decisions()[-1]



def _table_photo(case_at, shift=(0, 0), rot=0.0, zoom=1.0):
    img = np.full((960, 540, 3), 228, np.uint8)
    img[:] += np.linspace(0, 18, 960, dtype=np.uint8)[:, None, None]
    img[:110] = (60, 70, 90)
    cv2.rectangle(img, (110, 600), (190, 680), (245, 245, 250), -1)       # wrapper A
    cv2.putText(img, "Zurich", (114, 645), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 60, 30), 2)
    cv2.rectangle(img, (330, 330), (400, 390), (40, 120, 210), -1)        # orange box
    cv2.ellipse(img, case_at, (26, 38), 0, 0, 360, (30, 30, 30), -1)       # black case
    M = cv2.getRotationMatrix2D((270, 480), rot, zoom)
    M[:, 2] += shift
    return cv2.warpAffine(img, M, (540, 960), borderMode=cv2.BORDER_REPLICATE)


def test_video_demo_everyday_objects(sim):
    from echotwin.robot.features import everyday as E
    from echotwin.robot.features import video_everyday as VE
    from echotwin.robot.scene import Layout
    start = _table_photo((400, 560))
    end = _table_photo((230, 640), shift=(10, -15), rot=3, zoom=1.05)   # case now right next to the wrapper
    frames = [start] * 6 + [end] * 6
    res = VE.analyse(frames)
    assert "error" not in res, res
    A = res["start_items"]
    moved_box = A[res["moved"]]["box"]
    assert abs(moved_box[0] + moved_box[2] / 2 - 400) < 30, moved_box       # the case moved
    assert res["ref"] is not None, res                                      # ... next to something
    # the twin built from the first frame; the move maps onto it and is replayed as a demo
    ev = E.analyse(start, None)
    E.apply_names(ev, {"objects": [{"id": i + 1, "name": f"thing {i + 1}", "shape": "box"} for i in range(len(ev["items"]))]})
    import tempfile
    from pathlib import Path
    props, tex, _ = E.build(ev, Path(tempfile.mkdtemp()))
    lay = Layout()
    lay.props, lay.texture = props, tex
    sim._rebuild(lay)
    sim.base_layout = sim.world.layout.copy()
    sim.world.settle(20)
    sim.skills.demos = {}
    plan = VE.map_to_twin(res, sim.world.layout.props)
    assert plan and plan["goal"][0] == "near", plan
    out = []
    sim.emit = out.append
    sim.voice.emit = sim.emit
    sim.play_prop_video(plan, "watching")
    for _ in range(900):
        if sim.mode != "replay":
            break
        sim._tick()
    said = [m["text"] for m in out if m.get("t") == "log"]
    assert sim.mode == "review" and sim.review["source"] == "video", said[-2:]
    sim.keep(True)
    assert sim.skills.count("box") == 1


def test_practice_and_stacking(sim):
    from echotwin.robot.features import move_things as MT
    sim._rebuild(_everyday_layout())
    sim.base_layout = sim.world.layout.copy()
    sim.world.settle(20)
    sim.skills.demos = {}
    out = []
    sim.emit = out.append
    sim.voice.emit = sim.emit
    said = lambda: [m["text"] for m in out if m.get("t") == "log"]
    # practice creates demos per kind of object, only successful ones
    sim.practice("flat", per_kind=2, live=False)
    assert sim.skills.count("flat") >= 1, said()[-1]
    assert "practised" in said()[-1]
    # "on top of" is stacking, not "next to"
    plan = MT.parse("take the chocolate and put it on top of the case", sim.world.layout.props)
    assert plan["goal"] == ("near", 1, "on top of"), plan
    sim.handle_prop_task(plan, "chocolate on the case")
    for _ in range(900):
        if sim.mode != "move":
            break
        sim._tick()
    sim.world.settle(20)
    choc, case = sim.world.obj_pos("prop_2"), sim.world.obj_pos("prop_1")
    assert said()[-1] == "Done." or "Can you show me" in said()[-1], said()[-2:]
    if said()[-1] == "Done.":
        assert choc[2] > case[2], (choc, case)
