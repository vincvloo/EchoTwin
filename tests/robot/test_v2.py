"""v2 tests: parser, router, three-way decision, and the scan pipeline on synthetic phone photos."""
import cv2
import numpy as np
import pytest

from echotwin.robot.features import tasks as T
from echotwin.robot.features import vision as V
from echotwin.robot.features.scan import run_scan
from echotwin.robot.router import route

# ---------------- F1 parser ----------------
PHRASINGS = [
    "Put the red block in the green zone",
    "put red in green",
    "move the red cube into the green area",
    "Place the red one on the green note.",
    "could you put the red block onto the green zone please",
    "bring the red brick to the green zone",
    "red block to the green zone",
    "take the red one and put it in the green square",
    "Pip, put the red block in the green zone",
    "drop the red block in the green one",
]


@pytest.mark.parametrize("text", PHRASINGS)
def test_ten_phrasings(text):
    t = T.parse_instruction(text)
    assert t and t["object"] == "red" and t["target"] == "green", (text, t)


def test_blue_block_blue_tray():
    t = T.parse_instruction("put the blue block in the blue tray")
    assert (t["object"], t["target"]) == ("blue", "tray")
    t = T.parse_instruction("put the yellow one on the tray")
    assert (t["object"], t["target"]) == ("yellow", "tray")


def test_missing_slots_ask():
    t = T.parse_instruction("move a block")
    assert t and t["object"] is None
    assert "Which block" in T.clarify_question(t)
    t = T.fill_slots(t, "the red one")
    assert t["object"] == "red" and "Where" in T.clarify_question(t)
    t = T.fill_slots(t, "blue tray")
    assert t["target"] == "tray" and T.clarify_question(t) is None


def test_router_order():
    assert route("stop").kind == "safety"
    assert route("wait, stop moving the red block").kind == "safety"  # stop word never goes to the task layer
    assert route("why did you stop?").kind == "robot_q"
    assert route("continue").name == "continue"
    assert route("put the red block in the green zone").kind == "task"
    assert route("what do you see?").kind == "scene_q"
    assert route("how sure are you").name == "sure"
    assert route("clean up").kind == "task"
    assert route("tell me a joke").kind == "other"


# ---------------- F2 decision ----------------
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


def _decision(sim, text):
    out = []
    sim.emit = lambda m: out.append(m)
    sim.voice.emit = sim.emit
    sim.handle_intent(route(text))
    return [m for m in out if m.get("t") == "decision"], [m["text"] for m in out if m.get("t") == "log"]


def test_teach_when_no_demos(sim):
    sim.reset_scene(announce=False)
    d, said = _decision(sim, "put the red block in the blue tray")
    assert d[0]["kind"] == "teach" and "Can you show me" in said[-1]


def test_do_when_learned(sim):
    sim.reset_scene(announce=False)
    d, said = _decision(sim, "put the red block in the green zone")
    assert d[0]["kind"] == "do", said
    assert sim.ghost is not None and len(sim.ghost) > 10


def test_already_done(sim):
    lay = sim.base_layout.copy()
    zx, zy = lay.zones["green"]["pos"]
    lay.objects["yellow"]["pos"] = (zx, zy)
    sim._rebuild(lay)
    d, said = _decision(sim, "put the yellow block in the green zone")
    assert d[0]["kind"] == "done" and said[-1].startswith("Already done")


# ---------------- F0/F4 scan on synthetic photos ----------------
PX_PER_MM = 2.0
WORLD_MM = (700, 520)
TRUTH = {"red": (-80, -45), "blue": (15, -55), "yellow": (95, -30), "green": (90, 52), "tray": (-90, 52)}


def world_image(rng, truth=None, hand=None):
    truth = {**TRUTH, **(truth or {})}
    w, h = int(WORLD_MM[0] * PX_PER_MM), int(WORLD_MM[1] * PX_PER_MM)
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (70, 110, 150)  # wooden table (BGR)
    img = cv2.add(img, rng.integers(0, 25, img.shape, dtype=np.uint8))

    def rect(cx, cy, sx, sy, col):
        u0, v0 = (cx - sx / 2 + WORLD_MM[0] / 2) * PX_PER_MM, (WORLD_MM[1] / 2 - cy - sy / 2) * PX_PER_MM
        cv2.rectangle(img, (int(u0), int(v0)), (int(u0 + sx * PX_PER_MM), int(v0 + sy * PX_PER_MM)), col, -1)

    rect(0, 0, 297, 210, (238, 240, 240))
    rect(*truth["green"], 76, 76, (80, 190, 60))
    rect(*truth["tray"], 76, 76, (210, 120, 30))
    rect(*truth["red"], 24, 24, (40, 40, 210))
    rect(*truth["blue"], 24, 24, (200, 70, 20))
    rect(*truth["yellow"], 24, 24, (30, 210, 240))
    if hand is not None:  # a skin-coloured hand over the carried block
        rect(hand[0] + 10, hand[1] - 45, 70, 90, (120, 150, 205))
    return img


def photo(world, az_deg, elev_deg=50, dist=520, size=(1024, 768)):
    """Perspective photo of the table plane from a camera on a circle around the sheet."""
    az, el = np.radians(az_deg), np.radians(elev_deg)
    C = dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    fwd = -C / np.linalg.norm(C)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])  # world -> camera
    t = -R @ C
    K = V.camera_matrix((size[1], size[0]))
    H_plane = K @ np.c_[R[:, 0], R[:, 1], t]  # (x_mm, y_mm, 1) -> image
    M = np.array([[1 / PX_PER_MM, 0, -WORLD_MM[0] / 2], [0, -1 / PX_PER_MM, WORLD_MM[1] / 2], [0, 0, 1]])
    H = H_plane @ M
    img = cv2.warpPerspective(world, H, size, borderValue=(40, 40, 40))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def test_sheet_found_and_pose_front():
    world = world_image(np.random.default_rng(0))
    img = V.decode(photo(world, -90))
    corners = V.find_sheet(img)
    assert corners is not None
    pose = V.choose_pose(V.pose_candidates(corners, img.shape), None)
    assert pose["cam"][1] < 0  # camera in front


@pytest.mark.parametrize("seed", range(5))
def test_sweep_scan_places_everything(seed):
    rng = np.random.default_rng(seed)
    world = world_image(rng)
    start = rng.uniform(-150, -110)
    azs = np.linspace(start, start + 120, 12)  # a half-circle-ish sweep starting at the front
    frames = [photo(world, a, elev_deg=rng.uniform(40, 60), dist=rng.uniform(450, 600)) for a in azs]
    layout, summary = run_scan(frames)
    objs = summary["objects"]
    for name, (x, y) in TRUTH.items():
        assert name in objs, (name, objs.keys())
        err_sim_cm = np.hypot(objs[name]["x"] - x, objs[name]["y"] - y) / 10 * 2  # sim scale 2
        assert err_sim_cm < 2.0, (name, objs[name], (x, y))
    assert layout.objects["red"]["present"] and layout.texture


def test_single_photo_fallback_topdown():
    world = world_image(np.random.default_rng(1))
    # a cropped straight-down photo of just the sheet: no sheet edges visible
    u0, v0 = int((WORLD_MM[0] / 2 - 148) * PX_PER_MM), int((WORLD_MM[1] / 2 - 104) * PX_PER_MM)
    crop = world[v0:v0 + int(208 * PX_PER_MM), u0:u0 + int(296 * PX_PER_MM)]
    ok, buf = cv2.imencode(".jpg", crop)
    _, summary = run_scan([buf.tobytes()])
    assert summary["fallback"] and "red" in summary["objects"]


# ---------------- learning from a video of the hand ----------------
def test_video_demo_blue_to_tray():
    from echotwin.robot.features import video_demo as VD
    rng = np.random.default_rng(3)
    start, end = np.array(TRUTH["blue"]), np.array(TRUTH["tray"]) + [8, -5]
    frames = []
    for i in range(40):  # 8 s at 5 fps: still, carry, still
        a = np.clip((i - 8) / 22, 0, 1)
        pos = start + (end - start) * a
        carrying = 0 < a < 1
        w = world_image(rng, {"blue": tuple(pos)}, hand=pos if carrying else None)
        jb = photo(w, -100 + rng.normal(0, 1.5), elev_deg=58 + rng.normal(0, 1), dist=520)
        frames.append(V.decode(jb))
    zones = {"green": TRUTH["green"], "tray": TRUTH["tray"]}
    res = VD.analyse(frames, 5.0, zones)
    assert "error" not in res, res
    assert res["object"] == "blue" and res["target"] == "tray" and res["in_zone"], res
    assert np.hypot(*(np.array(res["end"]) - end)) < 15
    d = VD.to_sim(res)
    assert 0.02 < d["speed"] < 0.5 and len(d["path"]) >= 3


def test_video_demo_replays_in_sim(sim):
    from echotwin.robot.features import video_demo as VD
    sim.reset_scene(announce=False)
    tray = np.array(sim.world.zone_pos("tray")) * 1000 / 2
    start = np.array(sim.world.obj_pos("blue")[:2]) * 1000 / 2
    path = [(t, *(start + (tray - start) * t / 3)) for t in np.linspace(0, 3, 10)]
    res = {"object": "blue", "target": "tray", "start": list(start), "end": list(tray), "path": path, "duration": 3.0}
    out = []
    sim.emit = out.append
    sim.voice.emit = sim.emit
    sim.play_video_demo(VD.to_sim(res), "watching")
    for _ in range(600):
        if sim.mode != "replay":
            break
        sim._tick()
    assert sim.mode == "review" and sim.review["source"] == "video", [m for m in out if m.get("t") == "log"]
    n = len(sim.dataset.episodes)
    sim.keep(True)
    assert len(sim.dataset.episodes) == n + 1 and sim.dataset.episodes[-1]["source"] == "video"
    assert sim.policy.demo_counts.get("tray", 0) >= 1


# ---------------- everyday objects (no sheet) ----------------
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
    lay.show_zones = False
    for o in lay.objects.values():
        o["present"] = False
    lay.props = [
        {"name": "glass of water", "pos": (-0.1, 0.1), "size": (0.05, 0.05, 0.07), "rgb": (0.8, 0.8, 0.8), "shape": "cylinder"},
        {"name": "black earbud case", "pos": (0.2, -0.05), "size": (0.035, 0.03, 0.025), "rgb": (0.1, 0.1, 0.1), "shape": "box"},
        {"name": "chocolate bar", "pos": (-0.2, -0.2), "size": (0.06, 0.035, 0.006), "rgb": (0.9, 0.9, 0.9), "shape": "flat"},
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
    drive([o[0], o[1], 0.2], False)
    drive([o[0], o[1], max(o[2], 0.013)], False)
    for _ in range(4):
        sim.human_grip = True
        sim._tick()
    drive([o[0], o[1], 0.22], True)
    drive([g[0], g[1], 0.22], True)
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
    t2 = "move the glass to the left"
    sim.handle_prop_task(MT.parse(t2, sim.world.layout.props), t2)
    assert decisions()[-1]["kind"] == "teach" and decisions()[-1]["reason"] == "new_kind"



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
    assert VE.frames_everyday_ok(frames)
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
    lay.props, lay.texture, lay.show_zones = props, tex, False
    for o in lay.objects.values():
        o["present"] = False
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
