"""What the robot says about itself and the table (answers.py), the practice overlay and the rendered frame (render.py)."""
import re
import tempfile
from pathlib import Path

import numpy as np
import pytest

from echotwin.robot import answers, render
from echotwin.robot.router import route


@pytest.fixture(scope="module")
def sim():
    from echotwin.robot import dataset as D
    from echotwin.robot.scene import Layout
    from echotwin.robot.sim import Sim
    orig = D.EPISODES
    D.Dataset.__init__.__defaults__ = (Path(tempfile.mkdtemp()),)
    s = Sim(lambda m: None)
    lay = Layout()
    lay.props = [
        {"name": "cube", "shape": "box", "pos": (-0.1, -0.05), "yaw": 0.0, "size": (0.025, 0.02, 0.025), "rgb": (0.8, 0.3, 0.3)},
        {"name": "mark", "shape": "box", "pos": (0.12, 0.0), "yaw": 0.0, "size": (0.02, 0.02, 0.02), "rgb": (0.2, 0.3, 0.8)}]
    s._rebuild(lay)
    s.base_layout = s.world.layout.copy()
    s.world.settle(20)
    yield s
    D.Dataset.__init__.__defaults__ = (orig,)


def ask(sim, text):
    it = route(text)
    fn = {"robot_q": lambda: answers.about_robot(sim, it.name),
          "scene_q": lambda: answers.about_table(sim.world, it.name, {**it.data, "text": it.text})}[it.kind]
    return fn()


def test_the_robot_answers_about_the_table(sim):
    assert ask(sim, "what do you see?") == "On your table I see a cube and a mark."
    assert ask(sim, "how many things are there") == "I see 2 things on the table."
    assert ask(sim, "which is object 2") == "Object 2 is mark."
    assert ask(sim, "what is object 9") == "I have 2 objects, numbered 1 to 2."
    assert re.fullmatch(r"The cube is (at|in) the \w+ \w+ of the table\.", ask(sim, "where is the cube?"))
    assert "The mark is" in ask(sim, "where are they")                      # no object named: all of them


def test_the_robot_answers_about_itself(sim):
    assert ask(sim, "who are you").startswith("I'm Pip.")
    assert ask(sim, "help") == "Try: put the cube next to the mark. Or say stop, anytime."
    assert ask(sim, "what are you doing") == "Waiting for an instruction."
    assert ask(sim, "why did you stop?") == "I haven't stopped."
    assert ask(sim, "how sure are you?") == "Tell me what to move, and I'll tell you how sure I am."


def test_why_it_stopped_follows_the_reason(sim):
    sim.halted = True
    try:
        for reason, said in [("button", "You pressed stop."), ("shake", "You shook the phone."), ("fell", "Because fell.")]:
            sim.halt_reason = reason
            assert ask(sim, "why did you stop?") == said
        assert ask(sim, "what are you doing") == "Nothing. I'm stopped. Say continue when ready."
    finally:
        sim.halted = False


def test_without_a_table_it_asks_for_a_scan():
    from echotwin.robot.scene import Layout
    from echotwin.robot.world import World
    w = World(Layout())
    assert answers.about_table(w, "see", {}) == "I don't see anything yet. Snap a photo of your table first."
    assert answers.about_table(w, "which", {"index": 1}) == "I haven't mapped any objects. Snap a photo of your table first."


def test_the_practice_overlay_reports_progress(sim):
    from echotwin.robot.practice import Practice
    p = Practice(sim, ["flat", "small"], per_kind=3, prop=None)
    assert p.overlay() == ("PRACTICE 1/3 · flat things · try 0", None)
    p.kept, p.tries, p.cur = 2, 5, {"text": "put the cube next to the mark"}
    assert p.overlay() == ("PRACTICE 3/3 · flat things · try 5", "put the cube next to the mark")
    p.k = 1
    assert "small things" in p.overlay()[0]


@pytest.mark.parametrize("overlay", [None, ("halted", "button"), ("practice", "PRACTICE 1/3", "put it there"), ("rec",)])
def test_a_frame_is_a_jpeg_with_or_without_a_banner(sim, overlay):
    try:
        r = render.FrameRenderer(sim.world.model)
    except Exception as e:                                                  # no OpenGL on this machine
        pytest.skip(f"no renderer: {e}")
    ghost = np.array([[0.0, 0.0, 0.1], [0.05, 0.0, 0.1], [0.1, 0.0, 0.1], [0.15, 0.0, 0.1]])
    jpeg = r.jpeg(sim.world.view, ghost, 0.5, overlay)
    assert jpeg[:2] == b"\xff\xd8" and len(jpeg) > 5000
    r.reopen(sim.world.view.model)                                          # the twin was rebuilt
    assert r.jpeg(sim.world.view, None, 0.5, overlay)[:2] == b"\xff\xd8"
