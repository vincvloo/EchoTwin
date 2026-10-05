"""The step logic shared by the dashboard and the phone page (apps/robot_ui/ui/stage.js), run under node."""
import json
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from echotwin.robot import config, server

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")

STAGE = config.STATIC / "ui" / "stage.js"


def run(expr: str):
    js = f"const S = require({json.dumps(str(STAGE))}); console.log(JSON.stringify({expr}));"
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True, encoding="utf-8", check=True, timeout=20)
    return json.loads(out.stdout)


PROPS = [{"name": "glass"}]


@pytest.mark.parametrize("state, opts, stage", [
    (None, {}, "connecting"),
    ({"props": []}, {}, "scan"),
    ({"props": PROPS, "mode": "idle"}, {}, "ask"),
    ({"props": PROPS, "mode": "move"}, {}, "watch"),
    ({"props": PROPS, "mode": "practice"}, {}, "watch"),
    ({"props": PROPS, "mode": "replay"}, {}, "watch"),
    ({"props": PROPS, "mode": "teach"}, {}, "teach"),
    ({"props": PROPS, "mode": "review"}, {}, "review"),
    ({"props": [], "mode": "idle"}, {"scanning": True}, "scanning"),
    ({"props": PROPS, "mode": "move", "halted": True}, {"scanning": True}, "stopped"),   # stop beats everything
])
def test_the_stage_follows_the_server_state(state, opts, stage):
    assert run(f"S.stageOf({json.dumps(state)}, {json.dumps(opts)})") == stage


def test_every_stage_has_a_title_and_a_step_number():
    steps = run("S.STEPS")
    for k in ("connecting", "scan", "scanning", "ask", "watch", "teach", "review", "stopped"):
        assert steps[k]["title"]
    assert [steps[k]["n"] for k in ("scan", "ask", "watch", "teach")] == [1, 2, 3, 4]


def test_the_badge_says_what_is_happening():
    b = lambda s: run(f"S.badge({json.dumps(s)})")
    assert b({"halted": True, "halt_reason": "button"}) == {"text": "Stopped (button)", "tone": "stop"}
    assert b({"mode": "move", "name": "Pip", "task": {"name": "glass"}, "attempt": 2})["text"] == "Pip is moving the glass · try 2"
    assert b({"mode": "move", "name": "Pip", "task": {"name": "glass"}, "attempt": 1})["text"] == "Pip is moving the glass"
    assert b({"mode": "teach", "recording": 12})["text"] == "Recording your demo"
    assert b({"mode": "teach", "recording": 0})["text"] == "Your turn: show me"
    assert b({"mode": "review", "review": {"score": 87}})["text"] == "Demo scored 87"
    assert b({"mode": "idle", "authority": "human"}) == {"text": "Ready", "tone": "idle"}
    assert run("S.badge(null)")["text"] == "Connecting…"


def test_reasons_and_confidence_have_plain_words():
    assert run("S.reasonText('new_kind')").startswith("I haven't moved anything this size")
    assert run("S.reasonText('something_new')") == "something_new"          # an unknown reason is shown as is
    assert [run(f"S.sureLabel({p})") for p in (90, 50, 20)] == ["confident", "fairly sure", "not sure"]


def test_the_shared_style_and_script_are_served():
    c = TestClient(server.app)
    css = c.get("/ui/ui.css")
    assert css.status_code == 200 and "prefers-color-scheme:dark" in css.text and "text/css" in css.headers["content-type"]
    js = c.get("/ui/stage.js")
    assert js.status_code == 200 and "stageOf" in js.text


def test_pages_and_scripts_are_revalidated_not_cached_blindly():
    c = TestClient(server.app)
    for url in ("/", "/phone", "/ui/ui.css", "/ui/stage.js"):
        assert c.get(url).headers["cache-control"] == "no-cache", url
