"""While the AI names the objects, the scan summary says so (the pages show "naming..."), and it stops saying so after."""
import asyncio
import itertools

import cv2
import numpy as np
import pytest

from echotwin.robot.twin_import import photos as P
from echotwin.robot.twin_import.contract import TwinContext


def _photo() -> bytes:
    img = np.full((540, 960, 3), (190, 205, 215), np.uint8)
    cv2.ellipse(img, (300, 380), (70, 55), 0, 0, 360, (40, 40, 45), -1)
    cv2.rectangle(img, (600, 340), (760, 450), (40, 40, 45), -1)
    return cv2.imencode(".jpg", img)[1].tobytes()


def _run(tmp_path, answer):
    done, said = [], []
    ids = itertools.count()

    def new_dir(prefix=""):
        d = tmp_path / f"scan{next(ids)}"
        d.mkdir()
        return d.name, d

    async def ask(jpeg, prompt):
        return answer
    ctx = TwinContext(apply=lambda lay, s: None, rename=lambda props, line: None, say=said.append,
                      progress=lambda stage, data: done.append(dict(data["summary"])) if stage == "done" else None,
                      new_dir=new_dir, ask_ai_json=ask if answer != "no ai" else None)
    asyncio.run(P.import_photos([_photo()], [None], ctx))
    return done, said


def test_naming_is_on_until_the_ai_answers(tmp_path):
    done, _ = _run(tmp_path, {"objects": [{"id": 1, "name": "black case", "shape": "round"}, {"id": 2, "name": "book", "shape": "box"}]})
    assert [d["naming"] for d in done] == [True, False]
    assert all(n.startswith("object ") for n in done[0]["props"]) and sorted(done[1]["props"]) == ["black case", "book"]


def test_naming_stops_when_the_ai_has_no_answer(tmp_path):
    done, said = _run(tmp_path, None)
    assert [d["naming"] for d in done] == [True, False] and "can't tell what they are" in said[-1]


def test_without_ai_there_is_no_naming(tmp_path):
    done, _ = _run(tmp_path, "no ai")
    assert [d["naming"] for d in done] == [False]
