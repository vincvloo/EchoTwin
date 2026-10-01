"""twin_import: layout files, 3D scans, pluggable photo importer."""
import asyncio
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import trimesh

from echotwin.robot import twin_import as TI
from echotwin.robot.scene import Layout
from echotwin.robot.world import World


def _everyday():
    lay = Layout()
    lay.meta["sim_scale"] = 2.0
    lay.props = [
        {"name": "glass of water", "shape": "cylinder", "pos": (-0.1, 0.1), "size": (0.05, 0.05, 0.07), "rgb": (0.8, 0.8, 0.8)},
        {"name": "chocolate box", "shape": "flat", "pos": (0.15, -0.05), "yaw": 30, "size": (0.06, 0.035, 0.006), "rgb": (0.9, 0.9, 0.85)},
    ]
    return lay


def test_save_and_load_round_trip():
    w = World(_everyday())
    w.settle(20)
    d = Path(tempfile.mkdtemp())
    data = TI.export_zip(w.layout, w, d / "out")
    doc = TI.read_upload(data, "twin.zip", d / "in")
    assert doc["format"] == "phone-puppeteer-twin" and len(doc["objects"]) == 2
    choc = doc["objects"][1]
    assert choc["name"] == "chocolate box" and abs(choc["size_cm"][0] - 6.0) < 0.01  # 0.12 sim m / 2 = 6 cm
    assert abs(choc["pos_cm"][0] - 7.5) < 0.5 and abs(choc["yaw_deg"] - 30) < 2
    lay = TI.doc_to_layout(doc, d / "in")
    w2 = World(lay)
    assert np.allclose(w2.layout.props[1]["size"], (0.06, 0.035, 0.006), atol=1e-4)


def test_hand_written_twin_file():
    doc = {"format": "phone-puppeteer-twin", "sim_scale": 2.0,
           "objects": [{"name": "mug", "shape": "cylinder", "size_cm": [8, 8, 10], "pos_cm": [0, 5]}],
           "blocks": [{"color": "red", "pos_cm": [-10, -5]}]}   # old block files still load; blocks are ignored
    d = Path(tempfile.mkdtemp())
    lay = TI.doc_to_layout(json.loads(json.dumps(doc)), d)
    w = World(lay)
    assert w.things() == ["prop_0"] and w.layout.props[0]["name"] == "mug"
    with pytest.raises(TI.ImportError_):
        TI.doc_to_layout({"objects": [{"name": "x", "shape": "banana", "pos_cm": [0, 0]}]}, d)


def test_3d_scan_as_object_and_as_scene():
    d = Path(tempfile.mkdtemp())
    box = trimesh.creation.box(extents=[0.10, 0.02, 0.06])  # a glTF (y-up) scan of a 10 x 6 x 2 cm thing
    glb = d / "choc.glb"
    glb.write_bytes(trimesh.Scene(box).export(file_type="glb"))
    info = TI.convert_mesh(glb, d / "meshes", "choc")
    assert info["faces"] > 0 and abs(info["extent"][2] - 0.02) < 1e-3  # y-up turned into z-up
    lay = _everyday()
    lay.props[1]["mesh"] = info["file"]
    lay.props[1]["mesh_scale"] = TI.fit_scale(info["file"], lay.props[1]["size"])
    lay.scene = [{"file": info["file"], "pos": (0, 0.6, -0.5), "euler": (0, 0, 0), "scale": 2.0}]
    w = World(lay)  # MuJoCo accepts the mesh object and the scenery
    w.settle(20)
    assert w.obj_pos("prop_1")[2] > 0.0


def test_point_cloud_is_refused():
    d = Path(tempfile.mkdtemp())
    pc = trimesh.PointCloud(np.random.default_rng(0).random((200, 3)))
    ply = d / "cloud.ply"
    ply.write_bytes(pc.export(file_type="ply"))
    with pytest.raises(TI.ImportError_):
        TI.convert_mesh(ply, d / "out", "cloud")


async def fake_importer(frames, pitches, ctx):
    lay = _everyday()
    sid, folder = ctx.new_dir()
    ctx.apply(lay, {"id": sid, "mode": "everyday", "props": ["glass of water", "chocolate box"], "greeting": "hi"})


def test_pluggable_photo_importer(monkeypatch):
    monkeypatch.setenv("TWIN_PHOTO_IMPORTER", "tests.robot.test_twin_import:fake_importer")
    got = {}
    ctx = TI.TwinContext(apply=lambda lay, s: got.update(lay=lay, s=s), rename=lambda *a: None, say=lambda t: None,
                         progress=lambda *a: None, new_dir=lambda: ("x", Path(tempfile.mkdtemp())))
    asyncio.run(TI.photo_importer()([b"jpeg"], [None], ctx))
    assert got["s"]["props"] == ["glass of water", "chocolate box"]
