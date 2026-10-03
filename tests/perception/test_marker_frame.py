"""The object map in the frame the marker defines: true size, table top at z = 0, calibration in scene.json."""
import json

import numpy as np
import pytest
import trimesh

from echotwin.perception import mapping
from echotwin.perception.objects import main as objects_main
from echotwin.scene import schema

SCALE = 0.5                       # metres per cloud unit
R_UP = mapping.UP["y"]


def tilt(deg_x, deg_y, deg_z):
    a, b, c = np.radians([deg_x, deg_y, deg_z])
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def make_scan(tmp_path, reliable=True):
    """A table top at z = 0 with a cup (7 cm wide, 10 cm tall) and a book on it, seen in a tilted, shifted, unscaled cloud."""
    rng = np.random.default_rng(3)
    n = 30000
    desk = np.column_stack([rng.uniform(-0.4, 0.4, n), rng.uniform(-0.3, 0.3, n), rng.uniform(-0.004, 0.004, n)])
    cup = np.column_stack([rng.uniform(0.0, 0.07, 1500), rng.uniform(0.0, 0.07, 1500), rng.uniform(0.0, 0.10, 1500)]) + [0.1, 0.05, 0]
    book = np.column_stack([rng.uniform(0.0, 0.20, 8000), rng.uniform(0.0, 0.14, 8000), rng.uniform(0.0, 0.03, 8000)]) + [-0.3, -0.15, 0]
    table = np.vstack([desk, cup, book])                                       # metres, z up, marker at the origin
    Qt, s = tilt(9, -6, 35), np.array([0.7, -0.4, 1.1])
    cloud = (R_UP.T @ (Qt @ table.T + s[:, None])).T / SCALE                   # what VGGT + `upright` would give
    ply = tmp_path / "scan.ply"
    trimesh.PointCloud(cloud).export(ply)
    cls = np.r_[np.full(n, 60), np.full(1500, 41), np.full(8000, 73)].astype(np.int16)
    np.savez(ply.with_suffix(".labels.npz"), cls=cls, frame=rng.integers(0, 3, len(cls)).astype(np.int16),
             names=json.dumps({60: "desk", 41: "cup", 73: "book"}))
    marker = {"scale": SCALE, "side_units": 0.2, "photos_seen": 4, "photos_total": 5, "spread": 0.01, "reliable": reliable,
              "origin": (R_UP.T @ s / SCALE).tolist(), "x_axis": (R_UP.T @ Qt @ [1, 0, 0]).tolist(),
              "normal": (R_UP.T @ Qt @ [0, 0, 1]).tolist()}
    ply.with_suffix(".marker.json").write_text(json.dumps(marker))
    return ply


def run(tmp_path, *extra):
    ply = make_scan(tmp_path, reliable="--unreliable" not in extra)
    out = tmp_path / "out"
    objects_main([str(ply), "-o", str(out), "--up", "y", "--scale", str(SCALE), "--min-area", "0.001", "--min-views", "1",
                  "--res", "0.01", *[e for e in extra if e != "--unreliable"]])
    return schema.load(str(out) + "_scene.json")


def test_marker_gives_the_table_frame_and_true_sizes(tmp_path):
    doc = run(tmp_path)
    cup = [o for o in doc["objects"] if o["class"] == "cup"][0]
    assert cup["size_x"] == pytest.approx(0.07, abs=0.02) and cup["size_y"] == pytest.approx(0.07, abs=0.02)
    assert cup["height"] == pytest.approx(0.10, abs=0.02)
    assert (cup["x"], cup["y"]) == pytest.approx((0.135, 0.085), abs=0.03)       # in the marker's frame (x along its top edge)
    book = [o for o in doc["objects"] if o["class"] == "book"][0]
    assert (book["x"], book["y"]) == pytest.approx((-0.2, -0.08), abs=0.03)
    cal = doc["calibration"]
    assert cal["source"] == "marker" and cal["scale"] == SCALE and cal["photos_seen"] == 4


def test_cloud_to_map_takes_a_cloud_point_to_the_map(tmp_path):
    doc = run(tmp_path)
    M = np.array(doc["calibration"]["cloud_to_map"])
    cup = [o for o in doc["objects"] if o["class"] == "cup"][0]
    Qt, s = tilt(9, -6, 35), np.array([0.7, -0.4, 1.1])
    p_cloud = (R_UP.T @ (Qt @ np.array([0.135, 0.085, 0.0]) + s)) / SCALE
    p_map = M @ np.r_[p_cloud, 1.0]
    assert p_map[:2] == pytest.approx([cup["x"], cup["y"]], abs=0.03) and abs(p_map[2]) < 0.01


def test_unreliable_marker_is_ignored_and_the_old_levelling_is_used(tmp_path):
    doc = run(tmp_path, "--unreliable")
    assert doc["calibration"]["source"] == "estimate"
    assert any(o["class"] == "cup" for o in doc["objects"])


def test_scene_without_calibration_still_validates():
    doc = schema.build_scene([], {}, name="x")
    assert "calibration" not in doc
    schema.validate({**doc, "calibration": {"source": "estimate", "scale": 1.0}})
