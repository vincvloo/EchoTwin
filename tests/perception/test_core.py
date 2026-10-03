"""Cloud levelling, floor plan grid, camera-based up axis and the object map. Run with:  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage


from echotwin.perception.gridmap import GridMap
import trimesh                                           # noqa: E402

from echotwin.perception import mapping as mesh_to_grid
from echotwin.perception.mapping import (UP, align_walls, detect_up, level_floor, load_points,
                                         points_to_grid, voxel_downsample)


def box_room(w=4.0, h=3.0, res=0.02):
    """Empty rectangular room with 1-cell walls, origin at (0, 0)."""
    W, H = int(w / res), int(h / res)
    occ = np.zeros((H, W), bool)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    return GridMap(occ=occ, res=res, origin=(0.0, 0.0))


def test_level_floor_recovers_tilt():
    rng = np.random.default_rng(0)
    floor = np.column_stack([rng.uniform(0, 4, 20000), rng.uniform(0, 3, 20000), np.zeros(20000)])
    wall = np.column_stack([np.full(5000, 4.0), rng.uniform(0, 3, 5000), rng.uniform(0, 2.5, 5000)])
    pts = np.vstack([floor, wall])
    a = np.deg2rad(3)
    R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    _, tilt, _, _ = level_floor(pts @ R.T + [0, 0, 1.2], rng)
    assert abs(tilt - 3) < 0.3


def test_points_to_grid_marks_wall_and_ignores_high_shelf():
    rng = np.random.default_rng(1)
    floor = np.column_stack([rng.uniform(0, 4, 40000), rng.uniform(0, 3, 40000), np.zeros(40000)])
    wall = np.column_stack([np.full(8000, 2.0), rng.uniform(0, 3, 8000), rng.uniform(0, 2.5, 8000)])
    shelf = np.column_stack([rng.uniform(0.5, 1.0, 4000), rng.uniform(1, 2, 4000), np.full(4000, 1.0)])
    g = points_to_grid(np.vstack([floor, wall, shelf]), res=0.05)
    r, c = g.to_cell(np.array([2.0, 0.75]), np.array([1.5, 1.5]))
    assert g.occ[r[0], c[0]]          # wall is an obstacle
    assert not g.occ[r[1], c[1]]      # shelf at 1 m is above the height band


def test_align_walls_undoes_yaw():
    rng = np.random.default_rng(2)
    wall = np.column_stack([np.full(5000, 2.0), rng.uniform(0, 3, 5000), rng.uniform(0.1, 2, 5000)])
    wall2 = np.column_stack([rng.uniform(0, 4, 5000), np.full(5000, 3.0), rng.uniform(0.1, 2, 5000)])
    pts = np.vstack([wall, wall2])
    a = np.deg2rad(17)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    _, yaw, _ = align_walls(pts @ Rz.T)
    assert min(abs((yaw + 17) % 90), abs((yaw + 17) % 90 - 90)) < 1.0


def room_points(rng, n=40000, gap=None):
    """Z-up 4 x 3 m room: floor + 4 walls 0-2.5 m. `gap` = (y0, y1) hole in the x=4 wall."""
    floor = np.column_stack([rng.uniform(0, 4, n), rng.uniform(0, 3, n), np.zeros(n)])
    walls = []
    for x in (0.0, 4.0):
        y = rng.uniform(0, 3, n // 4)
        if gap and x == 4.0:
            y = y[(y < gap[0]) | (y > gap[1])]
        walls.append(np.column_stack([np.full(len(y), x), y, rng.uniform(0, 2.5, len(y))]))
    for y in (0.0, 3.0):
        walls.append(np.column_stack([rng.uniform(0, 4, n // 4), np.full(n // 4, y), rng.uniform(0, 2.5, n // 4)]))
    return np.vstack([floor, *walls])


def test_detect_up_finds_axis_and_sign():
    rng = np.random.default_rng(3)
    pts = room_points(rng)
    for up in ("y", "z", "-x"):
        in_file = pts @ UP[up]               # inverse of the file -> Z-up rotation
        assert detect_up(in_file, np.random.default_rng(0)) == up


def test_load_points_auto_up_and_voxel_downsample(tmp_path, monkeypatch):
    rng = np.random.default_rng(4)
    pts = room_points(rng, n=80000)
    f = tmp_path / "zup_color.ply"
    col = np.full((len(pts), 4), 200, np.uint8)
    trimesh.PointCloud(pts, colors=col).export(f)
    monkeypatch.setattr(mesh_to_grid, "VOXEL_ABOVE", 50_000)
    out, info = load_points(str(f), up=None, return_info=True)
    assert info["up"] == "z" and info["n_raw"] == len(pts)
    assert info["n"] < len(pts)
    assert np.allclose(out.min(0), pts.min(0), atol=0.03) and np.allclose(out.max(0), pts.max(0), atol=0.03)
    d = voxel_downsample(pts, 0.02)
    assert len(np.unique(np.floor((d - pts.min(0)) / 0.02), axis=0)) == len(d)


def test_load_points_multi_mesh_glb(tmp_path):
    scene = trimesh.Scene()
    a = trimesh.creation.box(extents=[1, 1, 1])
    a.visual.vertex_colors = [255, 0, 0, 255]
    scene.add_geometry(a, node_name="a")
    scene.add_geometry(trimesh.creation.box(extents=[1, 1, 1]), node_name="b",
                       transform=trimesh.transformations.translation_matrix([5, 0, 0]))
    f = tmp_path / "scene.glb"
    scene.export(f)
    pts = load_points(str(f), n=20000, up="z")
    assert pts[:, 0].min() < -0.4 and pts[:, 0].max() > 5.4      # both meshes, node transform applied
    assert ((pts[:, 0] > 0.6) & (pts[:, 0] < 4.4)).sum() == 0


def test_detect_up_when_ceiling_is_the_largest_plane():
    rng = np.random.default_rng(6)
    pts = room_points(rng)
    pts = pts[(pts[:, 2] > 0.01) | (rng.random(len(pts)) < 0.4)]      # patchy floor
    ceiling = np.column_stack([rng.uniform(0, 4, 40000), rng.uniform(0, 3, 40000), np.full(40000, 2.5)])
    bed = np.column_stack([rng.uniform(0, 2, 15000), rng.uniform(0, 1.6, 15000), rng.uniform(0, 0.5, 15000)])
    pts = np.vstack([pts, ceiling, bed])
    for up in ("y", "z"):
        assert detect_up(pts @ UP[up], np.random.default_rng(0)) == up


def test_video_upright_from_camera_axes():
    """Cameras pitched down 25-40 deg, yawing around, no roll, world 'up' along some odd direction."""
    from echotwin.perception.reconstruct import upright
    rng = np.random.default_rng(9)
    up_true = np.array([0.3, -0.5, 0.81]); up_true /= np.linalg.norm(up_true)
    t = np.cross(up_true, [1.0, 0, 0]); t /= np.linalg.norm(t); b = np.cross(up_true, t)
    extr = []
    for yaw, pitch in zip(rng.uniform(0, 2 * np.pi, 20), np.deg2rad(rng.uniform(25, 40, 20))):
        h = np.cos(yaw) * t + np.sin(yaw) * b                     # horizontal viewing direction
        z = np.cos(pitch) * h - np.sin(pitch) * up_true           # camera looks forward and down
        x = np.cross(z, up_true); x /= np.linalg.norm(x)           # horizontal (no roll)
        y = np.cross(z, x)                                         # OpenCV: y points down in the image
        extr.append(np.column_stack([np.vstack([x, y, z]), np.zeros(3)]))
    R = upright(np.array(extr))
    assert np.allclose(R @ up_true, [0, 1, 0], atol=1e-6)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)


def test_video_upright_tabletop_shot_from_one_side():
    """All cameras face the same way (x axes parallel): 'up' must come from the table plane."""
    from echotwin.perception.reconstruct import upright
    rng = np.random.default_rng(13)
    up_true = np.array([0.2, 0.9, -0.3]); up_true /= np.linalg.norm(up_true)
    t = np.cross(up_true, [1.0, 0, 0]); t /= np.linalg.norm(t); b = np.cross(up_true, t)
    table = rng.uniform(-0.5, 0.5, (5000, 1)) * t + rng.uniform(-0.3, 0.3, (5000, 1)) * b
    extr = []
    for pitch, side in zip(np.deg2rad(rng.uniform(40, 70, 8)), rng.uniform(-0.1, 0.1, 8)):
        z = np.cos(pitch) * t - np.sin(pitch) * up_true           # all look along +t and down
        x = np.cross(z, up_true); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.vstack([x, y, z])
        c = -0.4 * t + side * b + 0.45 * up_true                  # 45 cm above the table
        extr.append(np.column_stack([R, -R @ c]))
    Rup = upright(np.array(extr), table, np.random.default_rng(0))
    assert np.allclose(Rup @ up_true, [0, 1, 0], atol=1e-3)


def test_find_objects_groups_labelled_points():
    from echotwin.perception.objects import find_objects
    rng = np.random.default_rng(14)
    g = box_room(4.0, 3.0, 0.05)
    n = 4000
    chair = np.column_stack([rng.uniform(1.0, 1.6, n), rng.uniform(1.0, 1.5, n), rng.uniform(0.05, 0.8, n)])
    lone = np.array([[3.0, 2.0, 0.3]] * 3)                        # 3 points: below min_pts, ignored
    floor = np.column_stack([rng.uniform(0, 4, n), rng.uniform(0, 3, n), np.zeros(n)])   # z=0: not counted
    pts = np.vstack([chair, lone, floor])
    cls = np.r_[np.full(n, 56), np.full(3, 57), np.full(n, 56)].astype(np.int16)
    frame = rng.integers(0, 3, len(pts))
    objs, img = find_objects(pts, cls, frame, g, {56: "chair", 57: "couch"})
    assert len(objs) == 1 and objs[0]["class"] == "chair"
    o = objs[0]
    assert abs(o["x"] - 1.3) < 0.06 and abs(o["y"] - 1.25) < 0.06
    assert abs(o["size_x"] - 0.6) < 0.11 and abs(o["size_y"] - 0.5) < 0.11
    assert 0.7 < o["height"] <= 0.8 and o["photos"] == 3


def test_find_objects_merges_fragment_into_larger_object():
    """A cushion labelled 'chair' next to the couch, inside the couch outline, becomes part of the couch."""
    from echotwin.perception.objects import find_objects
    rng = np.random.default_rng(15)
    g = box_room(4.0, 3.0, 0.05)
    n = 4000
    couch = np.column_stack([rng.uniform(1.0, 2.5, n), rng.uniform(2.0, 2.8, n), rng.uniform(0.05, 0.8, n)])
    couch = couch[~((couch[:, 0] > 1.4) & (couch[:, 0] < 1.7) & (couch[:, 1] > 2.2) & (couch[:, 1] < 2.6))]  # cushion hole
    cushion = np.column_stack([rng.uniform(1.4, 1.7, 800), rng.uniform(2.2, 2.6, 800), rng.uniform(0.3, 0.5, 800)])
    chair = np.column_stack([rng.uniform(0.2, 0.7, n), rng.uniform(0.3, 0.8, n), rng.uniform(0.05, 0.8, n)])
    pts = np.vstack([couch, cushion, chair])
    cls = np.r_[np.full(len(couch), 57), np.full(800, 56), np.full(n, 56)].astype(np.int16)
    objs, _ = find_objects(pts, cls, rng.integers(0, 3, len(pts)), g, {56: "chair", 57: "couch"})
    got = sorted((o["class"], o["merged"]) for o in objs)
    assert got == [("chair", 0), ("couch", 1)]
    couch_obj = [o for o in objs if o["class"] == "couch"][0]
    assert abs(couch_obj["size_x"] - 1.5) < 0.11


def test_find_objects_keeps_things_on_a_table_apart_from_the_table():
    """A cup and a book standing on a desk are objects of their own, not fragments of the desk."""
    from echotwin.perception.objects import find_objects
    rng = np.random.default_rng(16)
    g = box_room(3.0, 3.0, 0.02)
    n = 60000                                                 # dense enough that every desk cell has points
    desk = np.column_stack([rng.uniform(0.8, 2.0, n), rng.uniform(0.8, 1.6, n), rng.uniform(0.05, 0.12, n)])
    under_cup = (desk[:, 0] > 0.98) & (desk[:, 0] < 1.14) & (desk[:, 1] > 0.98) & (desk[:, 1] < 1.14)
    under_book = (desk[:, 0] > 1.48) & (desk[:, 0] < 1.77) & (desk[:, 1] > 1.18) & (desk[:, 1] < 1.42)
    desk = desk[~(under_cup | under_book)]                    # the desk is hidden under the things on it
    n = len(desk)
    cup = np.column_stack([rng.uniform(1.0, 1.12, 800), rng.uniform(1.0, 1.12, 800), rng.uniform(0.12, 0.22, 800)])
    book = np.column_stack([rng.uniform(1.5, 1.75, 800), rng.uniform(1.2, 1.4, 800), rng.uniform(0.12, 0.15, 800)])
    pts = np.vstack([desk, cup, book])
    cls = np.r_[np.full(n, 60), np.full(800, 41), np.full(800, 73)].astype(np.int16)
    objs, _ = find_objects(pts, cls, rng.integers(0, 3, len(pts)), g,
                           {60: "desk", 41: "cup", 73: "book"}, min_area=0.001)
    assert sorted((o["class"], o["merged"]) for o in objs) == [("book", 0), ("cup", 0), ("desk", 0)]


def test_scale_gives_metres(tmp_path):
    """A photo reconstruction has arbitrary units: the room here is stored at 0.37 x its real size."""
    rng = np.random.default_rng(8)
    f = tmp_path / "recon.ply"
    floor = np.column_stack([rng.uniform(0, 4, 20000), rng.uniform(0, 3, 20000), np.zeros(20000)])
    trimesh.PointCloud(floor * 0.37).export(f)
    pts = load_points(str(f), up="z", scale=1 / 0.37, voxel=0)
    assert abs(np.ptp(pts[:, 0]) - 4.0) < 0.05 and abs(np.ptp(pts[:, 1]) - 3.0) < 0.05
