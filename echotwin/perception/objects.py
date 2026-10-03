"""Occupancy map + object layer from a labelled cloud (video_to_ply.py -> label_objects.py -> this).

    python -m echotwin.perception.objects data/lounge_vggt.ply -o out/lounge_objects --up y --scale 3.333 --floor-offset 0.29

Writes <out>.png/.yaml (the map, same as mesh_to_grid), <out>_scene.json (the scene file the robot twin
reads, see echotwin/scene/schema.md), <out>_objects.json ("map": grid origin and
size; "objects": one entry per object with class, centre and size in metres in the map frame,
height, how many photos saw it) and
<out>_objects.png (map with the objects drawn on it).

The object layer is for people and planning, not for the particle filter: sonars cannot tell a
chair from a wall, so localisation still uses the plain occupancy map.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage


from echotwin.scene import catalog
from echotwin.scene import schema as scene_schema
from echotwin.perception.mapping import (UP, add_scale_args, align_walls, level_floor, load_points,
                                   points_to_grid, resolve_scale)


TABLES = {"dining table"}      # catalog.normalize maps table, desk, coffee table... to this


def find_objects(pts, cls, frame, gmap, names, min_pts=5, min_views=2, min_area=0.02, max_z=1.5, pixels=None):
    """Group labelled points into objects on the map grid.

    A cell belongs to the class with most points in it (points between 2 cm and `max_z` above the
    floor), if that class has >= min_pts points there from >= min_views photos. Connected cells of
    one class form an object.

    `pixels` = (row, col, (H, W)): where each point sits in its photo (model-size pixels). Then every object
    also gets "views": up to MAX_VIEWS photos that see it best, with its box in that photo as fractions
    (x0, y0, x1, y1) of the image, so a reviewer can look at the right place.
    """
    ok = (cls >= 0) & (pts[:, 2] > 0.02) & (pts[:, 2] < max_z)
    r, c = gmap.to_cell(pts[ok, 0], pts[ok, 1])
    inside = gmap.inside(r, c)
    r, c, k, f, z = r[inside], c[inside], cls[ok][inside], frame[ok][inside], pts[ok, 2][inside]
    if pixels is not None:
        pr, pc = pixels[0][ok][inside], pixels[1][ok][inside]
    classes = np.unique(k)
    votes = np.zeros((len(classes),) + gmap.occ.shape, np.int32)
    views = np.zeros_like(votes)
    for j, kk in enumerate(classes):
        m = k == kk
        np.add.at(votes[j], (r[m], c[m]), 1)
        cell_frame = np.unique(np.column_stack([r[m], c[m], f[m]]), axis=0)
        np.add.at(views[j], (cell_frame[:, 0], cell_frame[:, 1]), 1)
    best = votes.argmax(0)
    good = (votes.max(0) >= min_pts) & (np.take_along_axis(views, best[None], 0)[0] >= min_views)
    parts = []                                                 # (cells, [class ids]) per connected blob
    for j, kk in enumerate(classes):
        mask = ndimage.binary_closing(good & (best == j), iterations=1)
        lab, n = ndimage.label(mask)
        parts += [(lab == i, [kk]) for i in range(1, n + 1) if (lab == i).sum() * gmap.res ** 2 >= min_area]
    tables = {int(k) for k in classes if catalog.normalize(names[int(k)]) in TABLES}
    parts = merge_parts(parts, keep_apart=tables)
    objects, label_img = [], np.full(gmap.occ.shape, -1, int)
    for cells, kks in parts:
        rr, cc = np.nonzero(cells)
        in_obj = np.isin(k, kks) & cells[r, c]
        objects.append({
            "class": names[int(kks[0])],
            "x": round(float(gmap.origin[0] + (cc.mean() + 0.5) * gmap.res), 3),
            "y": round(float(gmap.origin[1] + (rr.mean() + 0.5) * gmap.res), 3),
            "size_x": round(float((np.ptp(cc) + 1) * gmap.res), 2),
            "size_y": round(float((np.ptp(rr) + 1) * gmap.res), 2),
            "height": round(float(np.quantile(z[in_obj], 0.95)), 2) if in_obj.any() else None,
            "base_z": round(float(np.quantile(z[in_obj], 0.05)), 2) if in_obj.any() else 0.0,
            "photos": int(len(np.unique(f[in_obj]))),
            "points": int(in_obj.sum()),
            "merged": len(kks) - 1,
        })
        if pixels is not None:
            objects[-1]["views"] = object_views(f[in_obj], pr[in_obj], pc[in_obj], pixels[2])
        label_img[cells] = len(objects) - 1
    return objects, label_img


MAX_VIEWS = 3


def object_views(f, row, col, hw, max_views=MAX_VIEWS):
    """Photos that see an object best: [{"frame", "box": [x0, y0, x1, y1] as fractions of the photo, "points"}]."""
    H, W = hw
    out = []
    frames, counts = np.unique(f, return_counts=True)
    for i in frames[np.argsort(-counts)][:max_views]:
        m = f == i
        x0, x1 = np.quantile(col[m], [0.03, 0.97])
        y0, y1 = np.quantile(row[m], [0.03, 0.97])
        out.append({"frame": int(i), "points": int(m.sum()),
                    "box": [round(float(x0 / W), 3), round(float(y0 / H), 3),
                            round(float((x1 + 1) / W), 3), round(float((y1 + 1) / H), 3)]})
    return out


def _bbox(cells):
    rr, cc = np.nonzero(cells)
    return rr.min(), rr.max() + 1, cc.min(), cc.max() + 1


def merge_parts(parts, overlap=0.5, max_ratio=0.4, reach=3, keep_apart=()):
    """Merge fragments into the object they belong to (a couch cushion labelled 'chair').

    A blob is a fragment of a larger blob when it has at most `max_ratio` of its area, >= `overlap`
    of its bounding box lies inside the larger one's box, and it touches it (cells within `reach`
    cells). With several candidates it joins the one it touches most. The larger blob keeps its
    class. Two real neighbours of similar size stay apart. Blobs whose class id is in `keep_apart` (tables:
    things standing on a table are not part of it) never absorb anything.
    """
    parts = sorted(parts, key=lambda p: -p[0].sum())
    i = len(parts) - 1
    while i > 0:                                               # smallest first
        cells = parts[i][0]
        c0, c1, d0, d1 = _bbox(cells)
        best, best_touch = None, 0.0
        for j in range(i):
            big = parts[j][0]
            if cells.sum() > max_ratio * big.sum() or any(k in keep_apart for k in parts[j][1]):
                continue
            a0, a1, b0, b1 = _bbox(big)
            inter = max(0, min(a1, c1) - max(a0, c0)) * max(0, min(b1, d1) - max(b0, d0))
            if inter < overlap * (c1 - c0) * (d1 - d0):
                continue
            touch = (ndimage.binary_dilation(big, iterations=reach) & cells).sum() / cells.sum()
            if touch > best_touch:
                best, best_touch = j, touch
        if best is not None:
            parts[best] = (parts[best][0] | cells, parts[best][1] + parts[i][1])
            del parts[i]
        i -= 1
    return parts


def draw(gmap, objects, label_img, path, clean=False):
    """Map with coloured object footprints. clean=True: no axes or title, class names only (slides)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rgb = np.full(gmap.occ.shape + (3,), 0.80)
    rgb[gmap.known & ~gmap.occ] = 1.0
    rgb[gmap.occ] = 0.12
    palette = plt.get_cmap("tab10")
    kinds = sorted({o["class"] for o in objects})
    for i, o in enumerate(objects):
        col = np.array(palette(kinds.index(o["class"]) % 10)[:3])
        m = label_img == i
        rgb[m] = 0.45 * rgb[m] + 0.55 * col
    if clean:
        fig = plt.figure(figsize=(8, 8 * gmap.H / gmap.W), dpi=160)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
    else:
        fig, ax = plt.subplots(figsize=(8, 8 * gmap.H / gmap.W + 0.6), dpi=110)
    ax.imshow(rgb, origin="lower", extent=gmap.extent, interpolation="nearest")
    for o in objects:
        text = o["class"] if clean else f"{o['class']}\n{o['size_x']:.1f}x{o['size_y']:.1f} m"
        ax.text(o["x"], o["y"], text, ha="center", va="center", fontsize=11 if clean else 8,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    if clean:
        fig.savefig(path)
        plt.close(fig)
        return
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Occupancy map (black) + objects from photo segmentation (colour)", fontsize=10)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cloud", help=".ply from video_to_ply.py, with .labels.npz from label_objects.py")
    ap.add_argument("-o", "--out", required=True, help="output stem")
    ap.add_argument("--up", choices=["auto", *UP], default="y")
    ap.add_argument("--res", type=float, default=0.03)
    ap.add_argument("--band", type=float, nargs=2, default=(0.05, 0.35))
    ap.add_argument("--floor-offset", type=float, default=0.0, metavar="H")
    ap.add_argument("--min-views", type=int, default=2, help="photos that must agree on a cell")
    ap.add_argument("--min-area", type=float, default=0.02, metavar="M2",
                    help="smallest object footprint in m2 (a tabletop needs about 0.0015)")
    add_scale_args(ap)
    a = ap.parse_args(argv)
    lab = np.load(Path(a.cloud).with_suffix(".labels.npz"))
    names = {int(k): v for k, v in json.loads(str(lab["names"])).items()}
    pts = load_points(a.cloud, up=a.up, scale=resolve_scale(a), voxel=0)   # voxel=0 keeps point order
    if len(pts) != len(lab["cls"]):
        raise SystemExit(f"{len(pts)} points but {len(lab['cls'])} labels: rerun label_objects.py on this cloud")
    pts, *_ = level_floor(pts, np.random.default_rng(0))
    pts = pts - [0, 0, a.floor_offset]
    pts, *_ = align_walls(pts)
    gmap = points_to_grid(pts, res=a.res, band=tuple(a.band))
    gmap.save(a.out)
    pixels = None
    pix_path = Path(a.cloud).with_suffix(".pix.npz")
    if pix_path.exists():                                    # per-point photo pixels: lets the review find each object
        pix = np.load(pix_path)
        if len(pix["row"]) == len(pts):
            pixels = (pix["row"].astype(int), pix["col"].astype(int), tuple(int(v) for v in pix["model_hw"]))
    objects, label_img = find_objects(pts, lab["cls"], lab["frame"], gmap, names, min_views=a.min_views,
                                      min_area=a.min_area, pixels=pixels)
    meta = {"origin": [round(gmap.origin[0], 4), round(gmap.origin[1], 4)], "res": gmap.res,
            "width": gmap.W, "height": gmap.H}                   # lets run_demo check it is the same map
    Path(f"{a.out}_objects.json").write_text(json.dumps({"map": meta, "objects": objects}, indent=1))
    scene_doc = scene_schema.build_scene(objects, meta, name=Path(a.out).name)   # what the robot twin reads
    scene_schema.save(scene_doc, f"{a.out}_scene.json")
    draw(gmap, objects, label_img, f"{a.out}_objects.png")
    draw(gmap, objects, label_img, f"{a.out}_objects_clean.png", clean=True)
    for o in sorted(objects, key=lambda o: -o["points"]):
        print(f"{o['class']:>14}  at ({o['x']:+.2f}, {o['y']:+.2f}) m  {o['size_x']:.2f} x {o['size_y']:.2f} m"
              f"  h {o['height']} m  seen in {o['photos']} photos" + (f"  ({o['merged']} merged)" if o["merged"] else ""))
    print(f"{len(objects)} objects -> {a.out}_scene.json / {a.out}_objects.png")


if __name__ == "__main__":
    main()
