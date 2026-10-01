"""3D scans -> MuJoCo meshes.

Accepts what phone scanning apps export (Polycam, Scaniverse, Luma, Reality Composer): .glb/.gltf, .obj, .ply,
.stl. The mesh is merged, turned from Y-up (glTF convention) to Z-up, centred on its bounding box and written
as .obj (+ its colour texture as .png when there is one). Point clouds (no faces) are refused with a clear
message: export a mesh from the app instead.
"""
from pathlib import Path

import numpy as np

from .contract import ImportError_

MAX_FACES = 300_000


def convert(src: Path, out_dir: Path, stem: str) -> dict:
    """-> {"file": obj path, "texture": png path | None, "extent": [x, y, z] in file units, "faces": n}"""
    import trimesh
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        loaded = trimesh.load(str(src), force=None)
    except Exception as e:
        raise ImportError_(f"Could not read {src.name}: {e}")
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.dump() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ImportError_("That file has no surfaces (a point cloud?). Export it as a mesh.")
        mesh = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise ImportError_("That file has no surfaces (a point cloud?). Export it as a mesh.")
    if len(mesh.faces) == 0:
        raise ImportError_("That file has no surfaces (a point cloud?). Export it as a mesh.")
    if len(mesh.faces) > MAX_FACES:
        raise ImportError_(f"That scan has {len(mesh.faces):,} faces; please decimate it below {MAX_FACES:,} "
                           "in the scanning app (low/medium detail export).")
    if src.suffix.lower() in (".glb", ".gltf"):  # glTF is Y-up; the sim is Z-up
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    lo, hi = mesh.bounds
    mesh.apply_translation(-(lo + hi) / 2)
    texture = None
    try:
        img = getattr(getattr(mesh.visual, "material", None), "baseColorTexture", None) or \
            getattr(getattr(mesh.visual, "material", None), "image", None)
        if img is not None and getattr(mesh.visual, "uv", None) is not None:
            texture = out_dir / f"{stem}.png"
            img.convert("RGB").save(texture)
    except Exception:
        texture = None
    table = find_tabletop(mesh)
    obj = out_dir / f"{stem}.obj"
    obj.write_text(trimesh.exchange.obj.export_obj(mesh, include_texture=False, include_normals=True,
                                                   write_texture=False, mtl_name=None), encoding="utf-8")
    return {"file": str(obj.resolve()), "texture": str(texture.resolve()) if texture else None,
            "extent": [float(v) for v in (hi - lo)], "faces": int(len(mesh.faces)), "table": table}


def find_tabletop(mesh) -> dict | None:
    """The biggest upward-facing flat area at table height (0.4-1.2 m above the lowest point, file units = m).
    -> {"z": height relative to the mesh centre, "xy": its centre} or None."""
    n = mesh.face_normals
    up = n[:, 2] > 0.9
    if not up.any():
        return None
    c = mesh.triangles_center[up]
    area = mesh.area_faces[up]
    bottom = mesh.bounds[0][2]
    h = c[:, 2] - bottom
    keep = (h > 0.4) & (h < 1.2)
    if not keep.any():
        return None
    c, area, h = c[keep], area[keep], h[keep]
    bins = np.round(h / 0.02).astype(int)
    best = max(set(bins.tolist()), key=lambda b: area[bins == b].sum())
    sel = bins == best
    if area[sel].sum() < 0.1:  # less than 0.1 m2: not a table
        return None
    w = area[sel] / area[sel].sum()
    return {"z": float((c[sel, 2] * w).sum()), "xy": [float((c[sel, 0] * w).sum()), float((c[sel, 1] * w).sum())],
            "area": float(area[sel].sum())}


def extent(path: str) -> np.ndarray:
    import trimesh
    m = trimesh.load(path, force="mesh")
    lo, hi = m.bounds
    return np.maximum(hi - lo, 1e-6)


def fit_scale(path: str, half_size) -> tuple:
    """Uniform scale that makes the mesh's footprint match the object's size (keeps its proportions)."""
    ext = extent(path)
    target = 2 * np.array(half_size, float)
    s = float(min(target[0] / ext[0], target[1] / ext[1]))
    return (s, s, s)
