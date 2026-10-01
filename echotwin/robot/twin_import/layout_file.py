"""Twin layout files: save the current twin, edit it, load it back (or write one by hand).

A twin file is a .zip holding twin.json plus the images/meshes it refers to (or just a twin.json).
All positions and sizes are REAL centimetres on the table, origin at the table centre, x to the right,
y away from the viewer. The sim is drawn "sim_scale" times bigger (sim metres per real metre).

{
  "format": "phone-puppeteer-twin", "version": 1, "name": "DARE table",
  "sim_scale": 2.0,
  "table_texture": "table.png",                         # optional top-down photo of the table
  "objects": [                                           # everyday objects the robot can move
    {"name": "chocolate box", "shape": "flat",            # flat | box | cylinder | round
     "size_cm": [10, 6, 1.2], "pos_cm": [-8, -5], "yaw_deg": 20, "color": "#e8e4dc",
     "skin": "skins/prop_0.png",                          # optional photo wrapped on the shape
     "mesh": "meshes/choc.obj", "mesh_texture": "meshes/choc.png"}   # optional 3D-scanned shape
  ],
  "scene": [{"mesh": "scene/room.obj", "texture": "scene/room.png",   # optional 3D scan of the room
             "pos_cm": [0, 0, -37.5], "euler_deg": [0, 0, 0], "scale": 1.0}],  # scale: real metres per mesh unit
  "camera": {"pos_cm": [...], "xyaxes": [...], "fovy": 60}             # optional viewpoint
}
"""
import io
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

from ..scene import Layout
from .contract import ImportError_

FORMAT = "phone-puppeteer-twin"
SHAPES = ("flat", "box", "cylinder", "round")


def _hex(rgb) -> str:
    return "#" + "".join(f"{int(max(0, min(1, c)) * 255):02x}" for c in rgb)


def _rgb(hexcol: str | None):
    if not hexcol:
        return (0.7, 0.7, 0.7)
    h = hexcol.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _yaw(quat) -> float:
    w, x, y, z = quat
    return float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))


def _copy(src: str | None, file_dir: Path, sub: str) -> str | None:
    """Copy a referenced file next to twin.json; return its relative name."""
    if not src:
        return None
    p = Path(src)
    if not p.exists():
        return None
    rel = f"{sub}/{p.name}"
    dst = file_dir / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if p.resolve() != dst.resolve():
        shutil.copyfile(p, dst)
    return rel


# ---------------- twin -> document ----------------
def layout_to_doc(layout: Layout, world, file_dir: Path, name: str = "twin") -> dict:
    """Describe the twin as it is NOW (current object positions), copying its files into file_dir."""
    k = float(layout.meta.get("sim_scale", 2.0))
    cm = lambda v: round(float(v) / k * 100, 2)
    doc = {"format": FORMAT, "version": 1, "name": layout.meta.get("name", name), "sim_scale": k,
           "table_texture": _copy(layout.texture, file_dir, "."), "objects": [], "scene": []}
    if doc["table_texture"]:
        doc["table_texture"] = doc["table_texture"].lstrip("./")
    for i, pr in enumerate(layout.props):
        has_prop = (world is not None and hasattr(world, 'obj_qadr') and f"prop_{i}" in world.obj_qadr)
        pos = world.obj_pos(f"prop_{i}") if has_prop else (*pr.get("pos", (0, 0)), 0)
        q = world.data.qpos[world.obj_qadr[f"prop_{i}"] + 3:world.obj_qadr[f"prop_{i}"] + 7] if has_prop else None
        o = {"name": pr["name"], "shape": pr.get("shape", "box"),
             "size_cm": [cm(2 * v) for v in pr["size"]], "pos_cm": [cm(pos[0]), cm(pos[1])],
             "yaw_deg": round(_yaw(q), 1) if q is not None else pr.get("yaw", 0.0), "color": _hex(pr["rgb"])}
        for key, sub in (("skin", "skins"), ("mesh", "meshes"), ("mesh_texture", "meshes")):
            rel = _copy(pr.get(key), file_dir, sub)
            if rel:
                o[key] = rel
        doc["objects"].append(o)
    for m in layout.scene:
        e = {"mesh": _copy(m["file"], file_dir, "scene"), "pos_cm": [cm(v) for v in m.get("pos", (0, 0, 0))],
             "euler_deg": list(m.get("euler", (0, 0, 0))), "scale": round(m.get("scale", k) / k, 6)}
        tex = _copy(m.get("texture"), file_dir, "scene")
        if tex:
            e["texture"] = tex
        doc["scene"].append(e)
    if layout.view:
        doc["camera"] = {"pos_cm": [cm(v) for v in layout.view["pos"]], "xyaxes": layout.view["xyaxes"],
                         "fovy": layout.view["fovy"]}
    return doc


# ---------------- document -> twin ----------------
def doc_to_layout(doc: dict, file_dir: Path) -> Layout:
    if doc.get("format") not in (FORMAT, None):
        raise ImportError_(f"Not a twin file (format={doc.get('format')!r}).")
    k = float(doc.get("sim_scale", 2.0))
    m = lambda v: float(v) / 100 * k  # real cm -> sim metres

    def f(rel):
        if not rel:
            return None
        p = (file_dir / rel).resolve()
        if not p.exists():
            raise ImportError_(f"The twin file refers to {rel}, which is missing.")
        return str(p)

    lay = Layout()
    lay.meta = {"sim_scale": k, "name": doc.get("name", "twin")}
    lay.texture = f(doc.get("table_texture"))
    if doc.get("blocks") or doc.get("zones"):
        print("[twin] this file has coloured blocks or zones; they are no longer supported and are ignored.")
    for i, o in enumerate(doc.get("objects", [])):
        shape = o.get("shape", "box")
        if shape not in SHAPES:
            raise ImportError_(f"Object {i + 1}: shape must be one of {', '.join(SHAPES)}.")
        w, d, h = (max(0.2, float(v)) for v in o.get("size_cm", [5, 5, 5]))
        pr = {"name": str(o.get("name") or f"object {i + 1}")[:40], "shape": shape,
              "pos": (m(o["pos_cm"][0]), m(o["pos_cm"][1])), "yaw": float(o.get("yaw_deg", 0.0)),
              "size": (m(w) / 2, m(d) / 2, m(h) / 2), "rgb": _rgb(o.get("color")), "skin": f(o.get("skin"))}
        if o.get("mesh"):
            from .meshes import fit_scale
            from .meshes import extent
            pr["mesh"] = f(o["mesh"])
            pr["mesh_scale"] = fit_scale(pr["mesh"], pr["size"])
            pr["size"] = tuple(float(v) for v in extent(pr["mesh"]) * pr["mesh_scale"][0] / 2)  # true shape
            pr["mesh_texture"] = f(o.get("mesh_texture"))
        lay.props.append(pr)
    for e in doc.get("scene", []):
        lay.scene.append({"file": f(e["mesh"]), "texture": f(e.get("texture")),
                          "pos": tuple(m(v) for v in e.get("pos_cm", (0, 0, 0))),
                          "euler": tuple(e.get("euler_deg", (0, 0, 0))), "scale": float(e.get("scale", 1.0)) * k})
    if doc.get("camera"):
        c = doc["camera"]
        lay.view = {"pos": [m(v) for v in c["pos_cm"]], "xyaxes": c["xyaxes"], "fovy": c.get("fovy", 60)}
    return lay


# ---------------- zip files ----------------
def export_zip(layout: Layout, world, work_dir: Path, name: str = "twin") -> bytes:
    work_dir.mkdir(parents=True, exist_ok=True)
    doc = layout_to_doc(layout, world, work_dir, name)
    (work_dir / "twin.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in work_dir.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(work_dir).as_posix())
    return buf.getvalue()


def read_upload(data: bytes, filename: str, work_dir: Path) -> dict:
    """A .zip (twin.json + files) or a bare twin.json -> the document; files land in work_dir."""
    work_dir.mkdir(parents=True, exist_ok=True)
    if filename.lower().endswith(".zip") or data[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for info in z.infolist():
                target = (work_dir / info.filename).resolve()
                if not str(target).startswith(str(work_dir.resolve())):
                    raise ImportError_("The zip contains an unsafe path.")
                if not info.is_dir():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(z.read(info))
        found = list(work_dir.rglob("twin.json"))
        if not found:
            raise ImportError_("No twin.json inside the zip.")
        if found[0].parent != work_dir:  # zipped inside a folder: work from there
            for p in found[0].parent.iterdir():
                shutil.move(str(p), str(work_dir / p.name))
        return json.loads((work_dir / "twin.json").read_text(encoding="utf-8"))
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ImportError_(f"That is not a twin file: {e}")
