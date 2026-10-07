"""The twin editor's endpoints: save, load, edit the layout, add a 3D scan. The logic is in echotwin/robot/twin_import."""
import asyncio
import pathlib

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

from . import arm as A
from . import config
from . import twin_import as TI
from .features.everyday import listing
from .hub import emit, new_scan_dir, sim

router = APIRouter()


def _apply_doc(doc: dict, folder: pathlib.Path, sid: str, how: str):
    lay = TI.doc_to_layout(doc, folder)
    names = [p["name"] for p in lay.props]
    seen = [listing(names)] if names else []
    extra = f" and {len(lay.scene)} 3D scan{'s' if len(lay.scene) != 1 else ''}" if lay.scene else ""
    summary = {"id": sid, "mode": "file", "props": names, "objects": {}, "unsure": [], "thumbs": [],
               "views": 0, "frames": 0, "seconds": 0, "scene": len(lay.scene), "name": doc.get("name", "twin"),
               "texture": None, "twin": None,
               "greeting": f"{how} I see {' and '.join(seen) or 'an empty table'}{extra}. What should I move?"}
    emit({"t": "scan", "stage": "done", "summary": summary})
    sim.submit(sim.apply_scan, lay, summary)
    return summary


@router.get("/api/twin/export")
async def twin_export():
    sid, folder = new_scan_dir("export_")
    data = await asyncio.to_thread(TI.export_zip, sim.world.layout, sim.world, folder,
                                   sim.world.layout.meta.get("name", "twin"))
    return Response(data, media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="twin.zip"'})


@router.get("/api/twin/layout")
async def twin_layout():
    sid, folder = new_scan_dir("edit_")
    doc = await asyncio.to_thread(TI.layout_to_doc, sim.world.layout, sim.world, folder)
    doc["_dir"] = sid
    return JSONResponse(doc)


@router.post("/api/twin/layout")
async def twin_layout_apply(doc: dict):
    folder = config.SCANS / str(doc.get("_dir", ""))
    if not doc.get("_dir") or not folder.is_dir() or folder.parent != config.SCANS:
        return JSONResponse({"ok": False, "error": "Unknown twin folder; reload the editor."})
    try:
        _apply_doc(doc, folder, doc["_dir"], "Updated the twin.")
    except TI.ImportError_ as e:
        return JSONResponse({"ok": False, "error": str(e)})
    return JSONResponse({"ok": True})


@router.post("/api/twin/import")
async def twin_import(file: UploadFile = File(...)):
    sid, folder = new_scan_dir("import_")
    try:
        doc = TI.read_upload(await file.read(), file.filename or "", folder)
        _apply_doc(doc, folder, sid, "Loaded your twin.")
    except TI.ImportError_ as e:
        sim.submit(sim.say, str(e))
        return JSONResponse({"ok": False, "error": str(e)})
    return JSONResponse({"ok": True})


@router.post("/api/twin/mesh")
async def twin_mesh(file: UploadFile = File(...), target: str = Form("scene")):
    """Add a 3D scan: target 'scene' (the room/table, scenery) or an object number (its real shape)."""
    sid, folder = new_scan_dir("mesh_")
    src = folder / "upload" / (file.filename or "scan.glb")
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(await file.read())
    try:
        doc = await asyncio.to_thread(TI.layout_to_doc, sim.world.layout, sim.world, folder)
        sub_dir = "scene" if target == "scene" else "meshes"
        info = await asyncio.to_thread(TI.convert_mesh, src, folder / sub_dir, pathlib.Path(src).stem)
        rel = lambda path: pathlib.Path(path).relative_to(folder).as_posix() if path else None
        if target == "scene":
            t = info.get("table")
            if t:  # line up the scanned tabletop with the sim table (top at z = 0)
                pos = [round(-t["xy"][0] * 100, 2), round(-t["xy"][1] * 100, 2), round(-t["z"] * 100 - 0.2, 2)]
                how = f"Added your 3D scan and lined up its table ({t['area']:.1f} square metres of tabletop)."
            else:
                pos = [0, 0, round(-0.75 * 100 + info["extent"][2] * 100 / 2, 2)]
                how = "Added your 3D scan. I couldn't find its table, so adjust its position in the editor."
            doc["scene"].append({"mesh": rel(info["file"]), "texture": rel(info["texture"]),
                                 "pos_cm": pos, "euler_deg": [0, 0, 0], "scale": 1.0})
        else:
            i = int(target)
            if not 0 <= i < len(doc["objects"]):
                raise TI.ImportError_("Pick an object from the list.")
            doc["objects"][i]["mesh"] = rel(info["file"])
            if info["texture"]:
                doc["objects"][i]["mesh_texture"] = rel(info["texture"])
            how = f"The {doc['objects'][i]['name']} now has its scanned 3D shape."
        _apply_doc(doc, folder, sid, how)
    except (TI.ImportError_, ValueError) as e:
        sim.submit(sim.say, str(e))
        return JSONResponse({"ok": False, "error": str(e)})
    return JSONResponse({"ok": True, "faces": info["faces"]})


@router.post("/api/twin/scale")
async def twin_scale(body: dict):
    """Rescale the twin: {"factor": 0.5} or {"object": i, "width_cm": 8} (that object is really 8 cm wide)."""
    try:
        if body.get("width_cm") is not None:
            i = int(body.get("object", 0))
            now = 2 * sim.world.layout.props[i]["size"][0] * 100
            k = float(body["width_cm"]) / now
        else:
            k = float(body["factor"])
    except (KeyError, ValueError, TypeError, IndexError, ZeroDivisionError):
        return JSONResponse({"ok": False, "error": "Give an object and its real width in cm, or a factor."})
    lo, hi = sim.SCALE_RANGE
    total = float(sim.base_layout.meta.get("scale", 1.0)) * k
    if not k > 0 or not lo <= total <= hi:
        return JSONResponse({"ok": False, "error": f"That would make the table {total:.2f} times its real size; the limit is {lo} to {hi}."})
    sim.submit(sim.set_scale, k)
    return JSONResponse({"ok": True, "factor": round(k, 4), "scale": round(total, 4)})


@router.get("/api/arms")
async def arms():
    return JSONResponse({"arms": A.describe_arms(), "current": sim.world.arm.name if hasattr(sim.world, "arm") else None,
                         "can_switch": sim.world.name == "sim"})


@router.post("/api/arm")
async def arm_switch(body: dict):
    """{"name": "so_arm100"} switches the arm, {"size": 2.0} resizes it (1.0 = as described), both at once is fine."""
    name, size = body.get("name"), body.get("size")
    if name is not None:
        known = {a["name"]: a for a in A.describe_arms()}
        a = known.get(str(name))
        if a is None:
            return JSONResponse({"ok": False, "error": "No such arm."})
        if not a["ready"]:
            return JSONResponse({"ok": False, "error": f"Fetch it first: python -m echotwin.robot.arm --download {a['name']}"})
    if size is not None:
        lo, hi = sim.ARM_SIZE_RANGE
        try:
            size = float(size)
        except (TypeError, ValueError):
            return JSONResponse({"ok": False, "error": "The size is a number, 1 is the arm as described."})
        if not lo <= size <= hi:
            return JSONResponse({"ok": False, "error": f"The arm can be {lo} to {hi} times its size."})
    if name is None and size is None:
        return JSONResponse({"ok": False, "error": "Give an arm name, a size, or both."})
    sim.submit(sim.set_arm, None if name is None else str(name), size)
    return JSONResponse({"ok": True})


@router.get("/api/arm/needed")
async def arm_needed():
    """What size of arm each object needs, for the arm in use. Read from the sim thread's world: plain numbers, no rendering."""
    w = sim.world
    if not hasattr(w, "size_needed"):
        return JSONResponse({"ok": False, "error": "Only for the simulation."})
    try:
        s = float(w.arm.scale)
        objects = [w.size_needed(f"prop_{i}") for i in range(len(w.layout.props))]
    except (KeyError, IndexError, AttributeError):         # the sim thread is rebuilding the twin right now
        return JSONResponse({"ok": False, "error": "busy, try again"})
    can = [o for o in objects if o["needed"] is not None]
    all_k = max([o["needed"] for o in can], default=None)
    lo, hi = sim.ARM_SIZE_RANGE
    return JSONResponse({"ok": True, "arm": w.arm.name, "size": s, "range": [lo, hi],
                         "opening_cm": round(float(w.arm.max_opening or 0.08) / s * 100, 1),
                         "reach_cm": [round(w.workspace.r_min / s * 100, 1), round(w.workspace.r_max / s * 100, 1)],
                         "objects": objects,
                         "all": all_k if all_k is not None and all_k <= min(o["most"] for o in can) else None})
