"""HTTPS server: phone controller, dashboard, MJPEG video, websockets, scan upload.

Run:  .venv\\Scripts\\python -m puppeteer.server
"""
import asyncio
import base64
import datetime
import ipaddress
import json
import os
import pathlib
import re
import socket
import time

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import config
from . import docpage
from . import twin_import as TI
from .ai.brain import Brain
from .features import video_everyday as VE
from .features.imageutil import decode
from .router import Intent, route
from .scene import Layout
from .sim import Sim

app = FastAPI()
clients: dict[WebSocket, str] = {}
loop: asyncio.AbstractEventLoop | None = None
brain = Brain()
latest_photo: dict = {"jpeg": None, "t": 0}
scan_buf: dict = {"frames": [], "mode": None, "busy": False}


# ---------------- broadcast ----------------
async def _broadcast(msg: dict):
    to = msg.pop("to", "all")
    data = json.dumps(msg)
    for ws, role in list(clients.items()):
        if to != "all" and role != to:
            continue
        try:
            await ws.send_text(data)
        except Exception:
            clients.pop(ws, None)


def emit(msg: dict):
    if loop is None:
        return
    loop.call_soon_threadsafe(lambda: asyncio.ensure_future(_broadcast(dict(msg))))


sim = Sim(emit)


@app.on_event("startup")
async def _startup():
    global loop
    if loop is None:
        loop = asyncio.get_running_loop()
        sim.start()


# ---------------- pages ----------------
@app.get("/")
async def dashboard():
    return FileResponse(config.STATIC / "dashboard.html")


@app.get("/phone")
async def phone():
    return FileResponse(config.STATIC / "phone.html")


GUIDES = {"teaching": ("TEACHING.md", "Teaching the robot"), "capture": ("CAPTURE_GUIDE.md", "Taking the photos")}


@app.get("/docs/{name}")
async def guide_page(name: str):
    """A guide from docs/ as a page (only the ones listed above)."""
    if name not in GUIDES:
        return JSONResponse({"error": "no such guide"}, status_code=404)
    file, title = GUIDES[name]
    return Response(docpage.page((config.ROOT / "docs" / file).read_text(encoding="utf-8"), title),
                    media_type="text/html; charset=utf-8")


@app.get("/api/info")
async def info():
    ip = lan_ip()
    return {"phone_url": f"https://{ip}:{config.PORT}/phone",
            "phone_urls": [f"https://{a}:{config.PORT}/phone" for a in all_ips()], "name": config.ROBOT_NAME,
            "brain": brain.enabled, "voice": sim.voice.enabled, "camera": config.FEATURE_CAMERA,
            "shake": config.FEATURE_SHAKE, "stats": sim.dataset.stats(), "scan": sim.scan}


@app.get("/video.mjpg")
async def video():
    async def gen():
        last = -1
        while True:
            if sim.frame_id != last and sim.jpeg:
                last = sim.frame_id
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + sim.jpeg + b"\r\n"
            await asyncio.sleep(0.04)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/frame.jpg")
async def frame():
    return Response(sim.jpeg or b"", media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/photo.jpg")
async def photo():
    if not latest_photo["jpeg"]:
        return Response(status_code=404)
    return Response(latest_photo["jpeg"], media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/voice/{name}")
async def voice_file(name: str):
    p = config.VOICE_CACHE / name
    if not p.exists() or p.parent != config.VOICE_CACHE:
        return Response(status_code=404)
    return FileResponse(p, media_type="audio/mpeg")


app.mount("/scans", StaticFiles(directory=config.SCANS), name="scans")
app.mount("/img", StaticFiles(directory=config.STATIC / "img"), name="img")     # pictures used by the pages


check_store: dict[str, bytes] = {}


@app.post("/api/check")
async def check_photos(files: list[UploadFile] = File(...)):
    """Validate photos one by one before building the twin."""
    import cv2
    from .features.check import check_photo
    sid = time.strftime("%H%M%S") + f"{int(time.time() * 1000) % 1000:03d}"
    out = config.SCANS / f"check_{sid}"
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for n, f in enumerate(files):
        data = await f.read()
        res, preview = await asyncio.to_thread(check_photo, data)
        pid = f"{sid}_{n}"
        check_store[pid] = data
        if preview is not None:
            cv2.imwrite(str(out / f"{n}.jpg"), preview, [cv2.IMWRITE_JPEG_QUALITY, 80])
            res["preview"] = f"/scans/check_{sid}/{n}.jpg"
        res.update({"id": pid, "name": f.filename})
        results.append(res)
    while len(check_store) > 300:
        check_store.pop(next(iter(check_store)))
    return JSONResponse({"results": results})


@app.post("/api/check/build")
async def build_from_checked(body: dict):
    frames = [check_store[i] for i in body.get("ids", []) if i in check_store]
    if not frames:
        return JSONResponse({"ok": False, "error": "No photos selected."})
    asyncio.ensure_future(process_scan(frames))
    return JSONResponse({"ok": True, "frames": len(frames)})


VIDEO_EXT = (".mp4", ".mov", ".m4v", ".avi", ".webm")


@app.post("/api/scan")
async def upload_scan(files: list[UploadFile] = File(...)):
    """Photos from the laptop, or one video: its frames are used like the photos of a sweep scan."""
    blobs = [(f.filename or "", f.content_type or "", await f.read()) for f in files]
    if len(blobs) == 1 and (blobs[0][1].startswith("video/") or blobs[0][0].lower().endswith(VIDEO_EXT)):
        from .features.imageutil import frames_from_video
        n = int(os.environ.get("SCAN_FRAMES") or 12)
        frames = await asyncio.to_thread(frames_from_video, blobs[0][2], n)
        if len(frames) < 3:
            sim.submit(sim.say, "I couldn't read enough frames from that video.")
            return JSONResponse({"ok": False, "error": "unreadable video"})
    else:
        frames = [b[2] for b in blobs]
    asyncio.ensure_future(process_scan(frames))
    return JSONResponse({"ok": True, "frames": len(frames)})


def _examples() -> list[dict]:
    f = config.ROOT / "examples" / "examples.json"
    items = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
    for e in items:
        photos = sorted(p for p in (config.ROOT / "examples" / e["folder"]).glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        e["photos"] = len(photos)
    return [e for e in items if e["photos"]]


@app.get("/api/examples")
async def examples():
    return JSONResponse([{k: e[k] for k in ("id", "title", "what", "photos")} for e in _examples()])


@app.post("/api/examples/{name}")
async def run_example(name: str):
    """Scan the photos of an example, exactly as if they came from the phone."""
    ex = next((e for e in _examples() if e["id"] == name), None)
    if ex is None:
        return JSONResponse({"ok": False, "error": "no such example"}, status_code=404)
    folder = config.ROOT / "examples" / ex["folder"]
    frames = [p.read_bytes() for p in sorted(folder.glob("*")) if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    asyncio.ensure_future(process_scan(frames))
    return JSONResponse({"ok": True, "frames": len(frames)})


@app.post("/api/demo")
async def upload_demo(files: list[UploadFile] = File(...)):
    """A video (or a sequence of photos) of your hand doing the task."""
    blobs = [(f.filename or "", f.content_type or "", await f.read()) for f in files]
    asyncio.ensure_future(process_demo_upload(blobs))
    return JSONResponse({"ok": True, "files": len(blobs)})


async def process_demo_upload(blobs):
    if len(blobs) == 1 and not blobs[0][1].startswith("image/"):
        emit({"t": "demo", "stage": "decoding"})
        frames = await asyncio.to_thread(VE.frames_from_video_ends, blobs[0][2])
    else:
        blobs.sort(key=lambda b: b[0])
        frames = [img for img in (decode(b[2]) for b in blobs) if img is not None]
    await process_video_demo(frames)


async def process_everyday_video(frames):
    """Video of a hand moving an everyday object: find what moved where, then replay it in the twin."""
    import cv2
    emit({"t": "demo", "stage": "processing", "frames": len(frames)})
    sim.submit(sim.say, "Watching your video.")
    res = await asyncio.to_thread(VE.analyse, frames)
    if "error" in res:
        sim.submit(sim.say, res["error"])
        return
    if not sim.world.layout.props:  # no everyday twin yet: build it from the start of the video
        ok, buf = cv2.imencode(".jpg", res["start_frame"], [cv2.IMWRITE_JPEG_QUALITY, 90])
        await process_everyday([buf.tobytes()], [None])
        for _ in range(40):
            if sim.world.layout.props:
                break
            await asyncio.sleep(0.25)
    props = sim.world.layout.props
    plan = VE.map_to_twin(res, props)
    if plan is None:
        sim.submit(sim.say, "I saw something move, but I can't tell which object it is in my map.")
        return
    from .features import move_things as MT
    name = props[plan["prop"]]["name"]
    where = (f"next to the {props[plan['goal'][1]]['name']}" if plan["goal"][0] == "near"
             else {(-1, 0): "to the left", (1, 0): "to the right", (0, -1): "towards you", (0, 1): "away from you"}
             .get(tuple(plan["goal"][1]), "a bit"))
    emit({"t": "log", "who": "system", "text": f"Video: the {name} moved {res['distance'] * 100:.0f} cm, "
                                               f"{where} ({res['lined_up']} objects stayed put)"})
    sim.submit(sim.play_prop_video, plan, f"I watched you move the {name} {where}. Let me try it in my twin.")


async def process_video_demo(frames):
    if len(frames) < 3:
        sim.submit(sim.say, "That video is too short. Show me the whole move.")
        return
    await process_everyday_video(frames)


# ---------------- speech / text ----------------
async def handle_text(text: str, source: str):
    text = text.strip()
    if not text:
        return
    emit({"t": "log", "who": "user", "text": text, "source": source})
    it = route(text)
    props = sim.world.layout.props
    if props and re.search(r"\b(practi[cs]e|train (yourself|in (the )?sim)|generate (data|demos))\b", text.lower()):
        from .features import move_things as MT
        ment = MT._mentions(text, props)
        sim.submit(sim.say, "Practising in my twin. This takes a few seconds.")
        if ment:
            sim.submit(sim.practice, None, 3, True, ment[0][1])
        else:
            sim.submit(sim.practice)
        return
    if props and it.kind not in ("safety", "control", "robot_q") and it.name not in ("which", "see"):
        from .features import move_things as MT
        plan = MT.parse(text, props)
        if plan is None and sim.pending_move is not None:  # answer to "where should I put it?"
            again = MT.parse(f"move object {sim.pending_move['prop'] + 1} {text}", props)
            plan = again if again and again["goal"] else None
        if plan:
            sim.submit(sim.handle_prop_task, plan, text)
            return
        if it.kind == "other":
            # understand odd phrasings with the AI, using the real object names
            names = [p["name"] for p in props]
            if brain.enabled:
                emit({"t": "thinking", "on": True})
                parsed = await brain.parse_everyday(text, names)
                emit({"t": "thinking", "on": False})
                plan = _everyday_plan(parsed, len(names))
                if plan:
                    emit({"t": "log", "who": "system", "text": f"AI understood: move the {names[plan['prop']]}"})
                    sim.submit(sim.handle_prop_task, plan, text)
                    return
                if parsed and parsed.get("clarify"):
                    sim.submit(sim.say, str(parsed["clarify"])[:120])
                    return
                reply = await brain.chat(text, scene=", ".join(names))
                sim.submit(sim.say, reply or "I can move the things on your table. Tell me which one and where.")
                return
            sim.submit(sim.say, f"I can move the {names[0]}" + (f" or the {names[1]}" if len(names) > 1 else "")
                       + ". Tell me which one and where.")
            return
    if not props and it.kind == "other":
        reply = await brain.chat(text) if brain.enabled else None
        sim.submit(sim.say, reply or "Scan your table first, so I know what is on it.")
        return
    sim.submit(sim.handle_intent, it)
    if it.kind == "scene_q" and it.name == "see" and brain.enabled and latest_photo["jpeg"]:
        asyncio.ensure_future(describe_photo(text))


def _everyday_plan(parsed: dict | None, n: int) -> dict | None:
    """AI answer {object, relation, reference} -> move plan, or None."""
    from .features import move_things as MT
    if not parsed or "object" not in parsed:
        return None
    try:
        i = int(parsed["object"]) - 1
    except (TypeError, ValueError):
        return None
    if not 0 <= i < n:
        return None
    rel = str(parsed.get("relation", "")).lower()
    ref = parsed.get("reference")
    if ref is not None and rel in ("next to", "left of", "right of", "in front of", "behind", "near", "beside"):
        try:
            j = int(ref) - 1
        except (TypeError, ValueError):
            return None
        if 0 <= j < n and j != i:
            return {"prop": i, "goal": ("near", j, "next to" if rel in ("near", "beside") else rel)}
    if rel in ("middle", "center", "centre"):
        return {"prop": i, "goal": ("center",)}
    if rel in MT.DIRS:
        return {"prop": i, "goal": ("dir", MT.DIRS[rel], 0.2)}
    return {"prop": i, "goal": None}


async def describe_photo(question: str):
    emit({"t": "thinking", "on": True})
    desc = await brain.describe(latest_photo["jpeg"], question)
    emit({"t": "thinking", "on": False})
    if desc:
        sim.submit(sim.say, "From the photo: " + desc)


# ---------------- twin import (all the logic lives in echotwin/robot/twin_import) ----------------
def _new_dir(prefix: str = "") -> tuple[str, pathlib.Path]:
    sid = prefix + time.strftime("%H%M%S") + f"{int(time.time() * 1000) % 1000:03d}"
    out = config.SCANS / sid
    out.mkdir(parents=True, exist_ok=True)
    return sid, out


def twin_ctx(skip: asyncio.Event | None = None) -> "TI.TwinContext":
    async def ask(jpeg: bytes, prompt: str):
        emit({"t": "thinking", "on": True})
        try:
            return await brain.image_json(jpeg, prompt)
        finally:
            emit({"t": "thinking", "on": False})

    return TI.TwinContext(
        apply=lambda lay, summary: sim.submit(sim.apply_scan, lay, summary),
        rename=lambda props, line: sim.submit(sim.name_props, props, line),
        say=lambda text: sim.submit(sim.say, text),
        progress=lambda stage, data: emit({"t": "scan", "stage": stage, **data}),
        new_dir=_new_dir,
        ask_ai_json=ask if brain.enabled else None,
        skip=skip)


async def process_everyday(frames: list[bytes], pitches: list):
    """Build a twin from photos with no sheet (used by the video path when there is no twin yet)."""
    await TI.photo_importer()(frames, pitches, twin_ctx())


def _apply_doc(doc: dict, folder: pathlib.Path, sid: str, how: str):
    lay = TI.doc_to_layout(doc, folder)
    names = [p["name"] for p in lay.props]
    from .features.everyday import listing
    seen = []
    if names:
        seen.append(listing(names))
    extra = f" and {len(lay.scene)} 3D scan{'s' if len(lay.scene) != 1 else ''}" if lay.scene else ""
    summary = {"id": sid, "mode": "file", "props": names, "objects": {}, "unsure": [], "thumbs": [],
               "views": 0, "frames": 0, "seconds": 0, "scene": len(lay.scene), "name": doc.get("name", "twin"),
               "texture": None, "twin": None,
               "greeting": f"{how} I see {' and '.join(seen) or 'an empty table'}{extra}. What should I move?"}
    emit({"t": "scan", "stage": "done", "summary": summary})
    sim.submit(sim.apply_scan, lay, summary)
    return summary


@app.get("/api/twin/export")
async def twin_export():
    sid, folder = _new_dir("export_")
    data = await asyncio.to_thread(TI.export_zip, sim.world.layout, sim.world, folder,
                                   sim.world.layout.meta.get("name", "twin"))
    return Response(data, media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="twin.zip"'})


@app.get("/api/twin/layout")
async def twin_layout():
    sid, folder = _new_dir("edit_")
    doc = await asyncio.to_thread(TI.layout_to_doc, sim.world.layout, sim.world, folder)
    doc["_dir"] = sid
    return JSONResponse(doc)


@app.post("/api/twin/layout")
async def twin_layout_apply(doc: dict):
    folder = config.SCANS / str(doc.get("_dir", ""))
    if not doc.get("_dir") or not folder.is_dir() or folder.parent != config.SCANS:
        return JSONResponse({"ok": False, "error": "Unknown twin folder; reload the editor."})
    try:
        _apply_doc(doc, folder, doc["_dir"], "Updated the twin.")
    except TI.ImportError_ as e:
        return JSONResponse({"ok": False, "error": str(e)})
    return JSONResponse({"ok": True})


@app.post("/api/twin/import")
async def twin_import(file: UploadFile = File(...)):
    sid, folder = _new_dir("import_")
    try:
        doc = TI.read_upload(await file.read(), file.filename or "", folder)
        _apply_doc(doc, folder, sid, "Loaded your twin.")
    except TI.ImportError_ as e:
        sim.submit(sim.say, str(e))
        return JSONResponse({"ok": False, "error": str(e)})
    return JSONResponse({"ok": True})


@app.post("/api/twin/mesh")
async def twin_mesh(file: UploadFile = File(...), target: str = Form("scene")):
    """Add a 3D scan: target 'scene' (the room/table, scenery) or an object number (its real shape)."""
    sid, folder = _new_dir("mesh_")
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
                k = float(doc.get("sim_scale", 2.0))
                pos = [0, 0, round(-0.75 / k * 100 + info["extent"][2] * 100 / 2, 2)]
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


async def process_scan(frames: list[bytes], pitches: list | None = None):
    if scan_buf["busy"]:
        return
    scan_buf["busy"] = True
    scan_buf["skip"] = skip = asyncio.Event()      # "skip, use quick mode" from the phone or dashboard
    try:
        if frames:
            latest_photo["jpeg"] = frames[len(frames) // 2]
            latest_photo["t"] = time.time()
        await TI.photo_importer()(frames, pitches or [], twin_ctx(skip))
    except Exception as e:
        import traceback
        traceback.print_exc()
        emit({"t": "scan", "stage": "error", "error": str(e)})
        sim.submit(sim.say, "The scan failed. Please try again.")
    finally:
        scan_buf["busy"] = False
        scan_buf["skip"] = None


def _pitch(m: dict):
    """Camera pitch below horizontal from the phone's tilt (beta: 90 = upright, 0 = flat)."""
    b = m.get("beta")
    return None if b is None else float(max(15, min(89, 90 - abs(float(b)))))


def _b64(data_url: str) -> bytes:
    return base64.b64decode(data_url.split(",", 1)[-1])


# ---------------- websocket ----------------
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    role = ws.query_params.get("role", "dash")
    clients[ws] = role
    await ws.send_text(json.dumps({"t": "hello", "role": role, "name": config.ROBOT_NAME,
                                   "stats": sim.dataset.stats(), "scan": sim.scan,
                                   "flags": {"camera": config.FEATURE_CAMERA, "shake": config.FEATURE_SHAKE,
                                             "brain": brain.enabled, "voice": sim.voice.enabled}}))
    if role == "phone":
        emit({"t": "log", "who": "system", "text": "Phone connected", "to": "dash"})
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            await on_message(msg, role)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print("[ws]", e)
    finally:
        clients.pop(ws, None)
        if role == "phone":
            sim.submit(sim.set_human, 0, 0, 0)


async def on_message(m: dict, role: str):
    t = m.get("t")
    if t == "ctl":
        sim.set_human(m.get("vx", 0), m.get("vy", 0), m.get("vz", 0))
    elif t == "grip":
        sim.submit(sim.set_grip, bool(m.get("on")))
    elif t == "say":
        await handle_text(m.get("text", ""), role)
    elif t == "stop":
        reason = m.get("reason", "button")
        if reason == "shake" and not config.FEATURE_SHAKE:
            return
        print(f"[stop] from {role}: {reason}", flush=True)
        sim.submit(sim.halt, reason)
    elif t == "btn":
        name = m.get("name")
        if name == "stop":
            sim.submit(sim.halt, "button")
        elif name == "rescan_reset":
            sim.submit(sim.reset_scene)
        else:
            sim.submit(sim.handle_intent, Intent("control", name))
    elif t == "scan_skip":
        if scan_buf.get("skip") is not None:
            scan_buf["skip"].set()
    elif t == "scan_begin":
        scan_buf["frames"], scan_buf["mode"], scan_buf["pitches"] = [], m.get("mode", "sweep"), []
        emit({"t": "scan", "stage": "capturing", "mode": scan_buf["mode"]})
    elif t == "frame":
        if not config.FEATURE_CAMERA:
            return
        scan_buf["frames"].append(_b64(m["img"]))
        scan_buf.setdefault("pitches", []).append(_pitch(m))
        emit({"t": "scan", "stage": "receiving", "n": len(scan_buf["frames"])})
    elif t == "scan_end":
        frames, scan_buf["frames"] = scan_buf["frames"], []
        asyncio.ensure_future(process_scan(frames, scan_buf.pop("pitches", [])))
    elif t == "demo_begin":
        scan_buf["frames"], scan_buf["mode"] = [], "demo"
        emit({"t": "demo", "stage": "recording"})
        sim.submit(sim.say, "Recording. Show me.")
    elif t == "demo_end":
        jpegs, scan_buf["frames"] = scan_buf["frames"], []
        frames = [img for img in (decode(j) for j in jpegs) if img is not None]
        asyncio.ensure_future(process_video_demo(frames))
    elif t == "snap":  # one photo: quick one-view scan
        if not config.FEATURE_CAMERA:
            return
        asyncio.ensure_future(process_scan([_b64(m["img"])], [_pitch(m)]))


# ---------------- HTTPS ----------------
def lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def all_ips() -> list[str]:
    """Every IPv4 address of this laptop the phone might use: Wi-Fi, USB tethering, hotspot."""
    ips = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith(("127.", "169.254.")) and ip not in ips:
                ips.append(ip)
    except OSError:
        pass
    main_ip = lan_ip()
    if main_ip not in ips and not main_ip.startswith("127."):
        ips.insert(0, main_ip)
    return ips or ["127.0.0.1"]


def ensure_cert(ip: str):
    """Self-signed cert for localhost + this LAN IP (phones need HTTPS for camera and motion)."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    cert_p, key_p, ip_p = config.CERTS / "cert.pem", config.CERTS / "key.pem", config.CERTS / "ip.txt"
    ips = sorted(set(all_ips() + [ip]))
    if cert_p.exists() and ip_p.exists() and ip_p.read_text() == ",".join(ips):
        return str(cert_p), str(key_p)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "phone-puppeteer")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=30))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost"),
                                                        x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
                                                       + [x509.IPAddress(ipaddress.ip_address(a)) for a in ips]), False)
            .sign(key, hashes.SHA256()))
    cert_p.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_p.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                        serialization.NoEncryption()))
    ip_p.write_text(",".join(ips))
    return str(cert_p), str(key_p)


def main():
    import sys
    sys.stdout.reconfigure(line_buffering=True)  # logs show up immediately
    ip = lan_ip()
    cert, key = ensure_cert(ip)
    print(f"\n  Dashboard: http://localhost:{config.HTTP_PORT}/   (or https://localhost:{config.PORT}/)")
    for a in all_ips():
        print(f"  Phone:     https://{a}:{config.PORT}/phone   (accept the certificate warning once)")
    print(flush=True)
    servers = [uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=config.PORT, ssl_certfile=cert,
                                             ssl_keyfile=key, log_level="warning")),
               uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=config.HTTP_PORT, log_level="warning"))]

    async def serve():
        await asyncio.gather(*(s.serve() for s in servers))

    asyncio.run(serve())


if __name__ == "__main__":
    main()
