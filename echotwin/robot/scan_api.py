"""Scans: check photos, upload photos or a video, the examples, and the one pipeline that builds the twin."""
import asyncio
import os
import time
import traceback

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from . import config
from . import twin_import as TI
from .features.imageutil import frames_from_video
from .hub import brain, emit, latest_photo, new_scan_dir, sim

router = APIRouter()
check_store: dict[str, bytes] = {}
VIDEO_EXT = (".mp4", ".mov", ".m4v", ".avi", ".webm")
PHOTO_EXT = (".jpg", ".jpeg", ".png")


class ScanBuffer:
    """Frames the phone sends, and the state of the scan that is being built."""

    def __init__(self):
        self.frames: list[bytes] = []
        self.pitches: list = []
        self.mode: str | None = None
        self.busy = False
        self.skip: asyncio.Event | None = None      # "skip, use quick mode" from the phone or the dashboard

    def begin(self, mode: str):
        self.frames, self.pitches, self.mode = [], [], mode

    def take(self) -> tuple[list[bytes], list]:
        frames, pitches, self.frames, self.pitches = self.frames, self.pitches, [], []
        return frames, pitches


scan_buf = ScanBuffer()


async def read_blobs(files: list[UploadFile]) -> list[tuple[str, str, bytes]]:
    return [(f.filename or "", f.content_type or "", await f.read()) for f in files]


@router.post("/api/check")
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


@router.post("/api/check/build")
async def build_from_checked(body: dict):
    frames = [check_store[i] for i in body.get("ids", []) if i in check_store]
    if not frames:
        return JSONResponse({"ok": False, "error": "No photos selected."})
    asyncio.ensure_future(process_scan(frames))
    return JSONResponse({"ok": True, "frames": len(frames)})


@router.post("/api/scan")
async def upload_scan(files: list[UploadFile] = File(...)):
    """Photos from the laptop, or one video: its frames are used like the photos of a sweep scan."""
    blobs = await read_blobs(files)
    if len(blobs) == 1 and (blobs[0][1].startswith("video/") or blobs[0][0].lower().endswith(VIDEO_EXT)):
        n = int(os.environ.get("SCAN_FRAMES") or 12)
        frames = await asyncio.to_thread(frames_from_video, blobs[0][2], n)
        if len(frames) < 3:
            sim.submit(sim.say, "I couldn't read enough frames from that video.")
            return JSONResponse({"ok": False, "error": "unreadable video"})
    else:
        frames = [b[2] for b in blobs]
    asyncio.ensure_future(process_scan(frames))
    return JSONResponse({"ok": True, "frames": len(frames)})


def _example_photos(folder) -> list:
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in PHOTO_EXT)


def _examples() -> list[dict]:
    import json
    f = config.ROOT / "examples" / "examples.json"
    items = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
    for e in items:
        e["photos"] = len(_example_photos(config.ROOT / "examples" / e["folder"]))
    return [e for e in items if e["photos"]]


@router.get("/api/examples")
async def examples():
    return JSONResponse([{k: e[k] for k in ("id", "title", "what", "photos")} for e in _examples()])


@router.post("/api/examples/{name}")
async def run_example(name: str):
    """Scan the photos of an example, exactly as if they came from the phone."""
    ex = next((e for e in _examples() if e["id"] == name), None)
    if ex is None:
        return JSONResponse({"ok": False, "error": "no such example"}, status_code=404)
    frames = [p.read_bytes() for p in _example_photos(config.ROOT / "examples" / ex["folder"])]
    asyncio.ensure_future(process_scan(frames))
    return JSONResponse({"ok": True, "frames": len(frames)})


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
        new_dir=new_scan_dir,
        ask_ai_json=ask if brain.enabled else None,
        skip=skip)


async def process_scan(frames: list[bytes], pitches: list | None = None):
    if scan_buf.busy:
        return
    scan_buf.busy = True
    scan_buf.skip = skip = asyncio.Event()
    try:
        if frames:
            latest_photo["jpeg"] = frames[len(frames) // 2]
        await TI.photo_importer()(frames, pitches or [], twin_ctx(skip))
    except Exception as e:
        traceback.print_exc()
        emit({"t": "scan", "stage": "error", "error": str(e)})
        sim.submit(sim.say, "The scan failed. Please try again.")
    finally:
        scan_buf.busy = False
        scan_buf.skip = None
