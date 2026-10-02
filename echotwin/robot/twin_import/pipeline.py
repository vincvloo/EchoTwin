"""Phone photos -> twin through the 3D pipeline (VGGT + YOLO + review), with a way out.

The 3D route takes a few minutes. While it runs the dashboard and the phone show progress and a "skip" button;
skipping, a missing perception environment, fewer than 3 photos, a failed step, or a scene with nothing small
enough to move all end in the same place: the quick one-photo importer (`photos.py`) on the same photos.

The steps themselves are listed in `echotwin/perception/pipeline.py` (standard library only). They run as
subprocesses in the perception environment (PERCEPTION_PY in .env), so this robot environment needs no torch.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from echotwin.perception import pipeline as PL
from echotwin.scene import schema, to_twin

from ..features.everyday import listing
from . import photos
from .contract import TwinContext
from .layout_file import doc_to_layout

ROOT = Path(__file__).resolve().parents[3]
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


async def run_step(argv: list[str], cwd: Path, skip: asyncio.Event | None, tick,
                   env: dict | None = None) -> tuple[str, str]:
    """Run one command. -> ("done" | "failed" | "skipped", its output). `tick(seconds)` is called every 2 s."""
    proc = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", creationflags=NO_WINDOW,
                            env={**os.environ, "PYTHONIOENCODING": "utf-8", **(env or {})})
    comm = asyncio.ensure_future(asyncio.to_thread(proc.communicate))
    waiting = {comm}
    skipper = asyncio.ensure_future(skip.wait()) if skip is not None else None
    if skipper:
        waiting.add(skipper)
    t0 = time.time()
    try:
        while True:
            done, _ = await asyncio.wait(waiting, timeout=2.0, return_when=asyncio.FIRST_COMPLETED)
            if comm in done:
                return ("done" if proc.returncode == 0 else "failed"), comm.result()[0] or ""
            if skipper in done:
                proc.kill()
                await comm
                return "skipped", ""
            tick(time.time() - t0)
    finally:
        if skipper:
            skipper.cancel()


async def import_photos_auto(frames: list[bytes], pitches: list, ctx: TwinContext, cfg: dict | None = None,
                             run=run_step) -> None:
    """The default photo importer: 3D when possible, quick otherwise."""
    cfg = cfg or PL.settings()
    why_not = PL.usable(cfg, len(frames))
    if why_not:
        if cfg["mode"] == "3d":
            ctx.say(f"I can't build a 3D model: {why_not}. Using quick mode.")
        return await photos.import_photos(frames, pitches, ctx)

    sid, folder = ctx.new_dir()
    inputs = folder / "input"
    inputs.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        (inputs / f"{i:03d}.jpg").write_bytes(f)
    steps = PL.plan(inputs, folder, gpu_py=cfg["gpu_py"], maps_py=cfg["maps_py"], frames=cfg["frames"],
                    cam_height=cfg["cam_height"], review=bool(_has_key()), table=True,
                    vggt_path=cfg.get("vggt_path", ""))
    names = [s.name for s in steps] + ["Building your twin"]
    total = sum(s.typical_s for s in steps) + 2

    def progress(i: int, elapsed: float, state: str = "running"):
        done = sum(s.typical_s for s in steps[:i]) + (min(elapsed, 0.95 * steps[i].typical_s) if i < len(steps) else 0)
        ctx.progress("pipeline", {"steps": names, "i": i, "elapsed": round(elapsed), "state": state,
                                  "pct": int(100 * min(done / total, 1.0))})

    async def quick(reason: str):
        ctx.progress("pipeline", {"steps": names, "i": 0, "elapsed": 0, "state": "skipped", "pct": 0})
        ctx.say(reason)
        await photos.import_photos(frames, pitches, ctx)

    ctx.say("I'm building a 3D model of your table. This takes a few minutes. You can skip and use quick mode.")
    scale = None
    for i, step in enumerate(steps):
        progress(i, 0)
        state, out = await run(PL.fill(step.argv, scale), ROOT, ctx.skip, lambda el, i=i: progress(i, el), step.env)
        if state == "skipped":
            return await quick("Okay, quick mode.")
        if state == "failed":
            tail = [ln for ln in out.splitlines() if ln.strip()][-3:]
            print(f"[pipeline] {step.name} failed:", " | ".join(tail), file=sys.stderr)
            if step.optional:
                continue
            return await quick(f"The 3D model failed at '{step.name}'. Using quick mode.")
        if i == 0:
            scale = PL.find_scale(out)

    progress(len(steps), 0)
    try:
        scene = schema.load(PL.scene_path(folder))
        doc = to_twin.scene_to_twin(scene, name="Your table")
    except to_twin.NoTable:
        return await quick(f"I found {len(scene['objects'])} things, but nothing small enough to move. "
                           "Using quick mode.")
    except (OSError, ValueError) as e:
        print("[pipeline] no scene:", e, file=sys.stderr)
        return await quick("The 3D model found no objects. Using quick mode.")
    (folder / "twin.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    lay = doc_to_layout(doc, folder)
    shot = next(iter(sorted((folder / "objects_scene_review").glob("*.jpg"))), None) or inputs / "000.jpg"
    shutil.copyfile(shot, folder / "twin.jpg")
    movable = [p["name"] for p in lay.props]
    extra = f", with {len(lay.obstacles)} fixed thing{'s' if len(lay.obstacles) != 1 else ''} around it" \
        if lay.obstacles else ""
    summary = {"id": sid, "mode": "scan3d", "frames": len(frames), "props": movable, "objects": {}, "unsure": [],
               "thumbs": [], "views": 0, "seconds": 0, "twin": f"/scans/{sid}/twin.jpg", "texture": None,
               "obstacles": [o["name"] for o in lay.obstacles], "review": scene.get("review"),
               "greeting": f"I built a 3D model of your table. I can move {listing(movable)}{extra}. "
                           "What should I move?"}
    ctx.progress("pipeline", {"steps": names, "i": len(steps), "elapsed": 0, "state": "done", "pct": 100})
    ctx.progress("done", {"summary": summary})
    ctx.apply(lay, summary)


def _has_key() -> bool:
    from echotwin.scene import review
    return bool(review.config()["key"])
