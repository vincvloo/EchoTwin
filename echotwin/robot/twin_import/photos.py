"""Default photo importer: photos -> twin.

  A4 sheet visible in any photo -> block mode (multi-view scan: sheet pose, colour blobs, fused layout)
  no sheet                      -> everyday mode (sharpest photo: table segmentation, tilt-based camera,
                                   photo skins; the AI names the objects afterwards)
"""
import asyncio
import time

import cv2

from ..features import everyday as EV
from ..features import vision as VIS
from ..features.scan import NoSheet, run_scan
from ..scene import Layout
from .contract import TwinContext


def has_sheet(jpeg: bytes) -> bool:
    img = VIS.decode(jpeg)
    if img is None:
        return False
    c = VIS.find_sheet(img)
    return c is not None and bool(VIS.pose_candidates(c, img.shape))


async def import_photos(frames: list[bytes], pitches: list, ctx: TwinContext) -> None:
    if not frames:
        return
    sheets = await asyncio.gather(*(asyncio.to_thread(has_sheet, f) for f in frames[:12]))
    ctx.say("Mapping your table.")
    if not any(sheets):
        await import_everyday(frames, pitches or [], ctx)
        return
    ctx.progress("processing", {"frames": len(frames)})
    try:
        layout, summary = await asyncio.to_thread(run_scan, frames, ctx.progress)
    except NoSheet as e:
        ctx.progress("error", {"error": str(e)})
        ctx.say(str(e))
        return
    layout.meta.setdefault("sim_scale", 2.0)
    summary["mode"] = "blocks"
    ctx.progress("done", {"summary": summary})
    ctx.apply(layout, summary)


async def import_everyday(frames: list[bytes], pitches: list, ctx: TwinContext) -> None:
    scores = [VIS.sharpness(VIS.decode(f)) for f in frames]
    k = max(range(len(frames)), key=lambda i: scores[i])
    pitch = pitches[k] if k < len(pitches) else None
    ctx.progress("everyday", {"pitch": pitch})
    res = await asyncio.to_thread(EV.analyse, frames[k], pitch)
    sid, out = ctx.new_dir()
    EV.apply_names(res, None)
    props, tex, ann = await asyncio.to_thread(EV.build, res, out)
    cv2.imwrite(str(out / "twin.jpg"), ann, [cv2.IMWRITE_JPEG_QUALITY, 85])
    lay = Layout()
    lay.texture, lay.props, lay.show_zones = tex, props, False
    lay.meta["sim_scale"] = res.get("sim_scale", 2.0)
    for o in lay.objects.values():
        o["present"] = False
    n = len(props)
    ai = ctx.ask_ai_json is not None
    summary = {"id": sid, "mode": "everyday", "frames": len(frames), "with_sheet": 0, "views": 1, "fallback": False,
               "objects": {}, "unsure": [], "thumbs": [], "twin": f"/scans/{sid}/twin.jpg",
               "texture": f"/scans/{sid}/table.png", "seconds": 0, "pitch": res["pitch"],
               "props": [p["name"] for p in props],
               "greeting": (f"I've mapped your table. I see {n} thing{'s' if n != 1 else ''} on it. Let me look closer."
                            if n and ai else EV.greeting(res["items"]))}
    ctx.progress("done", {"summary": summary})
    ctx.apply(lay, summary)
    if not (n and ai):
        return
    ok, buf = cv2.imencode(".jpg", EV._prep(res["marks"], 768), [cv2.IMWRITE_JPEG_QUALITY, 80])
    named_json = await ctx.ask_ai_json(buf.tobytes(), EV.NAMING_PROMPT)
    if not named_json:  # the free queue is flaky: one retry with a smaller image
        ok, buf = cv2.imencode(".jpg", EV._prep(res["marks"], 512), [cv2.IMWRITE_JPEG_QUALITY, 75])
        named_json = await ctx.ask_ai_json(buf.tobytes(), EV.NAMING_PROMPT)
    if not named_json:
        ctx.say(f"I see {n} things, but I can't tell what they are right now.")
        return
    named = EV.apply_names({"items": [dict(i) for i in res["items"]]}, named_json)["items"]
    by_box = {tuple(i["box"]): i for i in named}
    res["items"] = [it for it in res["items"] if tuple(it["box"]) in by_box]  # drop what the AI called "skip"
    for it in res["items"]:
        it["name"] = by_box[tuple(it["box"])]["name"]
        it["shape"] = by_box[tuple(it["box"])]["shape"]
    props, _, ann = await asyncio.to_thread(EV.build, res, out)
    cv2.imwrite(str(out / "twin.jpg"), ann, [cv2.IMWRITE_JPEG_QUALITY, 85])
    summary["props"] = [it["name"] for it in res["items"]]
    summary["twin"] = f"/scans/{sid}/twin.jpg?named{int(time.time())}"
    ctx.progress("done", {"summary": summary})
    ctx.rename(props, EV.greeting(named))
