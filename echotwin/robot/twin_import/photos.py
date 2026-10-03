"""Default photo importer: photos -> twin (everyday mode).

Sharpest photo: table segmentation, tilt-based camera, photo skins; the AI names the objects afterwards.
"""
import asyncio
import time

import cv2

from ...perception import marker as MK
from ..features import everyday as EV
from ..features import imageutil as IMG
from ..scene import Layout
from .contract import TwinContext


async def import_photos(frames: list[bytes], pitches: list, ctx: TwinContext) -> None:
    if not frames:
        return
    ctx.say("Mapping your table.")
    await import_everyday(frames, pitches or [], ctx)


async def import_everyday(frames: list[bytes], pitches: list, ctx: TwinContext) -> None:
    scores = [IMG.sharpness(IMG.decode(f)) for f in frames]
    seen = await asyncio.to_thread(lambda: [i for i, f in enumerate(frames) if MK.detect(IMG.decode(f)) is not None])
    k = max(seen or range(len(frames)), key=lambda i: scores[i])         # a photo that shows the marker, if any
    pitch = pitches[k] if k < len(pitches) else None
    ctx.progress("everyday", {"pitch": pitch})
    res = await asyncio.to_thread(EV.analyse, frames[k], pitch, MK.size_m())
    ctx.say("I found the marker, so the sizes are measured." if res["calibration"]["source"] == "marker" else
            "No marker in the photos, so sizes are estimated and can be off. Print one: see the capture guide.")
    sid, out = ctx.new_dir()
    EV.apply_names(res, None)
    props, tex, ann = await asyncio.to_thread(EV.build, res, out)
    cv2.imwrite(str(out / "twin.jpg"), ann, [cv2.IMWRITE_JPEG_QUALITY, 85])
    lay = Layout()
    lay.texture, lay.props = tex, props
    lay.table_half = res.get("table_half", lay.table_half)
    n = len(props)
    ai = ctx.ask_ai_json is not None
    summary = {"id": sid, "mode": "everyday", "frames": len(frames), "views": 1, "fallback": False,
               "objects": {}, "unsure": [], "thumbs": [], "twin": f"/scans/{sid}/twin.jpg",
               "texture": f"/scans/{sid}/table.png", "seconds": 0, "pitch": res["pitch"], "calibration": res["calibration"],
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
