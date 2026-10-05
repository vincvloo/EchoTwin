"""Teaching by video: a hand moves an object, the robot works out what moved where and replays it in the twin."""
import asyncio

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from . import twin_import as TI
from .features import video_everyday as VE
from .features.imageutil import decode
from .hub import emit, sim
from .scan_api import read_blobs, twin_ctx

router = APIRouter()


@router.post("/api/demo")
async def upload_demo(files: list[UploadFile] = File(...)):
    """A video (or a sequence of photos) of your hand doing the task."""
    blobs = await read_blobs(files)
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


async def process_video_demo(frames):
    """Find what moved where in the frames, then replay it in the twin (building the twin from the video if there is none)."""
    import cv2
    if len(frames) < 3:
        sim.submit(sim.say, "That video is too short. Show me the whole move.")
        return
    emit({"t": "demo", "stage": "processing", "frames": len(frames)})
    sim.submit(sim.say, "Watching your video.")
    res = await asyncio.to_thread(VE.analyse, frames)
    if "error" in res:
        sim.submit(sim.say, res["error"])
        return
    if not sim.world.layout.props:  # no everyday twin yet: build it from the start of the video
        ok, buf = cv2.imencode(".jpg", res["start_frame"], [cv2.IMWRITE_JPEG_QUALITY, 90])
        await TI.photo_importer()([buf.tobytes()], [None], twin_ctx())
        for _ in range(40):
            if sim.world.layout.props:
                break
            await asyncio.sleep(0.25)
    props = sim.world.layout.props
    plan = VE.map_to_twin(res, props)
    if plan is None:
        sim.submit(sim.say, "I saw something move, but I can't tell which object it is in my map.")
        return
    name = props[plan["prop"]]["name"]
    where = (f"next to the {props[plan['goal'][1]]['name']}" if plan["goal"][0] == "near"
             else {(-1, 0): "to the left", (1, 0): "to the right", (0, -1): "towards you", (0, 1): "away from you"}
             .get(tuple(plan["goal"][1]), "a bit"))
    emit({"t": "log", "who": "system", "text": f"Video: the {name} moved {res['distance'] * 100:.0f} cm, "
                                               f"{where} ({res['lined_up']} objects stayed put)"})
    sim.submit(sim.play_prop_video, plan, f"I watched you move the {name} {where}. Let me try it in my twin.")
