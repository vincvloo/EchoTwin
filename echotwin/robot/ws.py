"""The WebSocket both pages use: what they send ("ctl", "say", "stop", "frame", ...) and what they get back."""
import asyncio
import base64
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from . import config
from .demo_api import process_video_demo
from .features.imageutil import decode
from .hub import brain, clients, emit, sim
from .router import Intent
from .chat import handle_text
from .scan_api import process_scan, scan_buf

router = APIRouter()


def _pitch(m: dict):
    """Camera pitch below horizontal from the phone's tilt (beta: 90 = upright, 0 = flat)."""
    b = m.get("beta")
    return None if b is None else float(max(15, min(89, 90 - abs(float(b)))))


def _b64(data_url: str) -> bytes:
    return base64.b64decode(data_url.split(",", 1)[-1])


@router.websocket("/ws")
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
            sim.submit(sim.set_drive, 0, 0, 0)


async def on_message(m: dict, role: str):
    t = m.get("t")
    if t == "ctl":
        sim.set_human(m.get("vx", 0), m.get("vy", 0), m.get("vz", 0))
    elif t == "drive":                          # a mobile base driven by hand: forward, sideways, turn (-1..1)
        sim.set_drive(m.get("forward", 0), m.get("sideways", 0), m.get("turn", 0))
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
        if scan_buf.skip is not None:
            scan_buf.skip.set()
    elif t == "scan_begin":
        scan_buf.begin(m.get("mode", "sweep"))
        emit({"t": "scan", "stage": "capturing", "mode": scan_buf.mode})
    elif t == "frame":
        if not config.FEATURE_CAMERA:
            return
        scan_buf.frames.append(_b64(m["img"]))
        scan_buf.pitches.append(_pitch(m))
        emit({"t": "scan", "stage": "receiving", "n": len(scan_buf.frames)})
    elif t == "scan_end":
        frames, pitches = scan_buf.take()
        asyncio.ensure_future(process_scan(frames, pitches))
    elif t == "demo_begin":
        scan_buf.begin("demo")
        emit({"t": "demo", "stage": "recording"})
        sim.submit(sim.say, "Recording. Show me.")
    elif t == "demo_end":
        jpegs, _ = scan_buf.take()
        frames = [img for img in (decode(j) for j in jpegs) if img is not None]
        asyncio.ensure_future(process_video_demo(frames))
    elif t == "snap":  # one photo: quick one-view scan
        if not config.FEATURE_CAMERA:
            return
        asyncio.ensure_future(process_scan([_b64(m["img"])], [_pitch(m)]))
