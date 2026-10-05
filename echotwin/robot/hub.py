"""What the web modules share: the connected pages, the broadcast, the robot (`sim`), the AI (`brain`)."""
import asyncio
import json
import pathlib
import time

from fastapi import WebSocket

from . import config
from .ai.brain import Brain
from .sim import Sim

clients: dict[WebSocket, str] = {}      # page -> its role ("dash" or "phone")
loop: asyncio.AbstractEventLoop | None = None
brain = Brain()
latest_photo: dict = {"jpeg": None}     # the middle photo of the last scan, for "what do you see?"


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
    """Send a message to every page (or only to msg["to"]); safe to call from the sim thread."""
    if loop is None:
        return
    loop.call_soon_threadsafe(lambda: asyncio.ensure_future(_broadcast(dict(msg))))


sim = Sim(emit)


def start():
    """Called once the event loop is running: the sim thread starts here."""
    global loop
    if loop is None:
        loop = asyncio.get_running_loop()
        sim.start()


def new_scan_dir(prefix: str = "") -> tuple[str, pathlib.Path]:
    sid = prefix + time.strftime("%H%M%S") + f"{int(time.time() * 1000) % 1000:03d}"
    out = config.SCANS / sid
    out.mkdir(parents=True, exist_ok=True)
    return sid, out
