"""The web app: the dashboard and phone pages, the video, the guides, and the routers for scans, twins, demos and the socket.

Run:  python -m echotwin.robot.server
"""
import asyncio

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import config, demo_api, docpage, hub, scan_api, twin_api, ws
from .hub import brain, sim
from .net import all_ips, ensure_cert, lan_ip

app = FastAPI()


@app.middleware("http")
async def revalidate_pages(request, call_next):
    """Pages, styles and scripts change with every update: the browser must ask the server (a 304 is cheap), not reuse a copy."""
    response = await call_next(request)
    if not request.url.path.startswith(("/video.mjpg", "/voice/", "/scans/", "/img/")):
        response.headers.setdefault("Cache-Control", "no-cache")
    return response


for module in (scan_api, twin_api, demo_api, ws):
    app.include_router(module.router)


@app.on_event("startup")
async def _startup():
    hub.start()


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


# ---------------- video, voice, files ----------------
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


@app.get("/voice/{name}")
async def voice_file(name: str):
    p = config.VOICE_CACHE / name
    if not p.exists() or p.parent != config.VOICE_CACHE:
        return Response(status_code=404)
    return FileResponse(p, media_type="audio/mpeg")


app.mount("/scans", StaticFiles(directory=config.SCANS), name="scans")
app.mount("/img", StaticFiles(directory=config.STATIC / "img"), name="img")     # pictures used by the pages
app.mount("/ui", StaticFiles(directory=config.STATIC / "ui"), name="ui")        # shared style and script of both pages


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
