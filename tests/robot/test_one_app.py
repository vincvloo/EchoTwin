"""The one app: example photos, video upload, pictures and guides served by the robot server."""
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from echotwin.robot import scan_api, server
from echotwin.robot.features import imageutil


@pytest.fixture
def client():
    return TestClient(server.app)


@pytest.fixture
def scans(monkeypatch):
    """Replace the scan with a recorder, and wait for the background task the endpoints start."""
    seen = []

    async def fake(frames, pitches=None):
        seen.append(frames)
    monkeypatch.setattr(scan_api, "process_scan", fake)

    def wait():
        for _ in range(100):
            if seen:
                break
            time.sleep(0.02)
        return seen
    return wait


def _video(path, n=40):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for i in range(n):
        w.write(np.full((120, 160, 3), (i * 6) % 255, np.uint8))
    w.release()
    return path.read_bytes()


def test_frames_from_video_are_evenly_spread_jpegs(tmp_path):
    data = _video(tmp_path / "v.mp4")
    frames = imageutil.frames_from_video(data, n=8)
    assert len(frames) == 8
    levels = [int(cv2.imdecode(np.frombuffer(f, np.uint8), cv2.IMREAD_COLOR).mean()) for f in frames]
    assert levels[0] < levels[3] < levels[7] or levels[0] != levels[7]            # they come from different times
    assert imageutil.frames_from_video(b"not a video", n=8) == []


def test_examples_are_listed_and_scanned_like_phone_photos(client, scans):
    listing = client.get("/api/examples").json()
    assert [e["id"] for e in listing] == ["table-photos"] and listing[0]["photos"] >= 10
    r = client.post("/api/examples/table-photos")
    assert r.json()["ok"] and r.json()["frames"] == listing[0]["photos"]
    frames = scans()[0]
    assert len(frames) == listing[0]["photos"] and all(f[:2] == b"\xff\xd8" for f in frames)    # JPEG bytes
    assert client.post("/api/examples/nope").status_code == 404


def test_a_video_upload_becomes_frames(client, scans, tmp_path):
    data = _video(tmp_path / "v.mp4")
    r = client.post("/api/scan", files=[("files", ("table.mp4", data, "video/mp4"))])
    assert r.json()["ok"] and 3 <= r.json()["frames"] <= 12
    assert len(scans()[0]) == r.json()["frames"]


def test_an_unreadable_video_is_refused_not_scanned(client, scans):
    r = client.post("/api/scan", files=[("files", ("broken.mp4", b"garbage", "video/mp4"))])
    assert r.json() == {"ok": False, "error": "unreadable video"}
    time.sleep(0.2)
    assert not scans()                                                           # nothing was scanned


def test_photos_still_go_straight_through(client, scans):
    jpg = cv2.imencode(".jpg", np.zeros((20, 30, 3), np.uint8))[1].tobytes()
    r = client.post("/api/scan", files=[("files", (f"{i}.jpg", jpg, "image/jpeg")) for i in range(4)])
    assert r.json() == {"ok": True, "frames": 4} and len(scans()[0]) == 4


def test_pictures_and_guides_are_served(client):
    for name in ("map_lounge.png", "map_splat.png", "map_home.png"):
        r = client.get(f"/img/{name}")
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert "Taking the photos" in client.get("/docs/capture").text
    assert "Teaching the robot" in client.get("/docs/teaching").text


def test_the_dashboard_has_the_new_controls(client):
    page = client.get("/").text
    for needle in ("id=\"examples\"", "id=\"videoFile\"", "id=\"capinfo\"", "/docs/capture", "/docs/teaching"):
        assert needle in page
    assert "sonar" not in page.lower() and "8765" not in page
