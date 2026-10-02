"""Local web app for the demo: upload a scan, photos or a video, get the map, objects and robot run.

    python apps/perception_web/server.py            # then open http://127.0.0.1:8765

Standard library only. Runs the existing scripts as subprocesses, one job at a time (one GPU):
  scan (.ply/.glb/.obj/.gltf)   -> run_demo.py (map + quality report + simulated robot)
  photos (.jpg/.png) or a video -> echotwin.perception.reconstruct (VGGT) -> detect (YOLO11) ->
                                   object_map.py -> run_demo.py
Every job gets a folder runs/<id>/ with its inputs, log and results. Examples in demo_inputs/ can be
run without uploading (examples.json next to this file lists them and their settings).
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root
WEB = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
EXAMPLES_DIR = ROOT / "examples"
PITCH = ROOT / "docs" / "pitch"
PERCEPTION_PY = Path(os.environ.get("PERCEPTION_PY") or sys.executable)   # GPU env with torch, VGGT, ultralytics
SCAN_EXT = {".ply", ".glb", ".gltf", ".obj"}
IMG_EXT = {".jpg", ".jpeg", ".png"}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".webm"}

# Robot sim integration
ROBOT_URL = os.environ.get("ROBOT_URL", "http://localhost:8000")   # robot server HTTP port

JOBS: dict[str, dict] = {}
GPU = threading.Lock()                                        # VGGT and YOLO need the whole 4 GB


def send_to_robot(job: dict) -> dict:
    """Convert a finished job to a Puppeteer twin zip and POST it to the Robot server."""
    import io
    import urllib.error
    import urllib.request

    job_d = job["dir"]
    scene_json = job_d / "objects_scene.json"
    if not scene_json.exists():
        return {"ok": False, "error": "No object map in this run (need photos/video with objects=on)."}

    title = job.get("title", "Sonar Scan") + " (robot twin)"
    twin_zip = job_d / "robot_twin.zip"

    # build the twin from the scene file (numpy-free, runs in the web app's Python)
    cmd = [sys.executable, "-m", "echotwin.scene.to_twin", str(scene_json),
           "--out", str(twin_zip),
           "--name", title]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        return {"ok": False, "error": f"Could not build the twin: {r.stderr.strip() or r.stdout.strip()}"}

    # POST the zip to the Robot server
    data = twin_zip.read_bytes()
    boundary = b"----SonarBridge"
    body = (b"--" + boundary + b"\r\n"
            b'Content-Disposition: form-data; name="file"; filename="robot_twin.zip"\r\n'
            b"Content-Type: application/zip\r\n\r\n"
            + data + b"\r\n"
            b"--" + boundary + b"--\r\n")
    req = urllib.request.Request(
        f"{ROBOT_URL}/api/twin/import",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary.decode()}"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            if result.get("ok"):
                return {"ok": True, "objects": len(job.get("results", {}).get("objects", [])),
                        "robot_url": ROBOT_URL}
            return {"ok": False, "error": result.get("error", "Robot rejected the twin.")}
    except urllib.error.URLError as e:
        return {"ok": False, "error": f"Robot server not reachable at {ROBOT_URL}: {e.reason}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def kind_of(files: list[Path]) -> str:
    ext = {f.suffix.lower() for f in files}
    if len(files) == 1 and ext <= SCAN_EXT:
        return "scan"
    if ext <= IMG_EXT and len(files) >= 3:
        return "photos"
    if len(files) == 1 and ext <= VIDEO_EXT:
        return "video"
    raise ValueError("Upload one scan file (.ply, .glb, .obj), 3 or more photos (.jpg, .png), or one video.")


def num(opts, key, default):
    try:
        return float(opts.get(key, default))
    except (TypeError, ValueError):
        return default


def plan(job: dict) -> list[tuple[str, list[str]]]:
    """(step name, command) list for a job."""
    d, o, inp = job["dir"], job["options"], job["input"]
    py = sys.executable
    common = ["--truth", "", "--no-bench", "--out", str(d / "demo"), "--floor-offset", str(num(o, "floor_offset", 0))]
    if num(o, "clutter", 0) > 0:
        common += ["--clutter", str(int(num(o, "clutter", 0)))]
    if job["kind"] == "scan":
        cmd = [py, "-m", "echotwin.navigation.demo", "--scan", str(inp), *common, "--up", o.get("up", "auto")]
        if num(o, "crop", 0) > 0:
            cmd += ["--crop", str(num(o, "crop", 0))]
        if num(o, "scale", 1) != 1:
            cmd += ["--scale", str(num(o, "scale", 1))]
        return [("Map and robot run", cmd)]
    if not PERCEPTION_PY.exists():
        raise RuntimeError(f"VGGT environment not found at {PERCEPTION_PY}; photos and videos need it.")
    cloud = d / "cloud.ply"
    steps = [("3D reconstruction (VGGT)", [str(PERCEPTION_PY), "-m", "echotwin.perception.reconstruct", str(inp), "-o", str(cloud),
                                           "--frames", str(int(num(o, "frames", 13))),
                                           "--cam-height", str(num(o, "cam_height", 1.3))])]
    # The scale comes from the reconstruction step's printed hint (or the user's own value).
    fo = str(num(o, "floor_offset", 0))
    if o.get("objects", "on") == "on":
        steps += [("Object detection (YOLO11)", [str(PERCEPTION_PY), "-m", "echotwin.perception.detect", str(cloud)]),
                  ("Object map", [py, "-m", "echotwin.perception.objects", str(cloud), "-o", str(d / "objects"), "--up", "y",
                                  "--scale", "{scale}", "--floor-offset", fo]),
                  ("Review of the objects (NVIDIA)", [py, "-m", "echotwin.perception.review", str(cloud),
                                                      str(d / "objects_scene.json")])]
    run = [py, "-m", "echotwin.navigation.demo", "--scan", str(cloud), "--up", "y", "--scale", "{scale}", *common]
    if o.get("objects", "on") == "on":
        run += ["--objects", str(d / "objects_objects.json")]
    return steps + [("Map and robot run", run)]


def run_job(job: dict):
    log = job["dir"] / "log.txt"
    try:
        steps = plan(job)
        job["steps"] = [{"name": n, "state": "waiting"} for n, _ in steps]
        with GPU:
            for i, (name, cmd) in enumerate(steps):
                job["steps"][i]["state"] = "running"
                cmd = [c.replace("{scale}", str(job.get("scale", 1.0))) for c in cmd]
                t0 = time.time()
                with open(log, "a", encoding="utf-8") as fh:
                    fh.write(f"\n=== {name}\n$ {' '.join(cmd)}\n")
                    fh.flush()
                    p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace")
                    fh.write(p.stdout)
                job["steps"][i]["seconds"] = round(time.time() - t0)
                if p.returncode != 0:
                    plain = re.sub(r"\x1b\[[0-9;]*m", "", p.stdout)          # drop terminal colours
                    tail = [l for l in plain.splitlines() if l.strip() and "warn" not in l.lower()][-3:]
                    raise RuntimeError(f"{name} failed: " + " | ".join(tail))
                m = re.search(r"--scale ([0-9.]+)\s+\(then fix", p.stdout)
                if "retrying with" in p.stdout:
                    job["report"].append(re.findall(r"GPU out of memory: retrying with \d+ frames", p.stdout)[-1])
                if m and "scale" not in job:
                    job["scale"] = float(job["options"].get("scale_override") or m.group(1))
                job["steps"][i]["state"] = "done"
                keep = [l for l in p.stdout.splitlines()
                        if re.match(r"(quality|wall gaps|  [0-9.]+ m at|walls aligned|loaded|floor levelled|demo run"
                                    r"|objects:|world:|\d+ frames|wrote|camera height|labelled|\s+\S+\s+at \(|reviewed|No AI_API_KEY|  renamed|  removed)", l)]
                job["report"] += keep
        job["results"] = collect(job["dir"])
        job["state"] = "done"
        save_job(job)
    except Exception as e:                                    # shown in the page, full trace in the log
        job["state"], job["error"] = "failed", str(e)
        for s in job.get("steps", []):
            if s["state"] == "running":
                s["state"] = "failed"
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(traceback.format_exc())
        save_job(job)


def save_job(job: dict):
    """Keep finished runs across server restarts (runs/<id>/job.json)."""
    data = {**public(job), "created": job["created"], "example": job.get("example")}
    (job["dir"] / "job.json").write_text(json.dumps(data, default=str, indent=1))


def load_saved_jobs():
    for f in sorted(RUNS.glob("*/job.json")):
        try:
            data = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        data["dir"] = f.parent
        if data.get("state") == "done":
            data["results"] = collect(f.parent)               # picks up outputs added by newer code
        JOBS[data["id"]] = data


def collect(d: Path) -> dict:
    rel = lambda p: p.relative_to(RUNS).as_posix()            # noqa: E731
    out = {}
    for key, name in [("map", "demo/map_vs_world.png"), ("gif", "demo/demo_run.gif"),
                      ("errors", "demo/demo_errors.png"), ("objects_img", "objects_objects_clean.png")]:
        if (d / name).exists():
            out[key] = "/runs/" + rel(d / name)
    shots = sorted((d / "cloud_detections").glob("*.jpg"))
    if shots:
        out["detections"] = ["/runs/" + rel(f) for f in shots]
    if (d / "demo/summary.json").exists():
        out["summary"] = json.loads((d / "demo/summary.json").read_text())
    if (d / "objects_scene.json").exists():            # reviewed names when the review ran
        scene = json.loads((d / "objects_scene.json").read_text())
        out["objects"] = [{**o, "class": o.get("label") or o["class"]} for o in scene["objects"]]
        out["review"] = scene.get("review")
        shots = sorted((d / "objects_scene_review").glob("*.jpg"))
        if shots:
            out["review_photos"] = ["/runs/" + rel(f) for f in shots]
    elif (d / "objects_objects.json").exists():
        out["objects"] = json.loads((d / "objects_objects.json").read_text())["objects"]
    return out


def new_job(kind: str, inp: Path, jdir: Path, options: dict, title: str, example: str | None = None) -> dict:
    job = {"id": jdir.name, "dir": jdir, "kind": kind, "input": inp, "options": options, "title": title,
           "state": "running", "steps": [], "report": [], "created": time.time(), "example": example}
    if options.get("scale_override"):
        job["scale"] = float(options["scale_override"])
    JOBS[job["id"]] = job
    threading.Thread(target=run_job, args=(job,), daemon=True).start()
    return job


def job_dir(label: str) -> Path:
    d = RUNS / (time.strftime("%Y%m%d-%H%M%S") + "-" + re.sub(r"[^a-z0-9]+", "-", label.lower())[:30].strip("-"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def public(job: dict) -> dict:
    keys = ("id", "kind", "title", "state", "steps", "report", "error", "results", "scale", "options", "example")
    return {k: job[k] for k in keys if k in job}


def parse_multipart(body: bytes, ctype: str):
    """(fields, files) from a multipart/form-data body. files: list of (filename, bytes)."""
    m = re.search(r'boundary="?([^";]+)"?', ctype)
    if not m:
        raise ValueError("missing multipart boundary")
    fields, files = {}, []
    for part in body.split(b"--" + m.group(1).encode())[1:-1]:
        head, _, data = part.partition(b"\r\n\r\n")
        data = data[:-2] if data.endswith(b"\r\n") else data
        disp = head.decode("utf-8", "replace")
        name = re.search(r'name="([^"]*)"', disp)
        fname = re.search(r'filename="([^"]*)"', disp)
        if fname and fname.group(1):
            files.append((Path(fname.group(1)).name, data))
        elif name:
            fields[name.group(1)] = data.decode("utf-8", "replace")
    return fields, files


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):                             # keep the console quiet
        pass

    def send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def json(self, obj, code=200):
        self.send(code, json.dumps(obj, default=str).encode())

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self.send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/pitch":
            self.send_response(301); self.send_header("Location", "/pitch/"); self.end_headers(); return
        if path == "/pitch/":
            return self.send(200, pitch_page().encode("utf-8"), "text/html; charset=utf-8")
        if path.startswith("/pitch/"):
            f = (PITCH / path[len("/pitch/"):]).resolve()
            if PITCH.resolve() in f.parents and f.is_file():
                return self.send(200, f.read_bytes(), mimetypes.guess_type(f.name)[0] or "application/octet-stream")
        if path == "/api/examples":
            return self.json(load_examples())
        if path == "/api/jobs":
            return self.json([public(j) for j in sorted(JOBS.values(), key=lambda j: -j["created"])])
        if path.startswith("/api/jobs/"):
            job = JOBS.get(path.rsplit("/", 1)[-1])
            return self.json(public(job)) if job else self.json({"error": "no such job"}, 404)
        if path == "/api/robot-status":
            import urllib.request
            try:
                urllib.request.urlopen(ROBOT_URL, timeout=2)
                return self.json({"ok": True, "url": ROBOT_URL})
            except Exception:
                return self.json({"ok": False, "url": ROBOT_URL})
        if path.startswith("/runs/"):
            f = (RUNS / path[len("/runs/"):]).resolve()
            if RUNS.resolve() in f.parents and f.is_file():
                return self.send(200, f.read_bytes(), mimetypes.guess_type(f.name)[0] or "application/octet-stream")
        self.json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            if self.path == "/api/analyze":
                n = int(self.headers.get("Content-Length", 0))
                fields, files = parse_multipart(self.rfile.read(n), self.headers.get("Content-Type", ""))
                if not files:
                    raise ValueError("No files received.")
                paths = [Path(f) for f, _ in files]
                kind = kind_of(paths)
                d = job_dir(paths[0].stem if len(paths) == 1 else f"{len(paths)} photos")
                inp_dir = d / "input"
                inp_dir.mkdir()
                for fname, data in files:
                    (inp_dir / fname).write_bytes(data)
                inp = inp_dir if kind == "photos" else inp_dir / files[0][0]
                title = files[0][0] if len(files) == 1 else f"{len(files)} photos"
                return self.json(public(new_job(kind, inp, d, fields, title)))
            if self.path.startswith("/api/examples/"):
                ex = {e["id"]: e for e in load_examples()}.get(self.path.rsplit("/", 1)[-1])
                if not ex or not ex["available"]:
                    raise ValueError("Example not found in demo_inputs/.")
                src = EXAMPLES_DIR / ex["path"]
                files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT) if src.is_dir() else [src]
                kind = kind_of(files)
                d = job_dir(ex["id"])
                return self.json(public(new_job(kind, src, d, dict(ex["options"]), ex["title"], ex["id"])))
            # Send a finished job's scan to the Robot simulator
            m = re.match(r"/api/jobs/([^/]+)/send-to-robot$", self.path)
            if m:
                job = JOBS.get(m.group(1))
                if not job:
                    return self.json({"error": "job not found"}, 404)
                if job.get("state") != "done":
                    return self.json({"error": "job not finished yet"}, 400)
                result = send_to_robot(job)
                return self.json(result, 200 if result["ok"] else 502)
            self.json({"error": "not found"}, 404)
        except ValueError as e:
            self.json({"error": str(e)}, 400)


PITCH_BAR = """
<style>
.appbar { position: sticky; top: 0; z-index: 5; display: flex; justify-content: space-between; align-items: center; gap: 12px;
          padding: 8px 16px; background: var(--frame); border-bottom: 1px solid var(--rule); font: 13px var(--body); }
.appbar a, .live a { color: var(--signal); font-weight: 600; text-decoration: none; }
.appbar a:hover, .live a:hover { text-decoration: underline; }
.live { position: absolute; right: 1.6cqw; bottom: 1.4cqw; display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; z-index: 2; }
.live a { font: 600 max(11px, 1.05cqw) var(--mono); border: 1px solid var(--signal); background: var(--frame);
          padding: .4em .8em; border-radius: 4px; }
@media (max-width: 760px) { .live { position: static; padding: 0 18px 18px; justify-content: flex-start; } }
</style>
<div class="appbar"><span>Scan-to-Sonar · pitch</span><a href="/">Open the live demo →</a></div>
"""

PITCH_SCRIPT = """
<script>
fetch("/api/examples").then((r) => r.json()).then((exs) => {
  const byId = Object.fromEntries(exs.map((e) => [e.id, e]));
  document.querySelectorAll("[data-examples]").forEach((slide) => {
    const box = document.createElement("div"); box.className = "live";
    slide.dataset.examples.split(" ").forEach((id) => {
      const ex = byId[id]; if (!ex) return;
      const a = document.createElement("a"); a.href = "/?show=" + id;
      a.textContent = "Show live: " + ex.title; box.appendChild(a);
    });
    slide.appendChild(box);
  });
});
</script>
"""


def pitch_page() -> str:
    """The deck source (docs/pitch/pitch_src.html) as a full page, with a bar back to the demo."""
    src = (PITCH / "pitch_src.html").read_text(encoding="utf-8")
    head = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
            '<base href="/pitch/">')
    title_end = src.index("</style>") + len("</style>")         # title, fonts and styles go in <head>
    return head + src[:title_end] + "</head><body>" + PITCH_BAR + src[title_end:] + PITCH_SCRIPT + "</body></html>"


def load_examples():
    exs = json.loads((WEB / "examples.json").read_text(encoding="utf-8"))
    done = sorted((j for j in JOBS.values() if j.get("state") == "done" and j.get("example")),
                  key=lambda j: j["created"])
    last = {j["example"]: j["id"] for j in done}                # newest wins
    for e in exs:
        e["available"] = (EXAMPLES_DIR / e["path"]).exists()
        e["last"] = last.get(e["id"])
    return exs


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    RUNS.mkdir(exist_ok=True)
    load_saved_jobs()
    print(f"Scan-to-Sonar demo on http://127.0.0.1:{port}  (runs in {RUNS})")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
