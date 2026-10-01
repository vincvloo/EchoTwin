# Pipeline

One run goes through eight steps. Each step hands one file to the next.

| # | Step | Code | Hands over |
|---|---|---|---|
| 1 | Capture: phone photos or a video (see `CAPTURE_GUIDE.md`) | phone | `examples/*/` |
| 2 | Reconstruct: VGGT, camera and depth heads | `perception/reconstruct.py` | cloud `.ply`, `.cams.npz`, `.pix.npz` |
| 3 | Level and scale: floor RANSAC, wall alignment, scale from phone height or `--ref` | `perception/mapping.py` | metric, levelled points |
| 4 | Detect: YOLO11-seg per photo, labels onto points, multi-view voting | `perception/detect.py`, `objects.py` | `<stem>_objects.json`, map |
| 5 | Review (planned, PR3): NVIDIA vision confirms, renames, skips | not yet | reviewed objects |
| 6 | Scene file (planned, PR2): one file both halves read | not yet | `scene.json` |
| 7a | Room: occupancy map, sonar particle filter | `navigation/` | GIF, error plots |
| 7b | Surface: pick a table, build the digital twin | `scene/bridge.py` (now), `scene/to_twin.py` (PR2) | twin `.zip` |
| 8 | Act and learn: say it and it does it, says it is already done, or asks to be shown | `robot/` | demos, skills |

Steps 5 and 6 and the table-fitting part of 7b do not exist yet; see the PR list in `TASKS.md`.
All capture alternatives are compared in `APPROACHES.md`.
