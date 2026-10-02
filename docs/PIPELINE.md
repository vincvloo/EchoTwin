# Pipeline

One run goes through eight steps. Each step hands one file to the next.

| # | Step | Code | Hands over |
|---|---|---|---|
| 1 | Capture: phone photos or a video (see `CAPTURE_GUIDE.md`) | phone | `examples/*/` |
| 2 | Reconstruct: VGGT, camera and depth heads | `perception/reconstruct.py` | cloud `.ply`, `.cams.npz`, `.pix.npz` |
| 3 | Level and scale: floor RANSAC, wall alignment, scale from phone height or `--ref` | `perception/mapping.py` | metric, levelled points |
| 4 | Detect: YOLO11-seg per photo, labels onto points, multi-view voting | `perception/detect.py`, `objects.py` | `<stem>_objects.json`, map |
| 5 | Review (planned, PR3): NVIDIA vision confirms, renames, skips | not yet | reviewed objects |
| 6 | Scene file: class catalog (shape, movable, surface), support, one file both halves read | `scene/catalog.py`, `scene/schema.py` | `<stem>_scene.json` ([format](../echotwin/scene/schema.md)) |
| 7a | Room: occupancy map, sonar particle filter | `navigation/` | GIF, error plots |
| 7b | Surface: pick a table-sized window, build the digital twin (objects to move, fixed obstacles) | `scene/to_twin.py` | twin `.zip` |
| 8 | Act and learn: say it and it does it, says it is already done, or asks to be shown | `robot/` | demos, skills |

Step 5 does not exist yet; see the PR list in `TASKS.md`.
All capture alternatives are compared in `APPROACHES.md`.
