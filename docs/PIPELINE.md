# Pipeline

One run goes through eight steps. Each step hands one file to the next.

| # | Step | Code | Hands over |
|---|---|---|---|
| 1 | Capture: phone photos or a video (see `CAPTURE_GUIDE.md`) | phone | `examples/*/` |
| 2 | Reconstruct: VGGT, camera and depth heads | `perception/reconstruct.py` | cloud `.ply`, `.cams.npz`, `.pix.npz` |
| 3 | Level and scale: floor RANSAC, wall alignment, scale from phone height or `--ref` | `perception/mapping.py` | metric, levelled points |
| 4 | Detect: YOLO11-seg per photo, labels onto points, multi-view voting | `perception/detect.py`, `objects.py` | `<stem>_objects.json`, map |
| 5 | Review: a vision model (NVIDIA) looks at each object in 2 or more photos; a name, shape or removal needs two photos to agree. Without a key: skipped | `perception/review.py`, `scene/review.py` | updated `<stem>_scene.json`, marked photos in `<stem>_scene_review/` |
| 6 | Scene file: class catalog (shape, movable, surface), support, one file both halves read | `scene/catalog.py`, `scene/schema.py` | `<stem>_scene.json` ([format](../echotwin/scene/schema.md)) |
| 7a | Room: occupancy map, sonar particle filter | `navigation/` | GIF, error plots |
| 7b | Surface: pick a table-sized window, build the digital twin (objects to move, fixed obstacles) | `scene/to_twin.py` | twin `.zip` |
| 8 | Act and learn: say it and it does it, says it is already done, or asks to be shown | `robot/` | demos, skills |

The review needs `AI_API_KEY` in `.env` (see `THIRD_PARTY.md`: photos are sent to NVIDIA). Without a key the detector's names are kept.
All capture alternatives are compared in `APPROACHES.md`.

## From the phone

A sweep scan on the phone runs steps 2 to 7b by itself (`robot/twin_import/pipeline.py`, command list in
`perception/pipeline.py`) and shows progress with a **Skip, use quick mode** button. About 2 minutes for 10
photos on an RTX 2050 (reconstruction 90 s, detection 20 s). Quick mode (`twin_import/photos.py`, one photo,
no GPU) takes over when you skip, when `PERCEPTION_PY` is not set, with fewer than 3 photos, when a step
fails, or when the 3D scene has nothing small enough to move.

Settings in `.env`, all paths relative to the repository or absolute: `PERCEPTION_PY`, `PERCEPTION_MAPS_PY`
(if the GPU environment has no scipy), `VGGT_PATH` (default `third_party/vggt`), `SCAN_MODE`
(`auto` | `quick` | `3d`), `SCAN_FRAMES`, `SCAN_CAM_HEIGHT`.

Known limits: the scale comes from the phone height (`SCAN_CAM_HEIGHT`), so absolute sizes can be off by a
factor; the twin is scaled to the sim table anyway. The table texture (photo of the table) is not built in 3D
mode yet.
