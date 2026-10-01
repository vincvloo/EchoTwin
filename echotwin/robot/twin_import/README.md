# twin_import: from the real world to the digital twin

Everything that turns outside data into a twin lives here. The server only calls this package; nothing in
here touches the simulator, the websocket or the robot. That makes it safe to work on (or replace) in parallel.

| Source | Module | Entry point |
|---|---|---|
| Photos (everyday objects from one photo) | `photos.py` | `import_photos(frames, pitches, ctx)` |
| Twin files (save / edit / load) | `layout_file.py` | `layout_to_doc`, `doc_to_layout`, `export_zip`, `read_upload` |
| 3D scans (.glb/.gltf/.obj/.ply/.stl from Polycam, Scaniverse, Luma…) | `meshes.py` | `convert(src, out_dir, stem)`, `fit_scale` |
| The contract | `contract.py` | `TwinContext`, `ImportError_` |

The geometry helpers the photo importer uses (table segmentation, camera model) are in
`echotwin/robot/features/everyday.py`; they are shared with the video-demo code.

## Plugging in your own photo importer

Write an async function with the same signature and point the app at it, no edits to this package needed:

```python
# my_twin/importer.py
from echotwin.robot.scene import Layout
from echotwin.robot.twin_import import TwinContext

async def import_photos(frames: list[bytes], pitches: list, ctx: TwinContext) -> None:
    lay = Layout()
    lay.props = [{"name": "chocolate box", "shape": "flat", "pos": (0.1, 0.0),       # sim metres
                  "size": (0.1, 0.06, 0.006), "rgb": (0.9, 0.9, 0.9)}]             # half sizes, sim metres
    lay.meta["sim_scale"] = 2.0                  # sim metres per real metre
    sid, folder = ctx.new_dir()                  # files you write here are served at /scans/<sid>/
    ctx.apply(lay, {"id": sid, "mode": "everyday", "props": ["chocolate box"],
                    "greeting": "I've mapped your table. What should I move?"})
```

```
# .env
TWIN_PHOTO_IMPORTER=my_twin.importer:import_photos
```

`pitches[i]` is the phone's downward camera angle for frame i (degrees, or None). Optional: `ctx.progress(stage, data)`
for dashboard progress, `ctx.ask_ai_json(jpeg, prompt)` to ask the vision model, `ctx.rename(props, line)` to update
names/shapes later, `ctx.say(line)` to talk.

A Layout prop (everyday object) is a dict: `name`, `shape` (flat|box|cylinder|round), `pos` (x, y), `size` (half
x, y, z), `rgb`, optional `yaw` (deg), `skin` (png path: photo wrapped on the shape), `mesh` (obj/stl path, centred) +
`mesh_scale` + `mesh_texture`. Scenery 3D scans go in `layout.scene`: `{file, texture, pos, euler, scale}`.

## Twin file format

See the docstring at the top of `layout_file.py`. Real centimetres, table centre at the origin, x right, y away.
The dashboard's **Save twin** writes one; **Load twin** reads one; the **Twin editor** table edits the current one.
