# scene.json (version 1)

The one file that perception writes and the rest of EchoTwin reads. Metres, in the levelled map frame
(z up, x right, y forward), the same frame as the occupancy map. Written by
`python -m echotwin.perception.objects`, read by `python -m echotwin.scene.to_twin`.

```json
{
  "format": "echotwin-scene", "version": 1, "name": "lounge",
  "map": {"origin": [-3.26, -1.05], "res": 0.03, "width": 197, "height": 152},
  "objects": [
    {"id": "o7", "class": "potted plant", "label": "potted plant", "source": "yolo", "conf": null,
     "x": 1.728, "y": 1.233, "size_x": 0.24, "size_y": 0.36, "height": 0.25, "base_z": 0.0,
     "shape": "cylinder", "movable": true, "surface": false, "on": null, "photos": 4, "points": 812}
  ]
}
```

| Field | Meaning |
|---|---|
| `class` | Detector class name (any name; the 80 YOLO classes are in `catalog.py`) |
| `label` | Name shown to people. Starts as `class`; the NVIDIA review (PR3) may rename it |
| `source` | Who named it: `yolo`, `nvidia` (reviewed), `quick` (one-photo importer), `manual` |
| `x`, `y`, `size_x`, `size_y` | Footprint centre and size on the floor plan |
| `height`, `base_z` | Top and bottom of the object above the floor. A cup on a table has `base_z` near the table height |
| `shape` | `flat`, `box`, `cylinder` or `round`, from the catalog (the shapes the robot has skills for) |
| `movable` | A small thing the gripper can move (catalog entry and size: at most 40 cm wide, 40 cm tall) |
| `surface` | Can hold other things (dining table, couch, bed, bench) |
| `on` | Id of the surface this object stands on, or `null` |

## From scene to twin

The robot works on a table, so `to_twin` picks a table-sized window: a surface that holds small things, or
else the densest group of small things. Movable objects in the window become objects to move, furniture that
reaches into it becomes a fixed obstacle (clipped to the window), and the rest is left out. The window is
scaled to the sim table (`sim_scale`).

Adding a class: put it in `catalog.py`. An unknown class still works; shape and `movable` are guessed from
its name and size.
