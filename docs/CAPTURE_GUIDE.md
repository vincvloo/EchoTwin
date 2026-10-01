# Scanning a room with the OnePlus 12

Goal: a metric 3D scan whose **lowest 40 cm** (where the robot's sonars look) is complete.

## Before you start

1. Install **Google Play Services for AR** from the Play Store.
2. Install one scanning app: Polycam, Scaniverse or KIRI Engine. Check that it can export
   `.ply`, `.glb` or `.obj` on your plan.
3. Pick the room: some furniture at floor level, lights on, little glass.
4. Tape-measure two reference distances (one wall length, one table edge). Write them down.

## Scaniverse (what we use)

- Tested 2026-09-29 on the OnePlus 12 (DARE Campus lounge stage). The mesh build failed in the app,
  but the **splat** export worked: a `.ply` of 1.26M Gaussian splats, metric, Y up.
- `mesh_to_grid` reads splat `.ply` files directly (keeps opaque splats under 5 cm). Export as
  **PLY**, not SPZ (compressed splat, not supported).
- Splats also capture the space around the target (hall, view through windows). Add `--crop 4`
  (metres around the densest area).
- If the robot drives on a raised surface (stage, platform), add `--floor-offset <height>`.

## Fallback: photos or a video with VGGT

No usable scan? 10-30 overlapping photos or a 30-60 s video work too, but have no metric scale:
`..\vggt-env\Scripts\python.exe scripts/video_to_ply.py <folder or .mp4> -o data/x.ply`, then
convert with `--up y --scale <printed hint>` and fix the scale with `--ref` and a tape measurement.
Keep the floor in view; one scene per run (do not mix photos of different places).

## Scanning

- Use the app's room or photo scanning mode (the OnePlus 12 has no LiDAR).
- **Hold the phone low, 30-60 cm above the floor**, pointing slightly down. Phones scan the floor
  line badly, and that is exactly the band the robot uses.
- Walk slowly along the walls. Keep plain walls at an angle so the camera sees texture next to them.
- Cover furniture legs and the bottom of cabinets and sofas.
- Finish where you started so the tracking can close the loop.
- One room per scan to start with (about 5-10 minutes).

## Export and convert

1. Export as `.ply` (point cloud) or `.glb` / `.obj` (mesh). Keep metric scale, no re-centering.
2. Copy it into `data/` on the laptop, then:
   ```bash
   python -m sonarloc.mesh_to_grid data/my_room.glb -o out/my_room --res 0.03 --band 0.05 0.35
   ```
   The up axis is detected automatically (printed as `up axis: ...`). Force it with `--up y` or `--up z`
   if the detection is wrong.
3. Read the quality lines: floor coverage should be above ~80 %, wall gaps should only be doorways.
   Open `out/my_room.png`. Walls and furniture legs should be black, the floor white.
4. Check scale: `python scripts/measure.py data/my_room.glb` prints wall-face positions along the map
   axes; compare the matching difference with the tape measure (or count cells x 3 cm on the PNG).
   Within 3 cm is good.

## Common problems

| Symptom | Cause | Fix |
|---|---|---|
| Floor comes out black | Floor points inside the band (tilted or thick floor) | Raise `--band` lower bound to 0.08 |
| Walls have gaps | Plain walls, poor texture | Rescan closer and at an angle; or lower `--res` to 0.05 |
| Map rotated 90 degrees | `align_walls` puts the longer side along x | Expected, harmless |
| Whole map black or empty | Wrong up axis (check the printed `up axis`) | Try `--up y` or `--up z` |
| Map is huge, room is a small patch | Scan caught the surroundings (splats do) | `--crop 4` |
| Platform or stage is one black block | Floor detected below the platform | `--floor-offset <platform height>` |
