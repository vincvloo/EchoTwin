# Taking the photos

EchoTwin builds a 3D model of a table from ordinary phone photos or a short video. No special app and no LiDAR.

## What to capture

- **One table, one scene.** Do not mix photos of different places.
- **10 to 20 overlapping photos**, or the phone's **Sweep scan** (6 seconds, a few photos a second; 12 are used).
  Move slowly in an arc around the table and keep the whole table top in view.
- **Put the printed marker on the table** (below): it gives the true size of everything.
- **Hold the phone about 45 cm above the table, looking down**. Without a marker the size of everything is guessed from
  this height (`SCAN_CAM_HEIGHT`, default 0.45 m). If you hold it much higher or lower, set the real height.
- **Plain table, good light**, no strong reflections. Glass and shiny things are hard.
- **Hands out of the picture.** A hand in several photos can become an obstacle in the twin.
- **Put 3 to 6 objects on the table**, with some space between them. Objects under about 5 cm are often lost.

## The marker: true sizes

Without it, sizes are a guess and can be off by a factor of two. With it, they are measured.

1. Print the page: `python -m echotwin.perception.marker --print out/marker.png`, then print `out/marker.png` on A4 at
   **100 %** (no "fit to page"). Measure the black square with a ruler: it must be 100 mm. If your printer scaled it, set
   `MARKER_SIZE_CM` in `.env` to what you measured (for example 9.4).
2. Lay the page **flat** on the table, in the middle of the objects, not under anything. Keep it clean and not curled.
3. Take the photos as usual. The marker should be clearly visible, at least a hand's width in the picture, in **at least
   two photos from different sides**.
4. The robot says whether it found the marker. The dashboard shows "sizes measured with the marker" or "sizes estimated".

`python -m echotwin.perception.marker --find <folder>` tells you which photos show it.

## How to give the photos to EchoTwin

- On the phone: open `https://<laptop-ip>:8443/phone`, **Scan**, **Sweep scan**.
- On the laptop: the dashboard, **Upload & check photos** (or a video), or the example photos.

A 3D model takes about 2 to 3 minutes on a laptop GPU; the dashboard shows progress and a **Skip, use quick mode**
button. Quick mode (one photo, no GPU) is the fallback in every case below.

## When the 3D model is not used

| The robot says | Why | What to do |
|---|---|---|
| (quick mode, nothing said) | `PERCEPTION_PY` not set, fewer than 3 photos, or `SCAN_MODE=quick` | Set `PERCEPTION_PY` in `.env` (see the README) |
| I found N things, but nothing small enough to move | The scene is a room, or the objects are too big or too far | Photograph the table from closer |
| The 3D model found no objects | Objects too small or too few photos | Closer, more photos, more light |
| The 3D model failed at '...' | A step crashed (the server log has the line) | Often the GPU is short of memory: close other programs |
| I did not see the marker, so the sizes are estimated | The marker is missing, too small, or in shadow or glare | Closer, flat, in the light |
| I saw the marker, but the photos disagree about its size | The page is curled, or only seen from a very flat angle | Flatten it, add a photo from above |

## Known limits

- Without the marker the size of the scene comes from the phone height, so absolute sizes can be off by a factor.
- The reconstructed table is flat to only 2 to 3 cm.
- Objects are named by a detector that sometimes says look-alike names ("mouse" for an earbud case); the review step with
  an `AI_API_KEY` improves this.
- The table texture comes from the photos by median, so a thing that stands in the same place in every photo is painted into it.

For other ways to capture a place (Scaniverse, LiDAR) and how they compare, see `APPROACHES.md`.
