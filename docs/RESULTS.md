# Results

## Benchmark 1 (2026-09-29), synthetic flat

Setup: 7 x 5 m flat with interior wall, sofa, table (legs only in the sonar band), kitchen island,
cabinet, armchair, sideboard. The real world differs from the scanned map by a glass partition, a
chair moved after the scan and a 1.5 % scan scale error. Robot wanders at 0.25 m/s, 4 Hz updates,
unknown start pose (global localization), 400 steps (100 s), 15 seeds per configuration.
"Converged" = error under 30 cm and 20 deg, held for 5 s.

| Configuration | Converged | Distance to converge (median) | Time (median) | Position error after | Heading error after |
|---|---|---|---|---|---|
| 4 sonars (F/L/R/B) | 14/15 | 0.8 m | 4 s | 4 cm | 1.7 deg |
| 2 sonars (F/B) | 14/15 | 2.5 m | 13 s | 58 cm | 21.6 deg |
| 4 sonars, narrow 10 deg beam | 14/15 | 0.3 m | 2 s | 4 cm | 1.5 deg |
| 4 sonars, harsh (30 deg beam, 3x noise, 8 % ghost echoes) | 13/15 | 1.1 m | 6 s | 5 cm | 2.2 deg |

Demo run (`out/demo_run.gif`, `out/demo_errors.png`): converges after about 9 s, robot is
kidnapped at 60 s, filter recovers in about 7 s.

## How to read this

- **4 sonars is the minimum worth building.** With 2 sonars (front/back) the filter briefly locks on
  and then drifts: 58 cm average error after "convergence".
- **Narrow beams help more than you would expect.** A 10 degree cone localizes about 3x faster than
  25 degrees. Worth considering sensors with a narrower beam than the HC-SR04.
- **These numbers are optimistic.** The simulated sensor uses the same physics the filter assumes.
  Missing: side lobes, multipath, cross-talk, non-synchronous firing, wheel slip bursts, real scan
  holes near the floor. Task T4 adds these. Plan on 10-30 cm in a real room.
- One seed fails in every configuration. Check it before T4 (likely a start pose in the narrow
  space next to the sideboard, where the room looks symmetric to four sonars).

## Detector benchmark (2026-10-03): which model finds and names the objects?

Command: `python -m echotwin.perception.bench_detect yolo11s-seg.pt yolo26s-seg.pt yoloe-26s-seg-pf.pt yoloe-26s-seg.pt:text`.
Scenes: the 13 lounge photos and the 10 table photos in `examples/`. Expected objects were written by hand from the
photos (`echotwin/perception/bench_data.json`, 12 items for the lounge, 4 for the table). **Recall** = expected objects
found in enough photos under one of their accepted names. **False** = share of detections whose name matches nothing
expected. RTX 2050 (4 GB), 640 px, confidence 0.25.

| Model | Lounge: recall, false | Table: recall, false | ms / photo | GPU MB |
|---|---|---|---|---|
| yolo11s-seg (the previous default, 80 classes) | 67 %, 0 % | 75 %, 7 % | 46 to 55 | 281 |
| yolo26s-seg (newer, 80 classes) | 58 %, 1 % | 75 %, 3 % | 45 to 50 | 297 |
| yoloe-26s-seg-pf (open vocabulary, built-in names) | 67 %, 27 % | 75 %, 78 % | 68 to 116 | 553 |
| **yoloe-26s-seg + text prompts** (open vocabulary, 65 generic indoor names) | **92 %, 2 %** | 75 %, 19 % | 63 to 119 | 351 |

Times differ by up to 2x between runs on this laptop GPU.

What this says:

- **A newer closed-set YOLO is not the answer.** yolo26s is not better than yolo11s here (worse on the lounge).
  Both can only say the 80 YOLO names, so cushions, rugs, speakers, screens and posters are invisible to them.
- **Open vocabulary with its own names is too noisy.** The prompt-free model has a vocabulary of thousands of words
  and uses them: "honey", "swinge", "chemistry lab" for a table with a glass and a charger.
- **Open vocabulary with a text prompt list is the clear winner on the lounge.** It finds cushions, rugs, speakers
  and a projector screen, and names the coffee tables as tables and the sofa as a sofa. Same speed class, 70 MB more
  GPU memory, 29 MB of weights plus a 242 MB text encoder.
- **Tabletop objects are the weak spot of every model.** All four miss the chocolate bar. The text model does name it
  ("chocolate bar") but in only 1 of the 10 photos, so it does not count. It also calls the earbud case a "remote
  control" (the 19 % false rate). Small, low-contrast objects need either a better model or the vision-model review.
- **Things on a table were being swallowed by the table.** The object map merged small blobs inside a bigger one (meant for
  sofa cushions), so a cup on a desk became part of the desk. Tables now never absorb what stands on them, and people and
  hands are left out of the scene. On the table photos the map now has the cup, the computer mouse (the earbud case), a
  bottle and five unnamed small things.
- **In the real pipeline (lounge cloud, same photos):** the object map has 13 objects with the right names (couch,
  coffee table, carpet, vase, 3 plants) instead of 8 where a couch and a coffee table were both "chair".

Limits of the benchmark: two scenes, scene-level scoring, accepted names written by hand (read them), and the
prompt list was written after seeing the scenes, so the text-prompt score is optimistic. Models that are not in
the Ultralytics family (for example OWLv2 or Grounding DINO, which have permissive licences) were not tested.

