# Results

Measurements of the parts that are in use. The sonar localization results from the removed mobile base are archived in [history/sonar/RESULTS.md](history/sonar/RESULTS.md).

## Detector benchmark (2026-10-03): which model finds and names the objects?

Command: `python -m echotwin.perception.bench_detect yolo11s-seg.pt yolo26s-seg.pt yoloe-26s-seg-pf.pt yoloe-26s-seg.pt:text=coco yoloe-26s-seg.pt:text=objects365 yoloe-26s-seg.pt:text=lvis yoloe-26s-seg.pt:text`.
Scenes: the 13 lounge photos and the 10 table photos in `examples/`. Expected objects were written by hand from the
photos (`echotwin/perception/bench_data.json`: 12 items for the lounge, 4 for the table). **Recall** = expected objects
found in enough photos under one of their correct names; a detected name satisfies one expected item only. **False** =
share of detections whose name matches nothing expected. RTX 2050 (4 GB), 640 px, confidence 0.25.

| Model | Lounge: recall, false | Table: recall, false | ms / photo | GPU MB |
|---|---|---|---|---|
| yolo11s-seg (previous default, 80 classes) | 50 %, 1 % | 25 %, 37 % | 56 | 281 |
| yolo26s-seg (newer, 80 classes) | 50 %, 1 % | 25 %, 49 % | 44 | 297 |
| yoloe-26s-seg-pf (open vocabulary, its own 4585 names) | 67 %, 29 % | 50 %, 76 % | 66 | 553 |
| yoloe-26s-seg, text prompts = COCO 80 names | 42 %, 6 % | 25 %, 52 % | 57 | 351 |
| **yoloe-26s-seg, text prompts = Objects365 (365 names, public)** | **83 %, 8 %** | **50 %, 45 %** | 59 | 409 |
| yoloe-26s-seg, text prompts = LVIS (1198 names, public) | 75 %, 14 % | 25 %, 70 % | 67 | 481 |
| yoloe-26s-seg, text prompts = our catalog list (65 names, written by us) | 92 %, 5 % | 25 %, 40 % | 62 | 398 |

Times differ by up to 2x between runs on this laptop GPU.

What this says:

- **Rooms and furniture: an open-vocabulary model with a text prompt list is clearly better.** With a public list we
  did not write (Objects365) it finds 83 % of the lounge objects against 50 % for the 80-class models, in the same
  time. It names cushions ("pillow"), rugs ("carpet"), coffee tables and speakers, which YOLO11 cannot say at all.
- **Our own 65-word list scores higher (92 %) but that is partly fitting.** We wrote it after seeing these photos; the
  gap to the public list, 9 points, is the size of that effect. The default is therefore the public list.
- **The vocabulary matters a lot, in both directions.** Plain COCO names as prompts are no better than the closed-set
  model. A huge list (LVIS, 1198 names) gets more wrong names on small things ("dagger", "crowbar", "honey" for a
  charger on a table: 70 % of detections). The prompt-free model has the same problem.
- **Small tabletop objects are not solved by any model.** The best is 50 %, and the false rate stays at 40 % or more.
  Names come out as look-alikes ("mouse" for the earbud case, "remote" for the chocolate bar). A newer closed-set
  model (yolo26s) does not help; neither does a larger vocabulary. What helps is the vision-model review step, and the
  geometry fallback: the robot only needs a shape and a size, not the right word.
- **A correction.** The first version of this benchmark let one wrong name ("remote") count for three expected
  objects and accepted look-alike names, which gave the table 75 to 100 %. The scoring above is the corrected one.
- **In the real pipeline (same photos):** the lounge object map has 13 objects with the right names (couch, coffee
  table, carpet, vase, three plants) instead of 8 where a couch and a coffee table were both "chair". On the table
  photos it finds the cup, the bottle, the mouse-like earbud case, and the charger as "converter".
- **Things standing on a table were being swallowed by the table.** The object map merged small blobs inside a bigger
  one (meant for sofa cushions), so a cup on a desk became part of the desk. Tables now never absorb what stands on
  them, and people and hands are left out of the scene.

Limits of the benchmark: two scenes, scene-level scoring, expected names written by hand (read them), and four
small objects on the table. Treat it as a direction, not a leaderboard. Models outside the Ultralytics family (for
example OWLv2 or Grounding DINO, which have permissive licences) were not tested.

Use your own words: put one name per line in a text file and set `DETECT_PROMPTS=my_words.txt` in `.env`, or try it
first with `bench_detect ... yoloe-26s-seg.pt:text=my_words.txt`.
