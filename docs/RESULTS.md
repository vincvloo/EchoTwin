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
example OWLv2 or Grounding DINO, which have permissive licences) were not tested here; see the next section.

Use your own words: put one name per line in a text file and set `DETECT_PROMPTS=my_words.txt` in `.env`, or try it
first with `bench_detect ... yoloe-26s-seg.pt:text=my_words.txt`.

## Do the default skills survive a physical simulation? (PR8)

Before PR8 the "gripper" was a floating hand that glued the nearest object to itself, on a table drawn twice the real
size. Every skill worked because nothing could go wrong. PR8 replaced it with a 5-joint arm, real scale, a grasp made
only of friction and force, and a mass from the object's size. `python -m echotwin.robot.skillcheck` runs each default
skill on four shapes (real sizes: flat 10 x 6 x 2 cm, box 5 x 4 x 4 cm, cylinder 7 cm wide and 10 cm tall, ball 5 cm)
and three tasks, with the object at a random reachable spot, and judges the result like a real run.

| Shape | Task | Before (kinematic, 6 trials) | Now (arm, contact grasp, 12 trials) | Seconds before / now |
|---|---|---|---|---|
| flat | next to | 100 % | 100 % | 7.1 / 5.6 |
| flat | to the left | 100 % | 100 % | 7.4 / 6.7 |
| flat | on top of | 100 % | 100 % | 7.0 / 8.0 |
| box | next to | 100 % | 100 % | 7.1 / 6.9 |
| box | to the left | 100 % | 100 % | 7.2 / 8.4 |
| box | on top of | 100 % | 100 % | 7.2 / 7.4 |
| cylinder | next to | 100 % | 83 % | 9.7 / 13.5 |
| cylinder | to the left | 100 % | 92 % | 9.2 / 13.7 |
| cylinder | on top of | 100 % | 67 % | 9.4 / 14.7 |
| ball | next to | 100 % | 100 % | 7.8 / 6.7 |
| ball | to the left | 100 % | 100 % | 7.4 / 6.6 |
| ball | on top of | 100 % | 92 % | 6.8 / 6.8 |

Overall: 100 % before, 94 % now (the last column of the run is a mean over the successful trials).

What this says:

- **The old 100 % said nothing.** It confirmed the premise: a glued object cannot fail. The new numbers are the first
  that could transfer to a real arm, and they show where it would break.
- **Tall objects are the weak spot.** The 10 cm cylinder is gripped by the lower part, swings when the arm turns,
  and tips over when set down; stacking it needs a carry height at the edge of what the arm reaches (22 cm at best,
  less far from the base). Failures: tips over, slides off the other object, lands off the spot.
- **Thin objects need the pad tips at the table.** With the grip aimed at the middle of the object, a 2 cm bar was
  lost during the lift (17 % next to, 0 % otherwise). The default now aims lower for thin things; the learned
  `grip` value (from demonstrations or practice) still moves it.
- **Things the robot now declines, with a reason:** objects wider than the jaws (the built-in arm opens 8 cm, so a
  12 cm box is refused), objects thinner than 1.5 cm (a pinch cannot lift them off a table), anything or any spot
  outside the arm's reach (8 to 34 cm from its base, in the built-in arm). The check draws only reachable spots; the
  refusals have their own tests.
- **The SO-ARM100 does not hold up yet (19 % overall, 6 trials per cell).** Its single moving jaw presses the object
  against the fixed jaw, and the grasp is much less forgiving than two symmetric pads: ball 83 / 67 / 0 %, box 50 /
  33 / 0 %, cylinder and flat 0 %. It passes `--check` (a 4 cm cube and a 6 cm cylinder) but not the full table, so
  the built-in arm is the default and the SO-ARM100 stays selectable (`ARM=so_arm100`) for work on it.

Limits: one simulator, one built-in arm, boxes of fixed size, 12 trials per cell (one failure is 8 points). Treat it
as a direction. The numbers will change when PR9 (skills from measurements) and PR12 (closed loop) land.

## Do skills carry over to other sizes? (PR9)

Skills used to be filed under four shape labels. In PR9 they are filed under measurements (grip width, height, length),
and a new object is planned from the demos of similar-sized objects. `python -m echotwin.robot.skillcheck --transfer`
tests this: the robot practises with random styles on 12 random objects (box, cylinder, ball; 3 to 7 cm wide, 2 to 12 cm
tall), keeps what worked, then moves objects it practised on ("seen sizes", 36 moves) and others ("unseen sizes", 72 moves)
with four ways of choosing the skill. Same object, spot and task in every row.

| Skill chosen by | Seen sizes | Unseen sizes | Hard set: seen | Hard set: unseen |
|---|---|---|---|---|
| tuned defaults (nothing learned) | 97 % | 99 % | 89 % | 81 % |
| shape label (the old way) | 92 % | 99 % | 81 % | 71 % |
| size class (flat, small, medium, tall) | 92 % | 99 % | 78 % | 71 % |
| measurements (nearest sizes) | 92 % | 99 % | 78 % | 68 % |

The hard set is only tall (8 to 13 cm), wide (5.5 to 7 cm) boxes and cylinders, where the default skill fails most.

What this says:

- **Measurements match the old shape label; they do not beat it.** On unseen sizes the three learned rows are within noise of
  each other (72 moves per cell, one move is 1.4 points). The gain of PR9 is not a higher success rate here. It is that the robot now
  asks when an object is unlike anything it has seen and does not when a similar one is known, whatever the label or the name says
  (9 of the 72 unseen moves would have been "show me" in the first set).
- **Learning from practice does not beat the tuned defaults.** Practice tries random styles and keeps every success, so the median
  of what worked is no better than the hand-tuned default, and a little worse on the hard set. This is a problem of how demos are
  chosen, not of how they are keyed. Keeping only the best tries, or narrower practice around a good style, is for PR13.
- **The simulator has little room above the defaults** for objects of this size: 97 to 99 % without learning. A real arm
  (PR11) will not be so kind, and that is where size-keyed skills should start to matter.

Limits: one seed, 12 objects, random styles; sizes come from the twin and are only as good as the scan.

## What do the skills lose when the arm is not the simulation? (PR11)

There is no real arm yet, so the "real" back-end was run against a **mock arm**: a second MuJoCo world with objects 1.4 times
heavier, pads with 80 % of the friction, servos that lag behind their targets (45 % of the gap per tick), encoders that read a
little off (about 0.7 degrees per joint, fixed) and noisy (0.1 degree), and half the tool speed. The skills plan and imagine in the
simulation (the twin), the mock arm does the move, and the result is judged on the mock's own objects (a perfect camera).
`python -m echotwin.robot.skillcheck --backend both --trials 12`, same objects and spots in both columns:

| Shape | Task | sim | mock arm | gap |
|---|---|---|---|---|
| flat | next to | 100 % | 100 % | 0 |
| flat | to the left | 100 % | 100 % | 0 |
| flat | on top of | 100 % | 100 % | 0 |
| box | next to | 100 % | 92 % | -8 |
| box | to the left | 100 % | 100 % | 0 |
| box | on top of | 100 % | 100 % | 0 |
| cylinder | next to | 83 % | 67 % | -17 |
| cylinder | to the left | 92 % | 67 % | -25 |
| cylinder | on top of | 67 % | 75 % | +8 |
| ball | next to | 100 % | 100 % | 0 |
| ball | to the left | 100 % | 100 % | 0 |
| ball | on top of | 92 % | 100 % | +8 |
| **all** | | **94 %** | **92 %** | **-3** |

*Correction (PR12).* PR11 first reported 87 % for the mock arm (cylinder stacking 25 %). That was partly a bug in the real
back-end: the tool target kept moving while the lagging arm caught up, so after a move was "done" the arm kept sinking and
pressed the object. PR12 limits how far the target may lead the measured tool (2 cm). The table above is the corrected one.

What this says:

- **The mock arm costs little once the controller is right**: 94 % in the simulation, 92 % on the mock arm. The weak spot is the tall
  cylinder (7 cm wide in 8.2 cm jaws): it needs the tool within about 6 mm sideways, and the mock's encoders are off by about 1 cm.
- **This is a stress test, not a prediction.** The mock is as hard as its parameters; real servos, friction and a camera will differ.
  `python -m echotwin.robot.runlog` prints the success rate per back-end from `data/robot/runs.jsonl`.


## Does looking again and retrying help? (PR12)

The closed loop (`CLOSED_LOOP=on`): park the arm high and to the side, look (a top-down camera frame, the blob that differs from the
table near where the object should be), plan the pick from what is seen, then check that the jaws closed on something, that it
lifted, and where it ended up, with up to two retries. `python -m echotwin.robot.skillcheck --ab [--backend mock] [--disturb before|during]`,
6 trials per cell, same objects and spots in both columns, judged on the mock arm's own world (not on what the robot believes):

| Situation | open loop | closed loop |
|---|---|---|
| Mock arm, nothing disturbed | 100 % | 79 % |
| Mock arm, object pushed 3 to 6 cm after the plan was made | 10 % | 71 % |
| Mock arm, object pushed 3 to 6 cm just as the jaws close | 10 % | 43 % |
| Plain simulation, nothing disturbed | 94 % | 96 % |

What this says:

- **When something changes, the loop is the difference between failing and mostly working.** A pushed object makes the open loop miss
  (10 %); looking again recovers it to 71 %, and a miss at the last moment is noticed and retried (43 %).
- **When nothing changes, it costs success on the mock arm (100 % to 79 %).** The causes are known and not all fixed: the camera
  sometimes does not see a low-contrast object (a pale flat bar, a ball) or an object next to the arm's base and reports "I can't see it",
  and a check can fire on a grasp that was fine (contacts flicker, a tilted cylinder reads wider), after which the retry disturbs a good
  position. In the plain simulation, where looking is exact, the loop is neutral (94 % to 96 %).
- **The tall cylinder is not helped.** Its pick fails because the jaws are 6 mm wider than the object and the arm is about 1 cm off:
  retrying with the same error lands in the same place. Aligning the tool to the object in the image (visual servoing) would address it
  and is not done.
- **So the default is a trade-off.** `CLOSED_LOOP=auto` turns it on for the real and mock arms, where objects can move and the robot
  only believes where they are; the plain simulation keeps the open loop. If your table is calm and your camera is poor, set `CLOSED_LOOP=off`.

Limits: 6 trials per cell (one failure is 17 points); the camera is a clean render of the simulation without cast shadows; no real
camera, lighting or calibration was tested.

## Can a network learn the move from demonstrations? (PR13)

A small network (3 layers of 512, state in, a chunk of the next 10 actions out, temporal ensembling: `docs/LEARNING.md`) was trained
by behaviour cloning on moves made by the scripted skill in the simulation, then asked to do the same 12 tasks as `skillcheck`
(12 trials per cell, the same objects and spots in every row). Training takes about a minute on the laptop GPU. The "demos" are
whole moves that worked; the scripted skill is the teacher.

| Moves by | Demos | Simulation | Mock arm |
|---|---|---|---|
| scripted skill (the teacher) | - | **94 %** | 92 % |
| learned policy | 600 | 73 % | 17 % |
| learned policy | 200 | 47 % | |
| learned policy | 50 | 25 % | |
| learned policy, demos made with noise on the executed motion (DART) | 600 / 200 / 50 | 62 % / 35 % / 31 % | 21 % (600) |

Per task, 600 demos without noise (simulation): flat 83 / 92 / 67 %, box 92 / 83 / 100 %, cylinder 42 / 42 / 8 %, ball 100 / 92 / 75 %
(next to / to the left / on top of).

What this says:

- **It works, and it is data hungry.** From 25 % at 50 demos to 73 % at 600, still climbing. It picks, carries and puts down objects it
  was never shown at that spot. It does not reach its teacher (94 %) and has no reason to exceed it.
- **The tall cylinder is the failure** (8 to 42 %): millimetre tolerances, and the teacher itself only manages 67 to 92 % there.
- **Noise in the demos did not help here.** With 600 demos the noisy set scored 62 % against 73 % for the clean one (200 demos: 35 % against 47 %;
  50 demos: 31 % against 25 %). A sensible guess is that 3 cm/s of noise blurs the last centimetres, where the grasp is decided.
  So `demos.py` makes clean demos unless asked.
- **It does not carry over to the mock arm** (17 to 21 %, against 92 % for the scripted skill). The scripted skill closes the loop on the
  measured tool position with a stiff proportional rule; the network was trained on a perfect, fast simulated arm and has never seen lag,
  encoder offsets or half speed. This is the same lesson as for any sim-to-real policy: the data has to include the real arm (or be randomised to cover it).
- **Two things made the difference between 1 % and 64 % on the way**, and are worth knowing: giving the network the offsets between the tool,
  the object and the goal (it cannot learn "difference" from a few hundred layouts), and demos with objects turned both ways (every early
  demo had the jaws at the same angle, so an object needing the other angle sent the input off the scale). A heavier weight on the grip output
  helped it let go of the object.

Limits: simulated demos from one scripted teacher, no images, no real robot, 12 trials per cell (one failure is 8 points), one training run
per row (no seeds averaged).

## Does looking at the gripper fix the tight fit? (PR14)

The tall cylinder (7 cm in 8.2 cm jaws, 6 mm to spare on each side) is the weakest skill because the arm is about a centimetre off.
With `ALIGN=on` the robot looks at its own gripper before going down: at the hover over the object, just above it, and at the grip
height it finds the two pads in the camera image, compares their midpoint with where the joints say the tool is, and moves the pick by the
difference (`find_pads`, `observe_tool`, the `align` step of the closed loop). Only for a tight fit (under 12 mm to spare per side). Mock arm,
closed loop with and without the alignment, same objects and spots, judged on the mock arm's own world:

| Mock arm, cylinder cells | closed loop | + align |
|---|---|---|
| encoder error 0.7 degrees per joint (the default), 6 trials per cell | 67 % / 50 % / 83 % | 83 % / 0 % / 67 % |
| 1.1 degrees per joint, 12 trials per cell | 17 % / 42 % / 50 % (36 % all) | 25 % / 17 % / 50 % (31 % all) |
| 1.7 degrees per joint, 12 trials per cell | 25 % / 25 % / 42 % (31 % all) | 33 % / 42 % / 67 % (47 % all) |

(next to / to the left / on top of.) Every other shape is unchanged by construction: a loose fit is not aligned (the whole table at the default
error: 79 % without, 75 % with, the difference is the cylinder cells).

What this says:

- **It halves the position error but does not make the tight fit reliable.** With the gripper over the cylinder at the grip height, the
  distance between the pads' midpoint and the object's centre is 3.7 mm on average with the alignment against 6.8 mm without (6 arms, one
  encoder error each; the unaligned error ranges from 2 to 10 mm). The jaws allow 6 mm.
- **It pays off only when the arm is badly off.** At 1.7 degrees per joint the cylinder improves from 31 % to 47 %; at the default error and
  at 1.1 degrees the results are within noise (12 trials per cell: one trial is 8 points) or worse in a cell. The alignment is off by default.
- **The limit is the measurement.** Seen from above the pads give the tool position within 2 mm on average, 6 mm at the 90th percentile and
  9 mm at worst; near the base the forearm hides a pad, and a lone pad is not used. With 6 mm of clearance and 2 to 6 mm of measurement
  error there is little left to gain. A better measurement (a mark on the hand, a second camera, or looking from the side) is the next step,
  not a better controller.
- **Three things went wrong on the way**, and are fixed: the first version measured the gripper before the arm had arrived at the hover and
  "corrected" 4 cm of travel (success fell from 79 % to 49 %); it took dark blue boxes for pads; and it aligned objects that fit loosely.

Limits: one camera model (a clean render), the jaws of the built-in arm, the mock arm's encoder error as a random offset per joint.

## Detectors that are not YOLO: OWLv2 and Grounding DINO (PR15)

Both are Apache-2.0 (YOLOE is AGPL-3.0), so a detector as good as YOLOE would remove the AGPL dependency from the detection step. Same
benchmark, same photos and scoring as above, same session, `transformers` 5.18, RTX 2050 (4 GB). OWLv2 is `google/owlv2-base-patch16-ensemble`
(fp16 on the GPU, 960 px), Grounding DINO is `IDEA-Research/grounding-dino-tiny` (fp32). Command:
`python -m echotwin.perception.bench_detect owlv2:text=objects365 gdino-tiny:text=objects365 yoloe-26s-seg.pt:text=objects365`.
Recall, false rate, with the public Objects365 vocabulary (365 names), threshold 0.25:

| Model | Lounge | Table | ms / photo | GPU MB | Licence |
|---|---|---|---|---|---|
| yoloe-26s-seg, text=objects365 (the default) | 83 %, 8 % | 50 %, 45 % | 67 | 325 | AGPL-3.0 |
| OWLv2 | 75 %, 14 % | 50 %, 58 % | 612 | 995 | Apache-2.0 |
| Grounding DINO tiny | 83 %, 16 % | 50 %, 78 % | 9 875 | 2 142 | Apache-2.0 |

With our own 65-word list (written after seeing these photos, so partly fitted) and with COCO's 80 names:

| Model, vocabulary | Lounge | Table | ms / photo |
|---|---|---|---|
| OWLv2, catalog | 100 %, 6 % | 50 %, 35 % | 586 |
| Grounding DINO tiny, catalog | 92 %, 19 % | 100 %, 62 % | 2 081 |
| OWLv2, COCO | 33 %, 5 % | 25 %, 48 % | 568 |
| Grounding DINO tiny, COCO | 50 %, 12 % | 25 %, 63 % | 2 130 |

What this says:

- **Neither is better than YOLOE with a fair vocabulary.** With the public list OWLv2 finds fewer lounge objects (75 % against 83 %) and names
  more things wrongly (58 % against 45 % on the table); Grounding DINO matches the recall and is the least accurate (78 % false on the table).
- **They are much slower.** OWLv2 takes about 0.6 s per photo (9 times YOLOE) and 1 GB; Grounding DINO about 10 s per photo with 365 names
  (it reads only about 40 names at a time, so the list is asked in 10 chunks) and 2.1 GB. For 12 photos that is 7 s against 120 s.
- **The small table objects are not solved by any model**, again: the chocolate bar and the earbud case are missed with the public vocabulary
  by all three. Only Grounding DINO with our fitted list finds all four table objects, at a 62 % false rate.
- **OWLv2 with our 65-word list is the best lounge result so far** (100 %, 6 % false), but that list was written after seeing the photos.
  It is a measure of what a good vocabulary buys, not of the model.
- **Neither gives masks.** The pipeline labels the 3D points through the detector's masks (`detect.py`); a box-only detector would need a box
  used as a coarse mask, or a segmenter such as SAM (Apache-2.0) after it. That work is only worth doing for a detector that wins the naming test,
  and neither did. So the default stays YOLOE; the AGPL-free detection step is possible (OWLv2 plus a segmenter) but costs recall, precision and time.

Limits: the same two scenes and hand-written names as before (a direction, not a leaderboard); one run each, times vary up to 2x; thresholds not
tuned per model (0.25 for all; OWLv2 and Grounding DINO scores are calibrated differently from YOLO's, so another threshold could shift the
recall/false trade-off).

## A policy for an arm that is not the simulation: domain randomisation (PR16)

PR13's policy scored 17 to 21 % on the mock arm. The fix tried here is the standard one: make the demonstrations with arms that vary. The scripted
expert drives the real-arm back-end on a *randomised* mock arm (per episode: lag 0.25 to 0.65, objects 1.0 to 1.8 times heavier, pad friction 0.6 to
1.0, encoder error 0 to 0.03 rad per joint, encoder noise 0.5 to 3 mrad, half speed), the observations are what that back-end reports, the recorded
action is the expert's, and only moves that worked on the mock's own objects are kept. Same network and training as PR13, 600 demos each, 12 trials per
cell, same objects and spots. Three arms to test on: the simulation; the default mock arm (lag 0.45, objects 1.4 times heavier, friction 0.8, encoder
error 0.012); and a **harder, held-out mock arm outside every training range** (lag 0.2, objects 1.8 times heavier, friction 0.6, encoder error 0.04).

| Policy trained on | Simulation | Default mock arm | Held-out hard arm |
|---|---|---|---|
| perfect simulated arm (PR13) | 73 % | 18 % | 14 % |
| randomised mock arms | 72 % | **64 %** | **59 %** |
| half simulation, half randomised mock arms | 68 % | 62 % | 51 % |
| *the scripted skill, for reference* | 94 % | 92 % | 65 % |

Per task, randomised-mock policy on the default mock arm (next to / to the left / on top of): flat 58 / 75 / 58 %, box 92 / 92 / 92 %, cylinder 0 / 8 / 0 %,
ball 92 / 100 / 100 %.

What this says:

- **Randomising the arm in the data closes most of the gap.** The same network goes from 18 % to 64 % on the default mock arm and from 14 % to 59 % on an
  arm it never saw, and loses nothing on the simulation (72 % against 73 %). On the hard arm it is within 6 points of the scripted skill (65 %),
  which itself falls from 92 % to 65 % there: the hard arm is hard for everyone.
- **Mixing in perfect-arm demos did not help** (62 % and 51 % against 64 % and 59 %): the cheap data dilutes the data that matters.
- **The boxes and the balls are nearly solved** (92 to 100 %), the flat bar is the middle (33 to 75 %), the **tall cylinder stays at 0 to 8 %**:
  6 mm of clearance and about a centimetre of arm error are beyond what a policy that only sees the arm's own readings can do (see PR14 for looking at the gripper).
- **What it is not:** a result on hardware. The held-out arm is the same kind of simulation as the training arms, with different numbers. A policy that
  survives it is more robust to lag, offsets and weight, not proven on a real arm, a real camera or real friction.

Limits: one training run per row (no seeds averaged), 12 trials per cell (one failure is 8 points), 600 demos each; the demos are still from one scripted
teacher, so the policy can only approach it.

## The SO-ARM100's single jaw, step by step (PRs #31 to #37)

A single moving jaw holds a thing off-centre, pressed against the fixed jaw. Two fixes in the skill (`prop_skills`):

- **The thing, not the tool point, follows the plan.** When the wrist turns on the way, the held thing swung around the tool
  point and landed about 1.3 cm off. Moves that carry something are corrected by where the thing really is (`_held_off`);
  a box or a round case now lands 5 to 6 mm from the spot. A parallel gripper is not affected.
- **No waiting at the ceiling.** The plan asked for a carry height above the SO-ARM100's ceiling at that distance (5 cm),
  so every such step waited out the 80-tick limit. A step more than 1 cm out of reach is now done once the tool is where it
  can get to and at rest; it is still aimed at the step, which keeps the carry slow enough that tall things do not swing.

Then three more (PR "tilted grasps"):

- **The fixed jaw on the side the arm can really hold it.** Closing the jaws along a line can be done two ways round, and the
  arm solver accepted either. The wrist cannot always turn half a turn, so at some spots it held the jaws the other way round
  and the fixed jaw came down on top of the thing instead of beside it: the "lands 12 cm off" failures were things that were
  never picked up. `World.grasp_yaw` now asks the arm which way round it can hold its jaws over the thing (single jaw only).
- **A wider gap for the fixed jaw** (8 mm instead of 3): at long reach the arm is not precise enough to come down 3 mm beside
  a thing without catching its edge. Cost: the thing ends up about 1 cm less precisely placed when the jaw lets go.
- **Tilted grasps** (`tilts_deg: [0, 20, 40]`): the gripper leans outward when straight down cannot reach. At 5 cm height it
  now works 12 to 36 cm from its base (was 12 to 30), at 9 cm 18 to 34 (was 18 to 24), so far from its base it carries at
  5 cm instead of dragging at 2. The skill table does not show it (its spots are all within the straight-down reach): the
  table gives 31 % with or without tilting.

Skill table (`skillcheck`, 6 trials per cell, same seed):

| SO-ARM100 | next to | to the left | on top of |
|---|---|---|---|
| flat | 0 -> 33 % | 0 -> 33 % | 0 -> 17 % |
| box | 50 -> 83 % | 33 -> 50 % | 0 -> 17 % |
| round | 83 -> 83 % | 67 -> 50 % | 0 -> 0 % |
| all | | | 19 -> 31 % |

Its cylinder (7 cm) is wider than its jaws: declined, not failed, so 31 % is about 42 % of what it can hold at all. The
built-in arm stays at 94 % (same demos, messages and positions as before). What still fails most: stacking (it tips over
or slides off) and round things carried to the side. Higher pad friction (`pad_contact`) did not help.

Then stacking (PR "stacking"). The carry height was one number for the whole table, but the arm's ceiling drops far from its
base: over the spot to stack on, the tool was held lower than planned and the carried thing hit the side or the top edge of
the other one. Now:

- **The carry height is the lowest ceiling on the way** (`World.path_ceiling`: over the pick, over the place, and between;
  for a mobile base, from where it stands at each end).
- **The palm** (`World.palm`): how far the tool can go down over a thing's top before the hand sits on it, measured with rays
  up from the thing, jaws open (5.4 cm built-in, 5.7 cm SO-ARM100). A thing taller than that sticks up into the hand and
  hangs lower. Under a low ceiling, the thing is held lower down so its bottom clears the other top by 1.5 cm.
- **It says so when it cannot**: "I can't lift it high enough over the box from here" (`refusal(name, goal, on=...)`), instead
  of trying and knocking the other thing over. skillcheck draws another layout in that case, as for other refusals, so the
  random layouts after it change too.

| SO-ARM100 | next to | to the left | on top of |
|---|---|---|---|
| flat | 33 -> 33 % | 33 -> 33 % | 17 -> 50 % |
| box | 83 -> 83 % | 50 -> 83 % | 17 -> 67 % |
| round | 83 -> 83 % | 50 -> 50 % | 0 -> 50 % |
| all | | | 31 -> 44 % |

The "to the left" box change comes from the different random layouts, not from this change. The built-in arm: 94 % -> 93 %
on the whole table (one cylinder stack in different layouts), 88 -> 93 % and 89 -> 92 % for cylinder and ball (12 trials,
seeds 8 and 9); on the same layouts its cylinder stacks fail less (6 of 16 before, 3 of 16 after, one layout now declined).
A scripted session gives the same messages and positions as before. What still fails: flat things slip out of the single jaw
when lifted, and balls tip over when set down.

Then the flat things (PR "flat grip"). They did not slip: the jaws were not across them. Three causes, all in which way round
the single jaw goes (`World._jaw_fits`):

- **The wrist falls short.** Its roll stops at 160 degrees each way, so on the right of the table, turning the jaws to 90
  degrees fell 50 to 70 degrees short. The old check only asked "more this way than the other", so it kept that direction;
  the moving jaw then caught the end of the thing and the fixed jaw sat on top of it. Now the arm is asked both ways round,
  where the tool will really stand beside the thing, low down; when the way asked works (reachable, within 30 degrees, the
  arm clear of itself) it is kept, else the better one is used.
- **The open jaw hits the arm.** Close to the base the arm folds up and the open jaw swings into its shoulder
  (`World._collides`, on the solver's copy). Where neither way round works the arm now says "I can't get my jaws around it
  from here" (a fixed base; about 12 cm in front of the base and at the far edge of the reach).
- **The side chosen again while carrying.** The correction that keeps a held thing on the plan (`_held_off`) asked for the
  jaw side again from where the thing was at that moment. When the answer flipped, the correction jumped by 7 cm and the
  thing was swung off. It now uses the side the jaws really hold it from (`grasp_offset(name, yaw)`).

| SO-ARM100 | next to | to the left | on top of |
|---|---|---|---|
| flat | 33 -> 67 % | 33 -> 83 % | 50 -> 67 % |
| box | 83 -> 83 % | 83 -> 100 % | 67 -> 67 % |
| round | 83 -> 67 % | 50 -> 83 % | 50 -> 50 % |
| all | | | 44 -> 56 % |

With 12 trials (seed 11), flat 67 / 75 / 25 % -> 75 / 100 / 50 %, box 50 / 67 / 50 % -> 67 / 100 / 75 %, round
75 / 83 / 83 % -> 75 / 83 / 75 % (one trial). 56 % is about 75 % of what its jaws can hold. The built-in arm (two pads)
is not affected: the same results in every cell, the same scripted session.

Then the balls (PR "ball judging"). Most balls that "tipped over" had not: they lay on the spot, within half a centimetre,
rolled 30 to 50 degrees. For a ball that is not a failure, but the judge measured how far its up axis leant, as for a box.
`World.has_up` says which things have no up (a round thing whose three sizes are within 15 % of each other: a ball, not an
egg or a lentil), and the judge (`prop_skills.tilt_deg`) does not count their lean. Only the ball cells change:

| SO-ARM100, 12 trials | next to | to the left | on top of |
|---|---|---|---|
| round, seed 11 | 83 -> 92 % | 67 -> 92 % | 75 -> 75 % |
| round, seed 13 | 75 -> 92 % | 75 -> 83 % | 75 -> 83 % |

Flat things and boxes give the same results in every cell, the built-in arm stays at 93 % and the scripted session is the same.

What the balls still do wrong, and what did not help:

- Far from its base the SO-ARM100 leans its tool out to hover at the carry height, which puts the gripper over the ball it
  is coming for; it presses the ball into the table and rolls it away. The reach grid's rows are 4 cm apart and it
  underestimates the ceiling by 2 to 3.5 cm in the middle of the reach, so the arm leans where it would not need to. Finding
  the ceiling between the rows, and trying straight down first, fixed that ball, but cost flat things and boxes 12 of 144
  trials (other spots became reachable and other paths were taken). Hovering lower, straight down, and waiting for the lean
  to settle before going down made no measurable difference. None of it is in this PR.
- A ball set on a box sometimes rolls off it.

Then the ball on a box (PR "ball stack"). It did not roll off: it hit the box on the way. The ceiling along the carry was
looked at in 5 places; far out to the right the reach grid's ceiling dips (9 cm, then 5, then 9 again: cells where the solver
did not find the pose), the carry ran through the dip, and the ball, hanging under the tool, hit the box's side. Now:

- `World.path_ceiling` looks every centimetre along the way. Where the ball cannot clear the box, the arm says so ("I can't
  lift it high enough over the box from here").
- skillcheck draws up to 60 layouts the arm would do. When it declines all 60 it used to run the last one anyway, and the
  knock counted as "bumps"; now the trial is "refused", as for a thing too wide for the jaws. The rates are the same (both
  are failures), the reason is honest.

Stacking, SO-ARM100, 12 trials, seeds 11 and 13 together: 52 -> 57 of 72 (flat 13 -> 16, box 19 -> 20, ball 20 -> 21).
The built-in arm stays at 93 % and the scripted session is the same.

The SO-ARM100 declines most ball-on-box stacks: of 80 random layouts, 65 because it cannot lift the ball high enough over the
box (it needs the tool about 8 cm up over the box, and its grid ceiling is 5 cm over most of the table), 11 because it cannot
get its jaws around the ball there, 4 it would do. The grid is cautious (rows 4 cm apart, the real ceiling is 2 to 3.5 cm
higher in the middle of the reach), so a finer ceiling would let it stack more; the first try at that (see above) cost flat
things and boxes elsewhere.

Then the finer reach map (PR "straight ceiling"). The reach grid has rows 4 cm apart, and some rows were missed where the
solver did not find the pose: for the SO-ARM100 the ceiling read 5 cm from 24 to 30 cm out, where the arm really reaches 9
to 7.5 cm pointing straight down. Two changes:

- **The straight-down ceiling is found between the rows** (`Workspace.tops`, bisection, to about 1 mm), for each distance
  from the base. It is now a smooth curve (SO-ARM100: 2.6 cm at 10 cm out, 10 cm at 20 to 22 cm, 2.5 cm at 32 cm). The
  ceilings when leaning out stay on the rows: refining those too (the first try) let the arm hover leaning over the thing it
  was about to pick up.
- **One lean per distance from the base** (`Workspace.column_tilt`), from the table to the ceiling: straight down where that
  reaches about as high as leaning (SO-ARM100: up to 30 cm out), else the smallest lean that reaches the table and that
  height. Before, the lean was chosen by height, so going down to a thing or up from it the arm could start leaning next to
  it; the open jaw swung over a flat thing and flipped it.

Results (SO-ARM100, 12 trials per cell, seeds 11 and 13, and flat at 17): 211 -> 211 of 252 in all. Balls 65 -> 69 of 72,
boxes 64 -> 66, flat stacking 20 -> 24 of 36, flat "next to" and "to the left" 62 -> 52 of 72. On the same layouts (flat
"next to" and "to the left", seeds 21, 23 and 29) flat is 51 -> 49 of 72: in the table above part of the flat loss comes
from different layouts (declined stacks draw new ones). Ball-on-box layouts it would do: 4 -> 9 of 80. The built-in arm
(its ceiling rises 2 to 3 cm too): 94, 92 and 93 % on its usual checks; the scripted session gives the same messages and
positions, its recorded moves differ.

## Geometry, not labels: measured on varied objects (2026-10)

The robot now reads each object's built geometry (`features/geometry.py`) instead of its shape label, scans build any
rounded or tapered thing from its outline, and the single jaw is chosen at grip height (see docs/ARMS.md). Measured with
`skillcheck` at seed 21, the same layouts before (main, with this skillcheck) and after: 12 trials per classic cell,
30 per varied cell (`--shapes varied`: boxes of any proportions, flat things, cylinders, balls, eggs and meshes revolved
from silhouettes, at any rotation).

| | before | after |
|---|---|---|
| built-in arm, classic table | 141 / 144 | 141 / 144 (every cell the same) |
| built-in arm, varied objects | 83 / 90 | 83 / 90 (every cell the same) |
| SO-ARM100, classic table | 96 / 144 | 98 / 144 |
| SO-ARM100, varied objects | 57 / 90 | 66 / 90 |

On the SO-ARM100's varied objects "tips over" fell from 21 to 11. Per kind after: balls 12 of 12, eggs 14 of 19, flat
things 15 of 19, bottles 6 of 8; tapered glasses (0 of 2) and cups (1 of 4) are its weakest. The scripted session gives
the same messages and positions; its recorded moves differ.

Two changes were tried and taken out because these checks showed they hurt: finding the leaning ceilings between the
grid's rows (the arm worked at the very edge of its leaning reach and thrashed), and a rule against the hand hanging over
the thing when leaning (it chose sides the arm could not execute).
