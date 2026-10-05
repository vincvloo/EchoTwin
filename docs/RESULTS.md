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
