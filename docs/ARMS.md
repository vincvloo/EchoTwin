# Arms: use the one you have

An arm sits on a base: fixed at the table edge, or on wheels. Robots (a base and an arm) are in `docs/ROBOTS.md`.

The simulated robot is an arm with a two-pad gripper. The arm is not hard-coded: a small JSON file describes it, and
`echotwin/robot/arm.py` builds it into the scene. Pick one with `ARM=` in `.env`.

| `ARM=` | What it is |
|---|---|
| (empty) or `builtin` | A 5-joint arm written in code, SO-101 sized, two parallel pads, opens 8 cm. Works offline. All numbers in `docs/RESULTS.md` are for this one. |
| `so_arm100` | The SO-ARM100 from MuJoCo Menagerie (Apache-2.0), one moving jaw. Fetch it first (below). Passes `--check`, not yet the full skill table. |
| `path/to/my_arm.json` | Your own arm (relative to the repository, or absolute). |

```bash
python -m echotwin.robot.arm --list                    # which arms are described and which files are missing
python -m echotwin.robot.arm --download so_arm100      # about 3.4 MB into models/arms/ (git-ignored), nothing else downloads
python -m echotwin.robot.arm --check so_arm100         # loads it, prints its reach, tries a 4 cm cube and a 6 cm cylinder
```

## What the robot does with an arm

- The skills still say "tool to (x, y, z), jaws open or closed". Inverse kinematics (tool position plus "tool points
  down", the wrist roll lines the jaws up with the object) turns that into joint targets.
- The reachable ring is measured once per arm. A request outside it, or an object the jaws cannot hold, is declined
  with the reason ("it is 12 cm wide and my gripper opens 8 cm", "it is out of my reach").
- An object counts as held only when two pads squeeze it. It can slip, drop or tip over, as it would on hardware.

## Describe your own arm

Copy `echotwin/robot/arms/builtin.json` and change it. You need an MJCF (MuJoCo XML) of the arm with position
actuators, and these fields:

| Field | Meaning |
|---|---|
| `mjcf` | Path of the MJCF, relative to the repository, or `builtin` |
| `base_body` | The root body of the arm (it is attached to the table at the front edge) |
| `mount_yaw_deg` | Turn the arm about the vertical axis so it faces into the table (try 0 or 180) |
| `joints`, `actuators` | Exactly five each, in order: base rotation, three bending joints, wrist roll |
| `gripper` | `mode` `parallel` (two sliding pads) or `single` (one moving jaw, set `fixed_side` to 1 or -1), the gripper `actuators`, and their `open` and `closed` values |
| `tool` | `body` and `pos` (the point between the pads, in that body's frame), `point_axis` (the tool's axis that must point down), `close_axis` (the direction the jaws close along) |
| `pads` | Parts of the pad geom names (used to see what is held) |
| `home` | Five joint values to start from |
| `max_opening` | How far the jaws open, in metres (objects wider than this are declined) |
| `download` | Optional: repo, commit, path and files for `--download` |
| `tilts_deg` | Optional: angles the gripper may lean outward from straight down, like `[0, 20, 40]`. The reach is measured for each, and the arm leans only as far as a spot needs (default `[0]`: always straight down) |
| `pad_contact` | Optional: how the pads touch things, set on the pad geoms when the arm is loaded (your MJCF stays as it is): `friction` (slide, spin, roll), `condim` (4 adds spin friction), `solref`, `solimp`, `margin` |

Names are looked up in your MJCF, so the order of things inside it does not matter. A wrong name gives an error that
says which one. Then run `python -m echotwin.robot.arm --check path/to/my_arm.json`. If the cube or cylinder fails,
look at the grip first: the pads need friction (1.5 or more), the closing force a few newtons, and the tool point
a few millimetres above the lowest tip of the pads.

Credit: the SO-ARM100 model is by The Robot Studio and Google DeepMind (MuJoCo Menagerie), Apache-2.0.

## Switching the arm, and what size of arm it would take

In the dashboard, open Advanced, then "Arm size: what would it take?" (or press "What arm size would grab it?" when Pip
declines an object). The arm list shows every descriptor in `echotwin/robot/arms/` (an arm whose files are missing is
greyed out until you download it); switching rebuilds the twin with the other arm and keeps the table. It takes a few
seconds, and the choice is not saved: `ARM=` in `.env` is what the server starts with. Skills learned with one arm are not
checked on another.

**Arm size.** The slider makes the arm 0.75 to 4 times its described size; the table and the objects keep their sizes. For
each object the card says what size it needs: big enough for the jaws to open around it and to reach it, not so big that it
cannot bend down to it, or why no size works (thinner than 15 mm cannot be pinched from a table). "Size for every object it
can grab" applies the smallest size that takes them all.

How the arm is resized (`arm.resize`): lengths times k, masses times k^3, inertias times k^5; torque gains, damping and
force limits of the joints times k^5 (a sliding joint k^3 and k^4), so the bigger arm moves the same way relative to its
size. The jaws are the exception: their squeeze only grows with k, because it holds the object, whose weight does not
change (a gripper squeezing k^4 times harder pops a light round thing out). The reach grid and the base's distance from the
table edge grow with k.

Measured with the scripted skill (`PS.imagine`, the same check as `--check`):

| Arm | Sizes | Result |
|---|---|---|
| builtin, objects scaled with the arm (4 cm cube, 6 cm cylinder) | 0.75, 0.8, 0.9, 1, 1.5, 2, 2.5, 3, 4 | all picked and placed |
| builtin, a 15 cm box and a 12 cm round case at their size | 1 / 1.5 | declined (wider than the jaws) |
| builtin, the same | 2, 2.5, 3, 4 | all picked and placed |
| so_arm100, objects scaled with the arm | 1, 1.5, 2 | the cube yes; the cylinder lands 6 to 25 cm off from 1.5 |
| so_arm100 | 3 | fails |

So the size a card shows is a geometric answer for any arm; that the resized arm then also moves the object is measured for
the built-in arm only. Below 0.75 the built-in arm can hardly reach the table, so the slider stops there.

Sizes of the objects: from one photo they are estimates. If you know how wide one object really is, "Rescale the table"
scales the table, its objects and the camera by one factor; the arm does not change. Any factor is fine as long as the
result is plausible: every thing between 5 mm and 2 m, the mapped area at most 20 m wide.

## The SO-ARM100: what is fixed, what is not (measured)

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
the single jaw goes (`World._jaw_side`):

- **The wrist falls short.** Its roll stops at 160 degrees each way, so on the right of the table, turning the jaws to 90
  degrees fell 50 to 70 degrees short. The old check only asked "more this way than the other", so it kept that direction;
  the moving jaw then caught the end of the thing and the fixed jaw sat on top of it. Now the arm is asked both ways round,
  where the tool will really stand beside the thing, low down; when the way asked works (reachable, within 30 degrees, the
  arm clear of itself) it is kept, else the better one is used.
- **The open jaw hits the arm.** Close to the base the arm folds up and the open jaw swings into its shoulder
  (`World._hits_itself`, on the solver's copy). Where neither way round works the arm now says "I can't get my jaws around it
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
