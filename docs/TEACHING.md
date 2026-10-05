# Teaching the robot

The robot does not have one fixed program per object. It learns how to handle a **size** of object: how wide it is to grip, how tall, how long (and an estimated weight).
The name does not matter: a skill learned on a mug carries over to a jar of similar size, and a 4 cm and a 7 cm box are
treated as different. When you ask it to move something it does one of three things:

1. **Already done.** It checks the table: the object is already where you asked.
2. **Do it.** It has moved something of this size before. It plans the move from what it learned, imagines it first
   on a copy of the twin (does it land on the spot, stay upright, bump nothing?), then does it.
3. **Teach me.** It has never moved something this size ("a tall thing, 6 cm wide and 10 cm tall"), or the imagined move looks wrong. It asks you to show it.

There are three ways to show it.

## 1. Drive it yourself

Open the phone page, hold the pad and tilt the phone (or use the keyboard on the laptop), and carry the object to
the place you asked for. The keys are `Z Q S D` to move, `A` and `E` for down and up, `Space` to grip (an AZERTY
layout).

While you drive, the robot records every step. When the object rests where you asked, it scores your demo and asks
"Keep this demo?". Say or tap **keep it**, or **discard**.

## 2. Film your hand

Film yourself moving one object with your hand. The robot works out **what moved and where it ended up**, replays
that in the twin with its own gripper, and offers to keep its own run as a demo.

### How to film

| Do | Why |
|---|---|
| Show the whole table with at least two other objects that you do not touch | The camera moves between the start and the end of your video. The robot lines the untouched objects up to cancel that motion, so it needs at least two |
| Keep your hands out of the picture for the first and last 1 to 2 seconds | It picks a clean frame (most objects visible, sharpest, no hand) from the start and from the end. Only those two moments are used |
| Move **one** object, clearly (more than 3 cm) | It looks for the one object that is not where the camera motion predicts |
| End next to another object, or clearly to the left, right, towards you or away | The result is a sentence like "next to the glass" or "to the left", which also works in the twin even though the twin's camera is different |
| Hold the phone about 45 cm above the table, looking down at about 50 degrees (like looking at a table from a chair) | The robot assumes this height and angle to turn pixels into positions on the table. Other heights still work, but distances get less exact |
| On the phone: tap record, do the move, wait 2 seconds, tap stop (up to 30 s) | The phone sends 5 pictures a second |
| When you upload a video file: 1080p is enough | A 4K video decodes much more slowly. Only the first and the last 20 % of the video are read |

The path your hand took in the middle is **not** used. Only where the object started and where it ended up.

### What happens to the video

1. **Pick two clean frames**, one near the start and one near the end.
2. **Find the objects** in both (the same table-and-objects detection as the quick scan) and match them by colour and
   relative size.
3. **Cancel the camera motion.** It finds the shift, rotation and zoom that lines up the most objects between the two
   frames. Those objects did not move.
4. **Find what moved.** An object that has a look-alike at the end, but not where the camera motion predicts, and that
   moved more than 3 cm.
5. **Describe where it went.** "Next to" another object if the gap is under 6 cm, otherwise left, right, towards you
   or away (8 cm for a small move, 20 cm for a bigger one).
6. **Match it to the twin** by colour and size, so "the small dark thing" in your video is object 2 in the twin. With
   no twin yet, it builds one from the first frame.
7. **Replay it in the twin** with the robot's gripper, using default grip settings.
8. If it worked ("That worked in my twin") it asks **"Keep this demo?"**. The robot's own run is stored as the demo,
   and the numbers below are learned from it.

### When it does not work

| The robot says | What happened | Try |
|---|---|---|
| That video is too short. Show me the whole move. | Fewer than 4 usable pictures | Record at least a few seconds |
| I need to see at least two things on the table, at the start and at the end. | Fewer than two objects detected in the start or end frame | Better light, plain table, whole table in view |
| I couldn't line up the start and the end of the video. Keep the same things in view. | Fewer than two untouched objects matched | Keep the same objects in view from start to end |
| I didn't see anything move. Try again, and move one thing clearly. | Nothing moved more than 3 cm | Move one object further |
| I saw something move, but I can't tell which object it is in my map. | The moved object has no match in the twin | Scan the table again, or load the matching twin first |
| I couldn't reproduce your video in my twin. Can you show me with the phone? | The replay did not put the object where you showed | Drive it yourself once (way 1) |

## 3. Let it practise

Say **practice** (or press **Practice in sim**). The robot tries moves in the twin with a different grip height,
lift, drop height and speed each time, keeps only the tries that land on the spot and stay upright, and stores them
as demos. It keeps 3 good tries per size class (flat, small, medium, tall) (it gives up after 12 tries). Moves include "next to",
a direction, and "on top of" (stacking). No video, no driving.

Say "practice the glass" to practise one object only.

## What it learns

For each kept demo it stores the object's measurements and four numbers:

| Number | Meaning |
|---|---|
| grip | where along its height the pads take it, from low (-1) to high (+1); thin things are always taken near the table |
| lift | how far to lift it above the tallest thing on the table |
| drop | how gently to set it down (height of the tool above the resting height at release) |
| speed | how fast you moved |

With one kept demo it can plan a move for objects of a similar size (within about 3 cm in width, height and length), using the
**median** of the demos, the closer in size the more a demo counts. It also
reports how sure it is. Uncertainty gets lower with more demos, higher when the new move is very different
from the ones it saw (a long move after only short ones), and higher when the demos disagreed. Above a limit,
or when the imagined move fails, it asks to be shown instead of guessing.

Ask it:

- **"How sure are you?"** for the percentage and how many demos it has.
- **"What have you learned?"** for the sizes of objects it knows.
- **"Why did you stop?"** after you pressed stop.

Your demos can also train a neural policy (`docs/LEARNING.md`).

Demos are saved as JSON files in `data/robot/episodes/`, with the instruction, the state at every step (hand,
grip, object, goal) and the actions. `python -m echotwin.robot.reset` deletes them.

## Limits

- The video route tells the robot **what to move and where**; it does not copy how your hand moved.
- Objects of very different sizes need a demo each. A glass does not teach it about a chocolate bar (a flat 2 cm bar),
  but two glasses of similar size share what they learned.
- Sizes come from the twin, so they are only as good as the scan until the table has a marker (PR10).
- Colour and size are used to recognise objects, so two look-alike objects can be confused.
- The twin is approximate (the camera height is assumed), so "next to" means about 6 cm, not millimetres.
