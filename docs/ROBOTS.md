# Robots: a base and an arm

A robot is a small JSON file in `echotwin/robot/robots/`: a base and an arm. The arm is any arm descriptor
(`docs/ARMS.md`), at any size. The dashboard picks the robot that fits where the things are, and you can choose another
one (Advanced, Robot).

| Robot | Base | When it is used |
|---|---|---|
| `table_arm` | fixed at the front edge of the surface | things on a table (the default, everything before this worked this way) |
| `mobile_arm` | drives in any direction, the arm mounted 12 cm up, 26 x 26 cm, 20 cm/s | things on the floor or on something else |

## Describe your own

```json
{
  "name": "my_robot",
  "about": "One sentence for the dashboard.",
  "base": {"kind": "mobile", "footprint_cm": [40, 30], "mount_height_cm": 20, "speed_cm_s": 25, "turn_deg_s": 90},
  "arm": "so_arm100",
  "arm_size": 1.5
}
```

`base.kind` is `fixed` (no other fields) or `mobile`. A file anywhere else works too: pass its path where a robot name is
asked for.

## How a mobile robot works

- **Drive, stop, pick.** The base only moves between arm moves. When the thing, or later the place, is out of the arm's reach
  from where it stands, the skill first drives to a free spot facing it (`World.standoff`: at the middle of the arm's reach,
  outside the base's own footprint, clear of the other things and the furniture), then the arm works as on a fixed base. So
  every skill, the practice, the imagining and the learning work the same on both kinds.
- The base is three joints (along x, along y, turning) held by position servos; it drives at its speed and turns at its rate.
  It does not collide with the objects or the surface: the planner keeps it clear of them.
- It starts just in front of the mapped area. "Out of my reach" never happens on wheels; it says "there is no free spot for me
  to stand next to it" instead, when things or furniture leave no room.
- In the arm size card, a mobile robot needs only the jaw width: it drives up to the thing.

Checked in the simulation: on a floor of 2 x 1.6 m the mobile robot drives to a cube 1.2 m away, picks it up, drives to a
place on the other side and puts it down 0.6 cm from the spot (`tests/robot/test_robots.py`). The table-top arm gives exactly
the same demos, messages and object positions as before (a scripted session compared before and after).

## Not done yet

- Driving by hand from the phone (teaching on the floor uses the arm from where the base stands).
- A two-wheel base that cannot slide sideways (would need path planning with turns).
- Things on two heights at once (a table and the floor): one surface per scene.
- A real robot: nothing here has run on hardware. LeRobot's LeKiwi (an SO-100/101 arm on a three-wheel base) is the closest
  real match to `mobile_arm`.
