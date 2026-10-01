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
