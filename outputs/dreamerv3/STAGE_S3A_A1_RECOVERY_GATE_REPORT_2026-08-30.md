# Stage S3-A A1-Recovery Gate Report (2026-08-30)

## Decision

**Recovery did not pass the full Gate. A2 was not started.**

The fixed +0.3 recovery objective passed, and collision remained low on the
full random interval. However, random success was 79% rather than 82%, fixed
+0.8 was 88% (6 pp below Conservative-2k), and the paired low-band improvement
over A1-final was only 2.7 pp. This is a curriculum near miss rather than a
collision or lateral-collapse failure, but it does not authorize automatic A2.

Recovery final checkpoint:

`stageS3A_A1_Recovery_4000/ckpt/20260830T153553F674727`

## Deterministic evaluation

| Scene | S/C/T/OOB | Mean clearance | Navigation time |
|---|---:|---:|---:|
| Random `[+0.30,+0.80]`, seed 91000 | 79/2/19/0 | 0.352 m | 12.755 s |
| Fixed +0.8, seed 70000 | 88/2/10/0 | 0.537 m | 11.751 s |
| Fixed +0.6, seed 71000 | 89/2/9/0 | 0.381 m | 10.763 s |
| Fixed +0.3, seed 72000 | 79/12/9/0 | 0.224 m | 9.886 s |
| Low band `[+0.30,+0.45]`, seed 92000 | 79/9/12/0 | 0.266 m | 10.593 s |

Clearance is the mean of each episode's minimum obstacle clearance.

## Near-obstacle actions (clearance < 0.70 m)

| Scene | States | Forward | World lateral | Route lateral |
|---|---:|---:|---:|---:|
| Random | 3134 | +0.0440 | +0.0778 | -0.1648 |
| Fixed +0.8 | 1482 | +0.0494 | +0.0289 | -0.2109 |
| Fixed +0.6 | 2541 | +0.0086 | +0.0252 | -0.1994 |
| Fixed +0.3 | 4089 | -0.0237 | -0.0166 | -0.1313 |
| Low band | 4136 | -0.0024 | -0.0002 | -0.1473 |

Compared with A1-final on the same full-random seed, near-obstacle forward
changed from +0.0276 to +0.0440 while absolute route-lateral magnitude changed
from 0.1621 to 0.1648 and clearance improved from 0.336 m to 0.352 m. Thus the
prohibited `forward up + lateral down + clearance down` combination did not
occur.

## Regression and recovery

| Fixed scene | Conservative-2k | A1-final | Recovery | Change vs Conservative-2k |
|---|---:|---:|---:|---:|
| +0.8 | 94% | 92% | 88% | -6 pp |
| +0.6 | 92% | 90% | 89% | -3 pp |
| +0.3 | 78% | 72% | 79% | +1 pp |

Fixed +0.3 recovered to 79% success and 12% collision, passing its required
`success >= 76%` and `collision <= 15%` thresholds.

Within the paired full-random evaluation, the 37 episodes whose sampled offset
fell in `[+0.30,+0.45]` changed from A1-final `28/2/7` (75.7/5.4/18.9%) to
Recovery `29/2/6` (78.4/5.4/16.2%). This is +2.7 pp success and -2.7 pp timeout,
with unchanged collision. It is positive but not a large low-band improvement.

## Actor drift on identical recorded histories

| Reference | State set | States | Action MAE | RMSE | Forward delta | Absolute lateral delta |
|---|---|---:|---:|---:|---:|---:|
| Conservative-2k | All | 55775 | 0.04273 | 0.05831 | +0.00401 | -0.01308 |
| Conservative-2k | Near obstacle | 15409 | 0.03603 | 0.05313 | +0.00359 | +0.00357 |
| A1-final | All | 55775 | 0.02426 | 0.03368 | +0.00032 | -0.00592 |
| A1-final | Near obstacle | 15409 | 0.02206 | 0.03375 | -0.00019 | +0.00221 |

## Gate

- Random success >=82%: **FAIL (79%)**
- Random collision <10%: **PASS (2%)**
- Fixed +0.3 success >=76%: **PASS (79%)**
- Fixed +0.3 collision <=15%: **PASS (12%)**
- Fixed +0.8/+0.6 regression <=5 pp: **FAIL at +0.8 (-6 pp)**
- Low-band performance clearly improves over A1: **MARGINAL, not clear (+2.7 pp)**
- No forward/lateral/clearance danger pattern: **PASS**

## Artifacts

- Training: `outputs/dreamerv3/stageS3A_A1_Recovery_4000`
- Five evaluations: `outputs/dreamerv3/eval_S3A_A1_Recovery_Final_*`
- Drift vs Conservative-2k:
  `outputs/dreamerv3/diagnostics/S3A_A1_Recovery_actor_drift_vs_Conservative2k.json`
- Drift vs A1-final:
  `outputs/dreamerv3/diagnostics/S3A_A1_Recovery_actor_drift_vs_A1_Final.json`
