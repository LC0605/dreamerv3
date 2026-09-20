# Stage S3-A Gate Report (2026-08-30)

## Decision

**STOP. S3-A did not pass the intermediate gate and triggered the mandatory
old-scene regression stop condition. Do not advance to S3-B from this branch.**

Parent (permanently retained):

`stageS2_Conservative_6000/ckpt/20260828T154503F849521`

S3-A final checkpoint:

`stageS3A_random_lateral_5000/ckpt/20260830T005532F686183`

## Training

- 5000 full WM/Value/Actor updates, parent optimizer state restored.
- 60% new random-lateral single-static episodes, with no policy anchor.
- 40% old fixed-scene episodes, with near-obstacle policy anchor scale 5.0.
- `bc_scale=0`, PPO=0, three-frame observation, standard reward/Continue.
- No NaN, Inf, loss explosion, or interface error.
- Final optimizer update count: 6999 (parent 1999 + S3-A 5000).

## Deterministic evaluation (100 complete episodes each)

| Scene | S/C/T/OOB | Mean minimum clearance | Mean navigation time |
|---|---:|---:|---:|
| Random lateral, seed 90000 | 56/37/7/0 | 0.355 m | 8.164 s |
| Fixed +0.8, seed 70000 | 91/0/9/0 | 0.550 m | 11.556 s |
| Fixed +0.6, seed 71000 | 86/1/13/0 | 0.403 m | 12.656 s |
| Fixed +0.3, seed 72000 | 77/13/10/0 | 0.230 m | 10.245 s |

Conservative-2k fixed baselines were +0.8 `94/1/5/0`, +0.6 `92/1/7/0`,
and +0.3 `78/14/8/0`. The +0.6 success regression is 6 percentage points,
exceeding the allowed 5 pp. Random success is below 80% and collision is far
above 15%.

## Near-obstacle action means (clearance < 0.70 m)

| Scene | Forward | World lateral | Route lateral |
|---|---:|---:|---:|
| Random lateral | +0.0210 | -0.0284 | -0.1589 |
| Fixed +0.8 | +0.0846 | +0.0730 | -0.2209 |
| Fixed +0.6 | +0.0191 | +0.0339 | -0.2230 |
| Fixed +0.3 | -0.0361 | -0.0122 | -0.1408 |

## Artifacts

- Training: `outputs/dreamerv3/stageS3A_random_lateral_5000`
- Random evaluation: `outputs/dreamerv3/eval_S3A_Final_random`
- Fixed evaluations: `outputs/dreamerv3/eval_S3A_Final_fixed0p8`,
  `eval_S3A_Final_fixed0p6`, `eval_S3A_Final_fixed0p3`
- Reproduction scripts: `scripts/train_stage_s3a_random_lateral_5000.sh`,
  `scripts/evaluate_stage_s3a_gate.sh`

