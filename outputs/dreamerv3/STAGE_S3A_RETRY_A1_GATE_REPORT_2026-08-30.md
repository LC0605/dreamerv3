# Stage S3-A-Retry A1 Gate Report (2026-08-30)

## Decision

**STOP. A1 did not pass the Gate. Do not enter A2 from this branch.**

Parent (permanently retained):

`stageS2_Conservative_6000/ckpt/20260828T154503F849521`

A1 final checkpoint:

`stageS3A_Retry_A1_4000/ckpt/20260830T121605F166990`

The random A1 success rate was 78%, below the required 80%. Fixed +0.3
success was 72%, a 6 percentage-point regression from Conservative-2k and
therefore beyond the allowed 5 pp. Collision remained below 15% on the A1
random set, but all Gate conditions must pass simultaneously.

## Training

- 4000 full world-model/value/actor updates from Conservative-2k.
- Continuous new-scene lateral range `[+0.3, +0.8]` metres.
- 60% new A1 replay without policy anchor and 40% old fixed replay with
  near-obstacle anchor scale 5.0.
- `bc_scale=0`, PPO disabled, three-frame observation, unchanged reward,
  observation, and action interfaces.
- No intermediate policy probes and no hyperparameter changes.
- No NaN, loss explosion, or interface error.

## Deterministic evaluation (100 completed episodes each)

| Scene | S/C/T/OOB | Mean clearance | Navigation time | Success change vs parent |
|---|---:|---:|---:|---:|
| A1 random `[+0.3,+0.8]`, seed 91000 | 78/5/17/0 | 0.336 m | 12.584 s | n/a |
| Fixed +0.8, seed 70000 | 92/0/8/0 | 0.523 m | 11.959 s | -2 pp (94% -> 92%) |
| Fixed +0.6, seed 71000 | 90/2/8/0 | 0.393 m | 11.585 s | -2 pp (92% -> 90%) |
| Fixed +0.3, seed 72000 | 72/17/11/0 | 0.231 m | 10.320 s | -6 pp (78% -> 72%) |

Clearance is the mean of each episode's minimum obstacle clearance.

## Near-obstacle actions (clearance < 0.70 m)

| Scene | States | Forward | World lateral | Route lateral |
|---|---:|---:|---:|---:|
| A1 random | 3150 | +0.0276 | +0.0708 | -0.1621 |
| Fixed +0.8 | 1561 | +0.0515 | +0.0630 | -0.1973 |
| Fixed +0.6 | 2487 | -0.0068 | +0.0350 | -0.2093 |
| Fixed +0.3 | 4048 | -0.0388 | -0.0183 | -0.1322 |

## Actor drift relative to Conservative-2k

Drift was measured by replaying the same recorded histories through the parent
and A1 deterministic policies and comparing their actions.

| State set | States | Action MAE | Action RMSE | Forward delta | Lateral delta | Absolute lateral delta |
|---|---:|---:|---:|---:|---:|---:|
| All four evaluations | 46457 | 0.03333 | 0.04490 | +0.00482 | -0.00379 | -0.00789 |
| Near obstacle | 11255 | 0.02752 | 0.04054 | +0.00339 | +0.00091 | +0.00254 |

The near-obstacle comparison does not show the prohibited combination of a
large forward increase and lateral collapse. The mandatory stop is caused by
the objective Gate failures above.

## Artifacts

- Training: `outputs/dreamerv3/stageS3A_Retry_A1_4000`
- Replay audit: `outputs/dreamerv3/stageS3A_Retry_A1_parent_replay/manifest.json`
- Joint replay: `outputs/dreamerv3/stageS3A_Retry_A1_joint_replay`
- Evaluations: `outputs/dreamerv3/eval_S3A_Retry_A1_Final_*`
- Actor drift: `outputs/dreamerv3/diagnostics/S3A_Retry_A1_actor_drift_vs_Conservative2k.json`
- Reproduction scripts: `scripts/train_stage_s3a_retry_a1_4000.sh`,
  `scripts/evaluate_stage_s3a_retry_a1_gate.sh`
