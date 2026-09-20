# Stage S3-A-Retry Curriculum

## Previous S3-A range audit

- Previous training/evaluation used `obstacle_randomization_level=1`.
- Neither script overrode `random_offset_low/high`.
- The active Quadrotor defaults were therefore `[-1.5, +1.5]` metres.
- This is the full two-sided range and is replaced by the narrower A1 interval.

## A1 locked settings

- Parent: `stageS2_Conservative_6000/ckpt/20260828T154503F849521`
- Lateral random interval: continuous `[+0.3, +0.8]` metres
- Updates: 4000
- New curriculum replay sampling: 60%; old fixed anchored replay: 40%
- `bc_scale=0`, PPO disabled, anchor scale 5.0 on old fixed replay only
- Standard world model, imagination, value, and actor training remain enabled
- Formal evaluation: one 100-episode A1 random set and three 100-episode fixed sets

## A1 replay collection

- Collection seed: 81000 (the permanent hold-out seed 35700 was not used)
- Episodes: 200 (174 success, 10 collision, 16 timeout, 0 OOB)
- Realized initial lateral range from episode traces: `[+0.300120, +0.799202]` metres
- The realized range validates continuous positive-side sampling inside the requested A1 bounds.
