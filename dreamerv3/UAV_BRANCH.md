# UAV DreamerV3 branch status

This directory vendors the upstream `danijar/dreamerv3` implementation at
commit `b65cf81`. The upstream algorithm, RSSM, replay, and training loop are
kept unchanged. Project-specific code is deliberately limited to:

- `embodied/envs/quadrotor.py`: Embodied adapter for `UAVNavigationEnv`.
- `dreamerv3/main.py`: registration of the `quadrotor` environment suite.
- `dreamerv3/configs.yaml`: the small `quadrotor` experiment preset.

## Current maturity

The adapter uses the same PyBullet CF2X dynamics and observation/reward contract
as the validated PPO branch. It replaces the earlier nine-state toy integrator,
which directly assigned velocity and was neither wired into DreamerV3 nor valid
for dynamics experiments.

No trained UAV DreamerV3 checkpoint exists yet. PPO remains the validated
baseline. Do not describe this branch as trained or benchmarked until the smoke
test and staged experiments below have produced saved metrics.

## Environment contract

- Observation key: `vector`, float32, shape determined by the environment.
- Action key: `action`, four physical commands `(vx, vy, vz, yaw_rate)`.
- Decision rate: 10 Hz; physics rate: 240 Hz.
- Termination and truncation remain distinct in `is_terminal` and `is_last`.
- Position and velocity are never assigned by the adapter.

## Staged use

Use Python 3.11+ with the DreamerV3 requirements and install the parent project
as editable. Start with a CPU contract smoke test, then use a CUDA JAX install:

```bash
cd /home/user/UAV_Dreamer/dreamerv3
python dreamerv3/main.py \
  --logdir ../outputs/dreamerv3/smoke/{timestamp} \
  --configs quadrotor debug \
  --run.steps 200 --run.envs 1 --jax.platform cpu
```

Training stages should be separate log directories:

1. `stage0_no_obstacle`: prove world-model and actor learning.
2. `stage1_static`: add randomized static obstacles.
3. `stage2_single_dynamic`: add one slow crossing obstacle.
4. `stage3_multi_dynamic`: match the PPO benchmark contract.

Each stage must save its resolved `config.yaml`, metrics, replay, and checkpoint.
Never reuse a log directory after changing observation shape.

## Retention rules

- Keep upstream source and license intact for provenance.
- Keep one final checkpoint plus the best validation checkpoint per stage.
- Keep resolved configuration and metrics with every retained checkpoint.
- Treat replay buffers and intermediate checkpoints as disposable only after a
  stage is complete and its result is reproduced.
- Store all new artifacts under `outputs/dreamerv3/`; do not mix them with PPO
  `.zip` models or `vec_normalize.pkl` files.
