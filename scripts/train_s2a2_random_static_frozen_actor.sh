#!/usr/bin/env bash
set -euo pipefail
cd /home/user/UAV_Dreamer
export PYTHONPATH=/home/user/UAV_Dreamer/src
export JAX_COMPILATION_CACHE_DIR=/home/user/UAV_Dreamer/outputs/dreamerv3/jax_compilation_cache

latest=$(tr -d '\n' < outputs/dreamerv3/stageS1fB5_temporal3_online_pure_short/ckpt/latest)
source_checkpoint="outputs/dreamerv3/stageS1fB5_temporal3_online_pure_short/ckpt/$latest"

exec /home/user/miniconda3/envs/dreamer_uav/bin/python dreamerv3/dreamerv3/main.py \
  --logdir outputs/dreamerv3/stageS2a2_temporal3_random_static_frozen_actor \
  --configs quadrotor --script train \
  --run.from_checkpoint "$source_checkpoint" --run.from_checkpoint_regex '^(?!opt/|bcopt/).*' \
  --run.steps 2000 --run.envs 1 --run.train_ratio 8 \
  --run.log_every 100 --run.report_every 1000 --run.save_every 20 \
  --batch_size 1 --batch_length 32 --report_length 32 --replay_context 0 \
  --agent.freeze_policy True --agent.loss_scales.policy 0 \
  --agent.bc_only False --agent.bc_scale 0 --agent.report False \
  --agent.opt.lr 0.000001 --agent.opt.warmup 0 \
  --agent.imag_loss.actent 0.00001 --agent.policy.minstd 0.03 \
  --jax.platform cpu --jax.prealloc False --jax.precompile True \
  --logger.outputs jsonl --logger.filter 'score|length|loss/|opt/|epstats/log/' \
  --env.quadrotor.seed 35200 --env.quadrotor.max_steps 300 \
  --env.quadrotor.temporal_history 3 --env.quadrotor.arena_x 16 \
  --env.quadrotor.arena_y 16 --env.quadrotor.arena_z 5 \
  --env.quadrotor.goal_distance_low 2 --env.quadrotor.goal_distance_high 4 \
  --env.quadrotor.horizontal_speed_limit 0.5 --env.quadrotor.obstacle_count 1 \
  --env.quadrotor.obstacle_spawn_probability 1 --env.quadrotor.obstacle_lateral_offset 0.8 \
  --env.quadrotor.obstacle_randomization_level 1 --env.quadrotor.boundary_margin 3 \
  --env.quadrotor.boundary_guard_distance 0 --env.quadrotor.safety_projection False
