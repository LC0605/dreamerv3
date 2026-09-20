#!/usr/bin/env bash
set -euo pipefail

cd /home/user/UAV_Dreamer
export PYTHONPATH=/home/user/UAV_Dreamer/src
export JAX_COMPILATION_CACHE_DIR=/home/user/UAV_Dreamer/outputs/dreamerv3/jax_compilation_cache

exec /home/user/miniconda3/envs/dreamer_uav/bin/python dreamerv3/dreamerv3/main.py \
  --logdir outputs/dreamerv3/stageS1_temporal3_static_single_mixed \
  --configs quadrotor \
  --run.from_checkpoint outputs/dreamerv3/earlystop_snapshots/stageT1d_temporal3_navigation_consolidation/step_32750_20260825T085605F934046 \
  --jax.platform cpu --jax.prealloc False --logger.outputs jsonl \
  --env.quadrotor.max_steps 300 \
  --env.quadrotor.arena_x 16.0 --env.quadrotor.arena_y 16.0 --env.quadrotor.arena_z 5.0 \
  --env.quadrotor.boundary_margin 3.0 --env.quadrotor.boundary_guard_distance 0.0 \
  --env.quadrotor.goal_distance_low 2.0 --env.quadrotor.goal_distance_high 4.0 \
  --env.quadrotor.horizontal_speed_limit 0.5 --env.quadrotor.temporal_history 3 \
  --env.quadrotor.obstacle_count 1 \
  --env.quadrotor.obstacle_spawn_probability 0.5 \
  --env.quadrotor.obstacle_lateral_offset 0.8 \
  --env.quadrotor.obstacle_randomization_level 0 \
  --env.quadrotor.dynamic_obstacles False \
  --env.quadrotor.obstacle_layout corridor \
  --env.quadrotor.safety_projection False \
  --agent.policy.minstd 0.05 --agent.imag_loss.actent 0.00003 --agent.opt.lr 0.00001 \
  --run.steps 30000 --run.envs 1 --run.train_ratio 64 \
  --run.log_every 60 --run.report_every 300 --run.save_every 60
