#!/usr/bin/env bash
set -euo pipefail
cd /home/user/UAV_Dreamer
export PYTHONPATH=/home/user/UAV_Dreamer/src
exec /home/user/miniconda3/envs/dreamer_uav/bin/python dreamerv3/dreamerv3/main.py \
  --logdir outputs/dreamerv3/stageS1fA_actor_overfit_probe \
  --configs quadrotor --script bc_distill \
  --run.from_checkpoint outputs/dreamerv3/earlystop_snapshots/stageS1_temporal3_static_single_mixed/step_10170_20260825T103851F203286 \
  --run.bc_dataset outputs/dreamerv3/teacher_episodes/stage4_ppo_static_offset0p8_temporal3_seed20000 \
  --run.bc_updates 50 --run.bc_validate_every 10 --run.bc_fixed_train True \
  --batch_size 8 --batch_length 301 --report_length 301 --replay_context 0 \
  --agent.bc_only True --agent.bc_scale 1 --agent.opt.lr 0.0001 --agent.opt.warmup 0 \
  --jax.platform cpu --jax.prealloc False --jax.precompile True \
  --logger.outputs jsonl --logger.filter 'bc/|bcopt/' \
  --env.quadrotor.max_steps 300 --env.quadrotor.temporal_history 3 \
  --env.quadrotor.arena_x 16 --env.quadrotor.arena_y 16 --env.quadrotor.arena_z 5 \
  --env.quadrotor.goal_distance_low 2 --env.quadrotor.goal_distance_high 4 \
  --env.quadrotor.horizontal_speed_limit 0.5 --env.quadrotor.obstacle_count 1 \
  --env.quadrotor.obstacle_spawn_probability 1 --env.quadrotor.obstacle_lateral_offset 0.8 \
  --env.quadrotor.obstacle_randomization_level 0 --env.quadrotor.boundary_margin 3 \
  --env.quadrotor.boundary_guard_distance 0 --env.quadrotor.safety_projection False
