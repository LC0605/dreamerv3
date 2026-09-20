#!/usr/bin/env bash
set -euo pipefail
cd /home/user/UAV_Dreamer
export PYTHONPATH=/home/user/UAV_Dreamer/src
export JAX_COMPILATION_CACHE_DIR=/home/user/UAV_Dreamer/outputs/dreamerv3/jax_compilation_cache
exec /home/user/miniconda3/envs/dreamer_uav/bin/python dreamerv3/dreamerv3/main.py \
  --logdir outputs/dreamerv3/stageS2h_fixed_offset0p4_takeover_lowbc \
  --configs quadrotor --script bc_distill \
  --run.from_checkpoint outputs/dreamerv3/stageS2h_fixed_offset0p4_takeover/ckpt/20260826T081443F977285 \
  --run.bc_dataset outputs/dreamerv3/teacher_episodes/stageS2h_ppo_fixed_offset0p4_temporal3_seed36000 \
  --run.bc_updates 25 --run.bc_validate_every 25 --run.bc_fixed_train False \
  --batch_size 1 --batch_length 64 --report_length 64 --replay_context 0 \
  --agent.bc_only False --agent.bc_scale 0.05 --agent.bc_fixed_std 0.3 \
  --agent.opt.lr 0.000002 --agent.opt.warmup 0 \
  --agent.imag_loss.actent 0.00001 --agent.policy.minstd 0.03 \
  --jax.platform cpu --jax.prealloc False --jax.precompile True \
  --logger.outputs jsonl --logger.filter 'bc/|loss/|opt/' \
  --env.quadrotor.max_steps 300 --env.quadrotor.temporal_history 3 \
  --env.quadrotor.arena_x 16 --env.quadrotor.arena_y 16 --env.quadrotor.arena_z 5 \
  --env.quadrotor.goal_distance_low 2 --env.quadrotor.goal_distance_high 4 \
  --env.quadrotor.horizontal_speed_limit 0.5 --env.quadrotor.obstacle_count 1 \
  --env.quadrotor.obstacle_spawn_probability 1 --env.quadrotor.obstacle_lateral_offset 0.4 \
  --env.quadrotor.obstacle_randomization_level 0 --env.quadrotor.boundary_margin 3 \
  --env.quadrotor.boundary_guard_distance 0 --env.quadrotor.safety_projection False
