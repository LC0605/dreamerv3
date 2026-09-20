#!/usr/bin/env bash
set -euo pipefail
cd /home/user/UAV_Dreamer
export PYTHONPATH=/home/user/UAV_Dreamer/src:/home/user/UAV_Dreamer/dreamerv3
export JAX_COMPILATION_CACHE_DIR=/home/user/UAV_Dreamer/outputs/dreamerv3/jax_compilation_cache

exec /home/user/miniconda3/envs/dreamer_uav/bin/python dreamerv3/dreamerv3/main.py \
  --logdir outputs/dreamerv3/diagnostics/tiny_head_overfit_run_b2 \
  --configs quadrotor --script bc_distill \
  --run.from_checkpoint outputs/dreamerv3/stageS1fB5_temporal3_online_pure_short/ckpt/20260825T223426F313624 \
  --run.from_checkpoint_regex '^(?!(bcopt|opt|slowval|slowval_count|retnorm|valnorm|advnorm)/).*' \
  --run.bc_dataset outputs/dreamerv3/diagnostics/tiny_head_overfit_dataset \
  --run.bc_updates 500 --run.bc_validate_every 100 \
  --run.bc_fixed_train False --run.bc_risk_stratified False \
  --batch_size 2 --batch_length 301 --report_length 301 --replay_context 0 \
  --agent.bc_only False --agent.bc_scale 0 \
  --agent.freeze_encoder True --agent.freeze_dynamics True \
  --agent.freeze_decoder True --agent.freeze_reward False \
  --agent.freeze_continue False --agent.freeze_policy True \
  --agent.freeze_value True --agent.freeze_world_model False \
  --agent.loss_scales.dyn 0 --agent.loss_scales.rep 0 --agent.loss_scales.rec 0 \
  --agent.loss_scales.policy 0 --agent.loss_scales.value 0 \
  --agent.loss_scales.repval 0 --agent.loss_scales.rew 1 --agent.loss_scales.con 1 \
  --agent.opt.lr 0.0001 --agent.opt.warmup 0 --agent.imag_length 15 \
  --agent.report False --jax.platform cpu --jax.prealloc False --jax.precompile True \
  --logger.outputs jsonl --logger.filter 'loss/(rew|con)|model/valid_fraction|opt/' \
  --env.quadrotor.max_steps 300 --env.quadrotor.temporal_history 3 \
  --env.quadrotor.arena_x 16 --env.quadrotor.arena_y 16 --env.quadrotor.arena_z 5 \
  --env.quadrotor.goal_distance_low 2 --env.quadrotor.goal_distance_high 4 \
  --env.quadrotor.horizontal_speed_limit 0.5 --env.quadrotor.obstacle_count 1 \
  --env.quadrotor.obstacle_lateral_offset 0.3 \
  --env.quadrotor.obstacle_randomization_level 0 --env.quadrotor.boundary_margin 3 \
  --env.quadrotor.boundary_guard_distance 0 --env.quadrotor.safety_projection False
