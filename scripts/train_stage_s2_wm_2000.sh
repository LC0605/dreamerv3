#!/usr/bin/env bash
set -euo pipefail
cd /home/user/UAV_Dreamer
export PYTHONPATH=/home/user/UAV_Dreamer/src:/home/user/UAV_Dreamer/dreamerv3
export JAX_COMPILATION_CACHE_DIR=/home/user/UAV_Dreamer/.jax_cache

exec /home/user/miniconda3/envs/dreamer_uav/bin/python dreamerv3/dreamerv3/main.py \
  --logdir outputs/dreamerv3/stageS2_WM_2000 \
  --configs quadrotor --script bc_distill \
  --run.from_checkpoint outputs/dreamerv3/stageS1fB5_temporal3_online_pure_short/ckpt/20260825T223426F313624 \
  --run.from_checkpoint_regex '^(enc|dyn|dec|rew|con|pol|val|slowval)/' \
  --run.bc_dataset outputs/dreamerv3/stageS2_WM_offline_mixed \
  --run.bc_updates 2000 --run.bc_validate_every 2000 \
  --run.bc_fixed_train False --run.bc_risk_stratified True \
  --batch_size 2 --batch_length 301 --report_length 301 --replay_context 0 \
  --agent.reward_decomposition True \
  --agent.outcome_mode conditional_terminal \
  --agent.bc_only False --agent.bc_scale 0 --agent.bc_fixed_std 0 \
  --agent.freeze_encoder False --agent.freeze_dynamics False \
  --agent.freeze_decoder False --agent.freeze_reward False \
  --agent.freeze_continue False --agent.freeze_outcome False \
  --agent.freeze_policy True --agent.freeze_value True \
  --agent.freeze_world_model False \
  --agent.loss_scales.dyn 1 --agent.loss_scales.rep 0.1 \
  --agent.loss_scales.rec 1 --agent.loss_scales.policy 0 \
  --agent.loss_scales.value 0 --agent.loss_scales.repval 0 \
  --agent.loss_scales.rew 1 --agent.loss_scales.outcome 1 \
  --agent.loss_scales.con 1 --agent.reward_strata_weights 0 0 0 \
  --agent.opt.lr 0.000002 --agent.opt.warmup 0 \
  --agent.imag_length 15 --agent.report False \
  --jax.platform cpu --jax.prealloc False --jax.precompile True \
  --logger.outputs jsonl \
  --logger.filter 'loss/(dyn|rep|vector|rew|outcome|con)|outcome/|opt/' \
  --env.quadrotor.max_steps 300 --env.quadrotor.temporal_history 3 \
  --env.quadrotor.arena_x 16 --env.quadrotor.arena_y 16 --env.quadrotor.arena_z 5 \
  --env.quadrotor.goal_distance_low 2 --env.quadrotor.goal_distance_high 4 \
  --env.quadrotor.horizontal_speed_limit 0.5 --env.quadrotor.obstacle_count 1 \
  --env.quadrotor.obstacle_lateral_offset 0.3 \
  --env.quadrotor.obstacle_randomization_level 0 \
  --env.quadrotor.boundary_margin 3 --env.quadrotor.boundary_guard_distance 0 \
  --env.quadrotor.safety_projection False
