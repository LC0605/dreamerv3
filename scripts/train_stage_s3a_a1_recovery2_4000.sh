#!/usr/bin/env bash
set -euo pipefail
cd /home/user/UAV_Dreamer
export PYTHONPATH=/home/user/UAV_Dreamer/src:/home/user/UAV_Dreamer/dreamerv3
export JAX_COMPILATION_CACHE_DIR=/home/user/UAV_Dreamer/.jax_cache

parent=outputs/dreamerv3/stageS3A_A1_Recovery_4000/ckpt/20260830T153553F674727
dataset=outputs/dreamerv3/stageS3A_A1_Recovery2_replay
logdir=outputs/dreamerv3/stageS3A_A1_Recovery2_4000
updates=${1:-4000}

exec /home/user/miniconda3/envs/dreamer_uav/bin/python dreamerv3/dreamerv3/main.py \
  --logdir "$logdir" --configs quadrotor --script bc_distill \
  --run.from_checkpoint "$parent" --run.from_checkpoint_regex '.*' \
  --run.bc_dataset "$dataset" --run.bc_updates "$updates" --run.bc_validate_every 1000 \
  --run.bc_candidate_every 500 \
  --run.bc_fixed_train False --run.bc_risk_stratified False \
  --run.bc_scene_balanced False --run.bc_curriculum_new_fraction 0.0 \
  --batch_size 2 --batch_length 301 --report_length 301 --replay_context 0 \
  --agent.reward_decomposition False --agent.anchor_scale 5.0 \
  --agent.anchor_schedule_enabled False --agent.separate_actor_opt True \
  --agent.bc_only False --agent.bc_scale 0 --agent.bc_fixed_std 0 \
  --agent.freeze_encoder False --agent.freeze_dynamics False \
  --agent.freeze_decoder False --agent.freeze_reward False \
  --agent.freeze_continue False --agent.freeze_policy False \
  --agent.freeze_value False --agent.freeze_world_model False \
  --agent.loss_scales.dyn 1 --agent.loss_scales.rep 0.1 \
  --agent.loss_scales.rec 1 --agent.loss_scales.policy 1 \
  --agent.loss_scales.value 1 --agent.loss_scales.repval 0.3 \
  --agent.loss_scales.rew 1 --agent.loss_scales.con 1 \
  --agent.opt.lr 0.000001 --agent.opt.warmup 0 \
  --agent.actor_opt.lr 0.0000003 --agent.actor_opt.warmup 0 \
  --agent.imag_last 16 --agent.imag_length 15 --agent.report False \
  --jax.platform cpu --jax.prealloc False --jax.precompile True \
  --logger.outputs jsonl \
  --logger.filter 'loss/(dyn|rep|vector|rew|con|anchor|policy|value|repval)|anchor/|opt/|actor/' \
  --env.quadrotor.max_steps 300 --env.quadrotor.temporal_history 3 \
  --env.quadrotor.arena_x 16 --env.quadrotor.arena_y 16 --env.quadrotor.arena_z 5 \
  --env.quadrotor.goal_distance_low 2 --env.quadrotor.goal_distance_high 4 \
  --env.quadrotor.horizontal_speed_limit 0.5 --env.quadrotor.obstacle_count 1 \
  --env.quadrotor.obstacle_lateral_offset 0.3 \
  --env.quadrotor.obstacle_randomization_level 1 \
  --env.quadrotor.random_offset_low 0.3 --env.quadrotor.random_offset_high 0.8 \
  --env.quadrotor.dynamic_obstacles False \
  --env.quadrotor.boundary_margin 3 --env.quadrotor.boundary_guard_distance 0 \
  --env.quadrotor.safety_projection False
