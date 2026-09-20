#!/usr/bin/env bash
set -euo pipefail
cd /home/user/UAV_Dreamer
export PYTHONPATH=/home/user/UAV_Dreamer/src
export JAX_COMPILATION_CACHE_DIR=/home/user/UAV_Dreamer/outputs/dreamerv3/jax_compilation_cache
latest=$(tr -d '\n' < outputs/dreamerv3/stageS1fB5_temporal3_online_pure_short/ckpt/latest)
source_checkpoint="outputs/dreamerv3/stageS1fB5_temporal3_online_pure_short/ckpt/$latest"
exec /home/user/miniconda3/envs/dreamer_uav/bin/python dreamerv3/dreamerv3/main.py \
  --logdir outputs/dreamerv3/stageS2e_temporal3_dagger_recovery_actor_bc \
  --configs quadrotor --script bc_distill \
  --run.from_checkpoint "$source_checkpoint" --run.from_checkpoint_regex '^(?!bcopt/).*' \
  --run.bc_dataset outputs/dreamerv3/teacher_episodes/stage5_level1_dagger_noise0p15_success_only \
  --run.bc_updates 300 --run.bc_validate_every 100 --run.bc_fixed_train False \
  --batch_size 8 --batch_length 301 --report_length 301 --replay_context 0 \
  --agent.bc_only True --agent.bc_scale 1 --agent.bc_fixed_std 0.3 \
  --agent.freeze_policy False --agent.freeze_world_model False \
  --agent.opt.lr 0.00005 --agent.opt.warmup 0 \
  --jax.platform cpu --jax.prealloc False --jax.precompile True \
  --logger.outputs jsonl --logger.filter 'bc/|bcopt/' \
  --env.quadrotor.max_steps 300 --env.quadrotor.temporal_history 3 \
  --env.quadrotor.arena_x 16 --env.quadrotor.arena_y 16 --env.quadrotor.arena_z 5 \
  --env.quadrotor.goal_distance_low 2 --env.quadrotor.goal_distance_high 4 \
  --env.quadrotor.horizontal_speed_limit 0.5 --env.quadrotor.obstacle_count 1 \
  --env.quadrotor.obstacle_spawn_probability 1 --env.quadrotor.obstacle_lateral_offset 0.8 \
  --env.quadrotor.obstacle_randomization_level 1 --env.quadrotor.boundary_margin 3 \
  --env.quadrotor.boundary_guard_distance 0 --env.quadrotor.safety_projection False
