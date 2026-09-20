#!/usr/bin/env bash
set -euo pipefail
cd /home/user/UAV_Dreamer
export PYTHONPATH=/home/user/UAV_Dreamer/src:/home/user/UAV_Dreamer/dreamerv3
export JAX_COMPILATION_CACHE_DIR=/home/user/UAV_Dreamer/.jax_cache

mixed=outputs/dreamerv3/stageS1fB2_temporal3_teacher_takeover_midbc/ckpt/20260825T222659F216085
cons=outputs/dreamerv3/stageS2_Conservative_6000/ckpt/20260828T154503F849521
for model_ckpt in "Mixed:$mixed" "Conservative2k:$cons"; do
  model=${model_ckpt%%:*}; checkpoint=${model_ckpt#*:}
  for spec in 0.8:70000 0.6:71000 0.3:72000; do
    offset=${spec%%:*}; seed=${spec##*:}; tag=${offset/./p}
    logdir=outputs/dreamerv3/eval_${model}_Paired_offset_${tag}
    DREAMER_EVAL_EPISODES=100 /home/user/miniconda3/envs/dreamer_uav/bin/python \
      dreamerv3/dreamerv3/main.py --logdir "$logdir" --configs quadrotor \
      --script eval_only --run.from_checkpoint "$checkpoint" \
      --run.from_checkpoint_regex '^(enc|dyn|dec|rew|con|pol|val|slowval|retnorm|valnorm|advnorm)/' \
      --run.steps 100000 --run.envs 1 --run.debug True --run.log_every 300 \
      --agent.reward_decomposition False --agent.anchor_scale 0 \
      --agent.separate_actor_opt False --jax.platform cpu --jax.prealloc False \
      --logger.outputs jsonl --env.quadrotor.seed "$seed" \
      --env.quadrotor.max_steps 300 --env.quadrotor.temporal_history 3 \
      --env.quadrotor.arena_x 16 --env.quadrotor.arena_y 16 --env.quadrotor.arena_z 5 \
      --env.quadrotor.goal_distance_low 2 --env.quadrotor.goal_distance_high 4 \
      --env.quadrotor.horizontal_speed_limit 0.5 --env.quadrotor.obstacle_count 1 \
      --env.quadrotor.obstacle_lateral_offset "$offset" \
      --env.quadrotor.obstacle_randomization_level 0 \
      --env.quadrotor.boundary_margin 3 --env.quadrotor.boundary_guard_distance 0 \
      --env.quadrotor.safety_projection False
  done
done
