#!/usr/bin/env bash
set -euo pipefail
cd /home/user/UAV_Dreamer
export PYTHONPATH=/home/user/UAV_Dreamer/src:/home/user/UAV_Dreamer/dreamerv3
export JAX_COMPILATION_CACHE_DIR=/home/user/UAV_Dreamer/.jax_cache

checkpoint=${1:?checkpoint is required}
label=${2:-S3A_Retry_A1}
for spec in randomA1:91000:0.0:1 fixed0p8:70000:0.8:0 fixed0p6:71000:0.6:0 fixed0p3:72000:0.3:0; do
  IFS=: read -r scene seed offset randomization <<< "$spec"
  logdir="outputs/dreamerv3/eval_${label}_${scene}"
  if [[ -e "$logdir" ]]; then
    echo "Refusing to overwrite existing evaluation directory: $logdir" >&2
    exit 1
  fi
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
    --env.quadrotor.obstacle_spawn_probability 1 \
    --env.quadrotor.obstacle_lateral_offset "$offset" \
    --env.quadrotor.obstacle_randomization_level "$randomization" \
    --env.quadrotor.random_offset_low 0.3 --env.quadrotor.random_offset_high 0.8 \
    --env.quadrotor.dynamic_obstacles False \
    --env.quadrotor.boundary_margin 3 --env.quadrotor.boundary_guard_distance 0 \
    --env.quadrotor.safety_projection False
done
