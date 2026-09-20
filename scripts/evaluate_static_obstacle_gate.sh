#!/usr/bin/env bash
set -euo pipefail

root=/home/user/UAV_Dreamer
snapshot=${1:?complete snapshot directory required}
tag=${2:?evaluation tag required}
eval_steps=${3:-6000}
eval_seed=${4:-0}
obstacle_count=${5:-1}
obstacle_offset=${6:-0.8}
randomization=${7:-0}
checkpoint_regex=${DREAMER_CHECKPOINT_REGEX:-.*}
eval_envs=${DREAMER_EVAL_ENVS:-1}

[[ -f "$snapshot/done" && -f "$snapshot/agent.pkl" ]]
evaldir="$root/outputs/dreamerv3/eval_${tag}"
gif="$root/outputs/dreamerv3/visualization/${tag}.gif"
if [[ -e "$evaldir/metrics.jsonl" || -e "$evaldir/evaluation_summary.json" ]]; then
  echo "Refusing to append to existing evaluation: $evaldir" >&2
  exit 2
fi
mkdir -p "$evaldir" "$(dirname "$gif")"

cd "$root"
export PYTHONPATH="$root/src"
export JAX_COMPILATION_CACHE_DIR="$root/outputs/dreamerv3/jax_compilation_cache"
export DREAMER_GIF_OUTPUT="$gif"
export DREAMER_GIF_STRIDE=2
export DREAMER_GIF_MAX_FRAMES=180
if (( eval_envs > 1 )); then
  export DREAMER_DISTINCT_ENV_SEEDS=1
fi

/home/user/miniconda3/envs/dreamer_uav/bin/python dreamerv3/dreamerv3/main.py \
  --logdir "$evaldir" --configs quadrotor --script eval_only \
  --run.from_checkpoint "$snapshot" --run.steps "$eval_steps" --run.envs "$eval_envs" \
  --run.from_checkpoint_regex "$checkpoint_regex" \
  --run.debug True --run.log_every 300 --jax.platform cpu --jax.prealloc False \
  --logger.outputs jsonl --env.quadrotor.seed "$eval_seed" --env.quadrotor.max_steps 300 \
  --env.quadrotor.arena_x 16.0 --env.quadrotor.arena_y 16.0 --env.quadrotor.arena_z 5.0 \
  --env.quadrotor.boundary_margin 3.0 --env.quadrotor.boundary_guard_distance 0.0 \
  --env.quadrotor.goal_distance_low 2.0 --env.quadrotor.goal_distance_high 4.0 \
  --env.quadrotor.horizontal_speed_limit 0.5 --env.quadrotor.temporal_history 3 \
  --env.quadrotor.obstacle_count "$obstacle_count" \
  --env.quadrotor.obstacle_spawn_probability 1.0 \
  --env.quadrotor.obstacle_lateral_offset "$obstacle_offset" \
  --env.quadrotor.obstacle_randomization_level "$randomization" \
  --env.quadrotor.dynamic_obstacles False --env.quadrotor.obstacle_layout corridor \
  --env.quadrotor.safety_projection False \
  --agent.policy.minstd 0.05 --agent.imag_loss.actent 0.00003

/usr/bin/python3 - "$evaldir/metrics.jsonl" "$evaldir/evaluation_summary.json" <<'PY'
import json
import sys

source, destination = sys.argv[1:]
totals = dict(episodes=0.0, successes=0.0, collisions=0.0,
              out_of_bounds=0.0, timeouts=0.0, lengths=0.0,
              minimum_clearance=0.0, path_length=0.0, action_magnitude=0.0)
totals.update(action_forward=0.0, action_lateral=0.0,
              action_route_lateral=0.0)
for line in open(source):
    row = json.loads(line)
    count = float(row.get('epstats/episode_count', 0.0))
    totals['episodes'] += count
    # episode/length is emitted once per completed episode, whereas epstats are
    # flushed as an aggregate row. Summing it directly preserves every episode
    # and avoids multiplying the final episode by the aggregate count.
    totals['lengths'] += float(row.get('episode/length', 0.0))
    totals['minimum_clearance'] += count * float(
        row.get('epstats/log/minimum_clearance/min', 0.0))
    totals['path_length'] += count * float(
        row.get('epstats/log/path_length/max', 0.0))
    totals['action_magnitude'] += count * float(
        row.get('epstats/log/action_magnitude/avg', 0.0))
    for name in ('action_forward', 'action_lateral', 'action_route_lateral'):
        totals[name] += count * float(row.get(f'epstats/log/{name}/avg', 0.0))
    for name, key in (
        ('successes', 'success'), ('collisions', 'collision'),
        ('out_of_bounds', 'out_of_bounds'), ('timeouts', 'timeout')):
        totals[name] += count * float(row.get(f'epstats/log/{key}/sum', 0.0))
summary = {key: int(round(value)) for key, value in totals.items()}
episodes = max(totals['episodes'], 1.0)
for name in ('successes', 'collisions', 'out_of_bounds', 'timeouts'):
    key = name.replace('successes', 'success').replace(
        'collisions', 'collision').replace('timeouts', 'timeout') + '_rate'
    summary[key] = totals[name] / episodes
summary['mean_episode_length'] = totals['lengths'] / episodes
summary['mean_navigation_time_seconds'] = summary['mean_episode_length'] * 0.1
summary['mean_minimum_clearance'] = totals['minimum_clearance'] / episodes
summary['mean_path_length'] = totals['path_length'] / episodes
summary['mean_action_magnitude'] = totals['action_magnitude'] / episodes
summary['mean_forward_action'] = totals['action_forward'] / episodes
summary['mean_lateral_action'] = totals['action_lateral'] / episodes
summary['mean_route_lateral_action'] = totals['action_route_lateral'] / episodes
with open(destination, 'w') as stream:
    json.dump(summary, stream, indent=2)
    stream.write('\n')
print(json.dumps(summary))
PY
