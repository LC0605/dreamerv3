#!/usr/bin/env bash
set -eu

root=/home/user/UAV_Dreamer
threshold=${1:-10000}
stage=${2:-stageT1b_temporal3_short_goal_explore}
boundary_margin=${3:-1.0}
policy_minstd=${4:-0.1}
entropy_scale=${5:-0.0003}
tag=${6:-}
eval_steps=${7:-3000}
eval_seed=${8:-0}
snapshot_root="$root/outputs/dreamerv3/earlystop_snapshots/$stage"
marker="$snapshot_root/threshold_${threshold}.done"

while [[ ! -f "$marker" ]]; do
  sleep 20
done

snapshot=$(find "$snapshot_root" -maxdepth 1 -type d -name "step_${threshold}*" -print -quit)
if [[ -z "$snapshot" ]]; then
  # The watcher can observe a few steps beyond the exact threshold. Select
  # the earliest complete snapshot at or above it; selecting the newest one
  # makes retrospective milestone evaluation silently use a later policy.
  snapshot=$(/usr/bin/python3 - "$snapshot_root" "$threshold" <<'PY'
from pathlib import Path
import re
import sys

root, threshold = Path(sys.argv[1]), int(sys.argv[2])
candidates = []
for path in root.glob('step_*'):
    match = re.match(r'step_(\d+)_', path.name)
    if match and int(match.group(1)) >= threshold and (path / 'done').is_file():
        candidates.append((int(match.group(1)), str(path)))
print(min(candidates)[1] if candidates else '')
PY
  )
fi
if [[ -z "$snapshot" || ! -f "$snapshot/done" ]]; then
  printf '%s no complete snapshot found for threshold %s\n' "$(date '+%F %T')" "$threshold"
  exit 1
fi

suffix=""
if [[ -n "$tag" ]]; then
  suffix="_${tag}"
fi
evaldir="$root/outputs/dreamerv3/eval_${stage}_milestone_${threshold}${suffix}"
gif="$root/outputs/dreamerv3/visualization/${stage}_milestone_${threshold}${suffix}.gif"
mkdir -p "$evaldir" "$(dirname "$gif")"

cd "$root"
export PYTHONPATH="$root/src"
export JAX_COMPILATION_CACHE_DIR="$root/outputs/dreamerv3/jax_compilation_cache"
export DREAMER_GIF_OUTPUT="$gif"
export DREAMER_GIF_STRIDE=2
export DREAMER_GIF_MAX_FRAMES=150

/home/user/miniconda3/envs/dreamer_uav/bin/python dreamerv3/dreamerv3/main.py \
  --logdir "$evaldir" --configs quadrotor --script eval_only \
  --run.from_checkpoint "$snapshot" --run.steps "$eval_steps" --run.envs 1 \
  --run.debug True --run.log_every 300 --jax.platform cpu --jax.prealloc False \
  --logger.outputs jsonl --env.quadrotor.max_steps 300 \
  --env.quadrotor.seed "$eval_seed" \
  --env.quadrotor.arena_x 16.0 --env.quadrotor.arena_y 16.0 \
  --env.quadrotor.arena_z 5.0 --env.quadrotor.boundary_margin "$boundary_margin" \
  --env.quadrotor.goal_distance_low 1.0 \
  --env.quadrotor.goal_distance_high 2.0 \
  --env.quadrotor.horizontal_speed_limit 0.4 \
  --env.quadrotor.temporal_history 3 \
  --agent.policy.minstd "$policy_minstd" \
  --agent.imag_loss.actent "$entropy_scale"

/usr/bin/python3 - "$evaldir/metrics.jsonl" "$evaldir/evaluation_summary.json" <<'PY'
import json
import sys

source, destination = sys.argv[1:]
episodes = successes = collisions = out_of_bounds = timeouts = 0.0
for line in open(source):
    row = json.loads(line)
    count = float(row.get('epstats/episode_count', 0.0))
    episodes += count
    successes += count * float(row.get('epstats/log/success/sum', 0.0))
    collisions += count * float(row.get('epstats/log/collision/sum', 0.0))
    out_of_bounds += count * float(row.get('epstats/log/out_of_bounds/sum', 0.0))
    timeouts += count * float(row.get('epstats/log/timeout/sum', 0.0))
summary = {
    'episodes': int(round(episodes)),
    'successes': int(round(successes)),
    'collisions': int(round(collisions)),
    'out_of_bounds': int(round(out_of_bounds)),
    'timeouts': int(round(timeouts)),
    'success_rate': successes / max(episodes, 1.0),
    'collision_rate': collisions / max(episodes, 1.0),
    'out_of_bounds_rate': out_of_bounds / max(episodes, 1.0),
    'timeout_rate': timeouts / max(episodes, 1.0),
}
with open(destination, 'w') as stream:
    json.dump(summary, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps(summary, ensure_ascii=False))
PY
