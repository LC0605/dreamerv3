#!/usr/bin/env bash
set -euo pipefail

root=/home/user/UAV_Dreamer
threshold=${1:?threshold required}
eval_steps=${2:-3000}
eval_seed=${3:-0}
stage=${DREAMER_STATIC_STAGE:-stageS1b_temporal3_static_single_focus}
spawn_probability=${DREAMER_STATIC_SPAWN_PROBABILITY:-0.75}
safety_distance=${DREAMER_STATIC_SAFETY_DISTANCE:-0.65}
safety_weight=${DREAMER_STATIC_SAFETY_WEIGHT:-4.0}
ttc_weight=${DREAMER_STATIC_TTC_WEIGHT:-4.0}
policy_minstd=${DREAMER_STATIC_POLICY_MINSTD:-0.03}
entropy_scale=${DREAMER_STATIC_ENTROPY_SCALE:-0.00001}
snapshot_root="$root/outputs/dreamerv3/earlystop_snapshots/$stage"
marker="$snapshot_root/threshold_${threshold}.done"

while [[ ! -f "$marker" ]]; do sleep 20; done
snapshot=$(/usr/bin/python3 - "$snapshot_root" "$threshold" <<'PY'
from pathlib import Path
import re, sys
root, threshold = Path(sys.argv[1]), int(sys.argv[2])
items = []
for path in root.glob('step_*'):
    match = re.match(r'step_(\d+)_', path.name)
    if match and int(match.group(1)) >= threshold and (path / 'done').is_file():
        items.append((int(match.group(1)), str(path)))
print(min(items)[1] if items else '')
PY
)
[[ -n "$snapshot" && -f "$snapshot/done" ]]

evaldir="$root/outputs/dreamerv3/eval_${stage}_milestone_${threshold}"
gif="$root/outputs/dreamerv3/visualization/${stage}_milestone_${threshold}.gif"
mkdir -p "$evaldir" "$(dirname "$gif")"
cd "$root"
export PYTHONPATH="$root/src"
export JAX_COMPILATION_CACHE_DIR="$root/outputs/dreamerv3/jax_compilation_cache"
export DREAMER_GIF_OUTPUT="$gif" DREAMER_GIF_STRIDE=2 DREAMER_GIF_MAX_FRAMES=150

/home/user/miniconda3/envs/dreamer_uav/bin/python dreamerv3/dreamerv3/main.py \
  --logdir "$evaldir" --configs quadrotor --script eval_only \
  --run.from_checkpoint "$snapshot" --run.steps "$eval_steps" --run.envs 1 \
  --run.debug True --run.log_every 300 --jax.platform cpu --jax.prealloc False \
  --logger.outputs jsonl --env.quadrotor.seed "$eval_seed" --env.quadrotor.max_steps 300 \
  --env.quadrotor.arena_x 16.0 --env.quadrotor.arena_y 16.0 --env.quadrotor.arena_z 5.0 \
  --env.quadrotor.boundary_margin 3.0 --env.quadrotor.boundary_guard_distance 0.0 \
  --env.quadrotor.goal_distance_low 2.0 --env.quadrotor.goal_distance_high 4.0 \
  --env.quadrotor.horizontal_speed_limit 0.5 --env.quadrotor.temporal_history 3 \
  --env.quadrotor.obstacle_count 1 --env.quadrotor.obstacle_spawn_probability "$spawn_probability" \
  --env.quadrotor.obstacle_lateral_offset 0.8 --env.quadrotor.obstacle_randomization_level 0 \
  --env.quadrotor.dynamic_obstacles False --env.quadrotor.obstacle_layout corridor \
  --env.quadrotor.safety_distance "$safety_distance" --env.quadrotor.safety_weight "$safety_weight" \
  --env.quadrotor.ttc_reward_weight "$ttc_weight" --env.quadrotor.safety_projection False \
  --agent.policy.minstd "$policy_minstd" --agent.imag_loss.actent "$entropy_scale"

/usr/bin/python3 - "$evaldir/metrics.jsonl" "$evaldir/evaluation_summary.json" <<'PY'
import json, sys
source, destination = sys.argv[1:]
totals = dict(episodes=0.0, successes=0.0, collisions=0.0, out_of_bounds=0.0,
              timeouts=0.0, obstacle_episodes=0.0, obstacle_successes=0.0,
              obstacle_collisions=0.0, obstacle_timeouts=0.0)
for line in open(source):
    row = json.loads(line); count = float(row.get('epstats/episode_count', 0.0))
    totals['episodes'] += count
    for name, key in (('successes','success'),('collisions','collision'),
        ('out_of_bounds','out_of_bounds'),('timeouts','timeout'),
        ('obstacle_episodes','obstacle_episode'),('obstacle_successes','success_with_obstacle'),
        ('obstacle_collisions','collision_with_obstacle'),('obstacle_timeouts','timeout_with_obstacle')):
        totals[name] += count * float(row.get(f'epstats/log/{key}/sum', 0.0))
summary = {key: int(round(value)) for key, value in totals.items()}
episodes=max(totals['episodes'],1.0); obstacle=max(totals['obstacle_episodes'],1.0)
for name in ('successes','collisions','out_of_bounds','timeouts'):
    key=name.replace('successes','success').replace('collisions','collision').replace('timeouts','timeout')+'_rate'
    summary[key]=totals[name]/episodes
summary['obstacle_success_rate']=totals['obstacle_successes']/obstacle
summary['obstacle_collision_rate']=totals['obstacle_collisions']/obstacle
summary['obstacle_timeout_rate']=totals['obstacle_timeouts']/obstacle
with open(destination,'w') as stream: json.dump(summary,stream,indent=2); stream.write('\n')
print(json.dumps(summary))
PY
