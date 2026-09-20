#!/usr/bin/env bash
set -u

ROOT=/home/user/UAV_Dreamer
STATUS="$ROOT/outputs/dreamerv3/curriculum_status.json"
LOG="$ROOT/outputs/dreamerv3/curriculum_watchdog.log"

# Do not disturb the currently healthy curriculum session. If it exits before
# the full curriculum is complete, reload the latest runner code and resume
# from immutable summaries/checkpoints. This also recovers ordinary crashes.
while tmux has-session -t dreamer_curriculum 2>/dev/null; do
  sleep 30
done

while true; do
  state=""
  if [[ -f "$STATUS" ]]; then
    state=$(/usr/bin/jq -r '.state // ""' "$STATUS" 2>/dev/null || true)
  fi
  if [[ "$state" == "training_and_generalization_complete" ]]; then
    exit 0
  fi
  printf '%s restarting curriculum after state=%s\n' "$(date '+%F %T')" "$state" >> "$LOG"
  cd "$ROOT" || exit 1
  PYTHONPATH="$ROOT/src" /home/user/miniconda3/envs/dreamer_uav/bin/python \
    scripts/continue_dreamer_curriculum.py >> "$ROOT/outputs/dreamerv3/curriculum_runner.log" 2>&1
  sleep 15
done
