#!/usr/bin/env bash
set -eu

if [[ $# -lt 2 ]]; then
  echo "usage: $0 TRAINING_FOLDER STEP [STEP ...]" >&2
  exit 2
fi

folder=$1
shift
root=/home/user/UAV_Dreamer
source_dir="$root/outputs/dreamerv3/$folder"
snapshot_root="$root/outputs/dreamerv3/earlystop_snapshots/$folder"
mkdir -p "$snapshot_root"

for threshold in "$@"; do
  marker="$snapshot_root/threshold_${threshold}.done"
  while [[ ! -f "$marker" ]]; do
    step=0
    if [[ -s "$source_dir/metrics.jsonl" ]]; then
      step=$(/usr/bin/jq -s 'map(.step // 0) | max // 0' \
        "$source_dir/metrics.jsonl" 2>/dev/null || echo 0)
    fi
    if (( step >= threshold )); then
      latest=$(<"$source_dir/ckpt/latest")
      checkpoint="$source_dir/ckpt/$latest"
      if [[ -f "$checkpoint/done" ]]; then
        destination="$snapshot_root/step_${step}_${latest}"
        cp -a "$checkpoint" "$destination"
        printf '%s copied %s at observed_step=%s\n' \
          "$(date '+%F %T')" "$destination" "$step" >> "$snapshot_root/snapshots.log"
        touch "$marker"
        break
      fi
    fi
    if [[ -f "$root/outputs/dreamerv3/eval_${folder}/evaluation_summary.json" ]]; then
      printf '%s stage ended before threshold=%s\n' \
        "$(date '+%F %T')" "$threshold" >> "$snapshot_root/snapshots.log"
      touch "$marker"
      break
    fi
    sleep 20
  done
done
