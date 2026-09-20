#!/usr/bin/env bash
set -euo pipefail
cd /home/user/UAV_Dreamer
export PYTHONPATH=/home/user/UAV_Dreamer/src:/home/user/UAV_Dreamer/dreamerv3

parent=outputs/dreamerv3/stageS3A_A1_Recovery_4000/ckpt/20260830T153553F674727
new=outputs/dreamerv3/stageS3A_A1_Recovery2_new_policy_replay
joint=outputs/dreamerv3/stageS3A_A1_Recovery2_replay
train=outputs/dreamerv3/stageS3A_A1_Recovery2_4000

if [[ ! -e "$new" ]]; then
  /home/user/miniconda3/envs/dreamer_uav/bin/python scripts/collect_dreamer_recovery2_episodes.py \
    --config outputs/dreamerv3/stageS3A_A1_Recovery_4000/config.yaml \
    --checkpoint "$parent" --output "$new" --seed 93000
fi
if [[ ! -e "$joint" ]]; then
  /home/user/miniconda3/envs/dreamer_uav/bin/python scripts/build_recovery2_replay.py \
    --new "$new" --old outputs/dreamerv3/stageS3A_A1_Recovery_replay --output "$joint"
fi

for updates in 1000 2000 3000 4000; do
  scripts/train_stage_s3a_a1_recovery2_4000.sh "$updates"
  checkpoint="$train/candidates/update_$(printf '%09d' "$updates")"
  label="S3A_A1_Recovery2_U${updates}_Quick"
  scripts/evaluate_stage_s3a_recovery2.sh "$checkpoint" "$label" 20
  selection="outputs/dreamerv3/checkpoint_selection/recovery2/${label}"
  /home/user/miniconda3/envs/dreamer_uav/bin/python scripts/checkpoint_selector.py \
    --checkpoint "$checkpoint" --update "$updates" \
    --candidate-prefix "outputs/dreamerv3/eval_${label}" \
    --parent-prefix outputs/dreamerv3/eval_S3A_A1_Recovery_Final \
    --parent-eval random=outputs/dreamerv3/eval_S3A_A1_Recovery_Final_randomA1 \
    --parent-checkpoint "$parent" \
    --history outputs/dreamerv3/checkpoint_selection/recovery2/candidate_history.json \
    --output "$selection/checkpoint_metrics.json" --top-k 3
  screening=$(/home/user/miniconda3/envs/dreamer_uav/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["screening"])' \
    "$selection/checkpoint_metrics.json")
  if [[ "$screening" != PASS ]]; then
    echo "Stopping before another training window: screening=$screening"
    exit 2
  fi
done

checkpoint="$train/ckpt/$(cat "$train/ckpt/latest")"
scripts/evaluate_stage_s3a_recovery2.sh "$checkpoint" S3A_A1_Recovery2_Final 100
