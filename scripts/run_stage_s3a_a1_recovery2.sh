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
  checkpoint="$train/ckpt/$(cat "$train/ckpt/latest")"
  label="S3A_A1_Recovery2_U${updates}_Quick"
  scripts/evaluate_stage_s3a_recovery2.sh "$checkpoint" "$label" 20
  /home/user/miniconda3/envs/dreamer_uav/bin/python scripts/check_recovery2_quick_gate.py \
    --label "$label" --history outputs/dreamerv3/stageS3A_A1_Recovery2_quick_gate.json
done

checkpoint="$train/ckpt/$(cat "$train/ckpt/latest")"
scripts/evaluate_stage_s3a_recovery2.sh "$checkpoint" S3A_A1_Recovery2_Final 100
