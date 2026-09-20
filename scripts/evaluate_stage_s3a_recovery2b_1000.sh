#!/usr/bin/env bash
set -euo pipefail

cd /home/user/UAV_Dreamer

checkpoint=outputs/dreamerv3/stageS3A_A1_Recovery2B_1000/ckpt/20260919T155040F646354
label=S3A_A1_Recovery2B_U1000_Latest_Quick

# The shared evaluator takes both values as arguments and refuses to overwrite
# existing evaluation directories. This wrapper pins the Recovery-2B artifact
# without changing the historical Recovery-2 scripts.
exec scripts/evaluate_stage_s3a_recovery2.sh "$checkpoint" "$label" 20
