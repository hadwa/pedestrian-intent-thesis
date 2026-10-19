#!/usr/bin/env bash
set -euo pipefail

cd /home/abdelrahman/Hadwa/BIU-PID-Distillation-clean
root=experiments/114_psi_native_extended_xgb/test_evaluation
python=/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python

test -f "$root/metrics/video_recovery.json"
test -f "$root/cache/features/S187.npy"
for video_part in "$root"/cache/road_parts/*.npz; do
  test -s "$video_part"
done
test "$(find "$root/cache/road_parts" -maxdepth 1 -name '*.npz' | wc -l)" -eq 49

"$python" -u "$root/evaluate.py" extract-road
"$python" -u "$root/evaluate.py" evaluate
"$python" -u "$root/evaluate.py" verify
"$python" -u "$root/verify_results.py"
