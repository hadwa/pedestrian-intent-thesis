#!/usr/bin/env bash
set -euo pipefail

cd /home/abdelrahman/Hadwa/BIU-PID-Distillation-clean
root=experiments/114_psi_native_extended_xgb/test_evaluation
python=/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python

for _ in $(seq 1 360); do
  complete=0
  for shard in 0 1 2 3; do
    if test -f "$root/metrics/video_recovery_shard_${shard}_of_4.json"; then
      complete=$((complete + 1))
    fi
  done
  if test "$complete" -eq 4; then
    break
  fi
  sleep 10
done
test "$complete" -eq 4

for shard in 0 1 2 3; do
  "$python" -c "import json; p=json.load(open('$root/metrics/video_recovery_shard_${shard}_of_4.json')); assert p['status']=='complete' and p['shard_count']==4 and p['shard_index']==${shard}"
done

"$python" -u "$root/scripts/recover.py" videos
"$python" -u "$root/evaluate.py" prepare-source
"$python" -u "$root/evaluate.py" extract-road
"$python" -u "$root/evaluate.py" evaluate
"$python" -u "$root/evaluate.py" verify
"$python" -u "$root/verify_results.py"
