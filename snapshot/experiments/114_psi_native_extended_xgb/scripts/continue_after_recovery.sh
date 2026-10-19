#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/114_psi_native_extended_xgb
PYTHON=/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python

printf 'Waiting for four disjoint video recovery manifests\n'
while true; do
  complete=0
  for shard in 0 1 2 3; do
    marker="$ROOT/metrics/video_recovery_shard_${shard}_of_4.json"
    if [[ -f "$marker" ]]; then
      complete=$((complete + 1))
    fi
  done
  printf '%s recovery shards complete at %s\n' "$complete" "$(date --iso-8601=seconds)"
  [[ "$complete" -eq 4 ]] && break
  sleep 60
done

# The unsharded resume pass authenticates every local MP4 and publishes the
# single manifest consumed by the GPU stage; it does not redownload valid files.
"$PYTHON" -u "$ROOT/scripts/recover_sources.py" videos >> "$ROOT/video_finalize.log" 2>&1
"$PYTHON" -u "$ROOT/run.py" extract-road > "$ROOT/road_extraction.log" 2>&1
"$PYTHON" -u "$ROOT/run.py" train > "$ROOT/train.log" 2>&1
"$PYTHON" -u "$ROOT/run.py" verify > "$ROOT/verify.log" 2>&1
printf 'Exp114 amended train/validation run verified at %s\n' "$(date --iso-8601=seconds)"
