#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean"
EXPERIMENT_ROOT="$REPO_ROOT/experiments/050_mslstm_practical_behavior_replacement_pie_16f_tte30_60"
PYTHON="/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python"
RUNNER="$EXPERIMENT_ROOT/scripts/run_experiment.py"
LOG="$EXPERIMENT_ROOT/logs/detached_run.log"
FAILURE="$EXPERIMENT_ROOT/DETACHED_RUN_FAILED"

cd "$REPO_ROOT"
exec > >(tee -a "$LOG") 2>&1
rm -f "$FAILURE"

on_exit() {
    status=$?
    if [[ $status -ne 0 ]]; then
        printf 'exit_code=%s\nfailed_at=%s\n' "$status" "$(date --iso-8601=seconds)" > "$FAILURE"
        echo "Exp050 detached run failed with exit code $status"
    fi
}
trap on_exit EXIT

echo "Exp050 detached run started at $(date --iso-8601=seconds)"
"$PYTHON" "$RUNNER" prepare
"$PYTHON" "$RUNNER" train-all
"$PYTHON" "$RUNNER" evaluate
"$PYTHON" "$RUNNER" verify
echo "Exp050 detached run completed at $(date --iso-8601=seconds)"
