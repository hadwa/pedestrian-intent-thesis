#!/usr/bin/env bash
set -euo pipefail

REPO="/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean"
PYTHON="/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python"
RUNNER="$REPO/experiments/084_pedcmt_biu_transfer/scripts/run_experiment.py"

cd "$REPO"
"$PYTHON" "$RUNNER" train-all --device cuda
"$PYTHON" "$RUNNER" evaluate --device cuda
