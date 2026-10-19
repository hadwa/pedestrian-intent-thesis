#!/usr/bin/env bash
set -euo pipefail

REPO=/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean
EXP="$REPO/experiments/089_pedcmt_standardized_jaad_biu_transfer"
PY=/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python
RUNNER="$EXP/scripts/run_experiment.py"

cd "$REPO"
mkdir -p "$EXP/logs"

if [[ ! -f "$EXP/PREPARATION_COMPLETE.json" ]]; then
  "$PY" "$RUNNER" prepare
fi
if [[ ! -f "$EXP/logs/SMOKE_COMPLETE.json" ]]; then
  "$PY" "$RUNNER" smoke --device cuda
fi
"$PY" "$RUNNER" train-all --device cuda
"$PY" "$RUNNER" evaluate --device cuda
