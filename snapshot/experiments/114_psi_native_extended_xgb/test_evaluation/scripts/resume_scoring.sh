#!/usr/bin/env bash
set -euo pipefail

cd /home/abdelrahman/Hadwa/BIU-PID-Distillation-clean
root=experiments/114_psi_native_extended_xgb/test_evaluation
python=/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python

test -s "$root/cache/features/S187.npy"
test -s "$root/cache/features/R247.npy"
test ! -d "$root/predictions"
"$python" -u "$root/evaluate.py" evaluate
"$python" -u "$root/evaluate.py" verify
"$python" -u "$root/verify_results.py"
