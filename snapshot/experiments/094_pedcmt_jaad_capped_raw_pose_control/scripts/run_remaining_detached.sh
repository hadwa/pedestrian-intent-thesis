#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean
python_bin=/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python
runner=experiments/094_pedcmt_jaad_capped_raw_pose_control/scripts/run_experiment.py
status_file=experiments/094_pedcmt_jaad_capped_raw_pose_control/logs/DETACHED_FINISH_STATUS.txt

cd "$repo_root"
trap 'exit_code=$?; printf "%s\n" "$exit_code" > "$status_file"' EXIT

"$python_bin" "$runner" analyze
"$python_bin" "$runner" report
"$python_bin" "$runner" verify
"$python_bin" -m pytest -q tests/test_exp094_pedcmt_jaad_capped_raw_pose_control.py
"$python_bin" -m compileall -q \
  experiments/094_pedcmt_jaad_capped_raw_pose_control/scripts \
  tests/test_exp094_pedcmt_jaad_capped_raw_pose_control.py
