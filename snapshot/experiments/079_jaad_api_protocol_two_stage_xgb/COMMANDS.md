# Exp079 Commands

Commands were run from the active repository root.

```bash
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/079_jaad_api_protocol_two_stage_xgb/scripts/prepare_pose_manifest.py

/home/abdelrahman/miniconda3/envs/cv_env/bin/python -m src.pose.extract_jaad_vitpose_jsonl \
  --dataset_root /home/abdelrahman/Hadwa/datasets/JAAD \
  --required_rows_csv experiments/079_jaad_api_protocol_two_stage_xgb/cache/required_missing_pose_rows.csv \
  --output_jsonl experiments/079_jaad_api_protocol_two_stage_xgb/cache/jaad_api_protocol_missing_pose.jsonl \
  --prepare_only

LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  /home/abdelrahman/miniconda3/envs/cv_env/bin/python \
  experiments/079_jaad_api_protocol_two_stage_xgb/scripts/run_pose_extraction.py \
  --device cuda:0 --warmup-source-frames 20 --progress-every 500

/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python \
  experiments/079_jaad_api_protocol_two_stage_xgb/scripts/verify_pose_completion.py

/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python \
  experiments/079_jaad_api_protocol_two_stage_xgb/scripts/run_experiment.py prepare

/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python \
  experiments/079_jaad_api_protocol_two_stage_xgb/scripts/run_experiment.py validate

/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python \
  experiments/079_jaad_api_protocol_two_stage_xgb/scripts/run_experiment.py evaluate

/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python \
  experiments/079_jaad_api_protocol_two_stage_xgb/scripts/analyze_results.py

/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python \
  experiments/079_jaad_api_protocol_two_stage_xgb/scripts/run_experiment.py report
```
