# Experiment 010: JAAD qualitative ambiguous-negative casebook

## Purpose
Create small qualitative case sheets for JAAD ambiguous-negative, actual-positive, and clean-negative test cases using bbox + keypoint overlays.

## Inputs used
- `experiments/009_jaad_biu_v2_benchmark_comparison/metrics/`
- `results/jaad_temporal_biu_transition_audit/horizon_feature_rows.csv`
- Archived JAAD split/window artifacts from `/home/abdelrahman/Hadwa/BIU-PID-Distillation/data/...`
- Archived JAAD `30_60` per-sample BBox / BIU v1 predictions from `/home/abdelrahman/Hadwa/BIU-PID-Distillation/results/jaad_event_16_30_60_overlap08_default/per_sample_fixed_0p5_predictions.csv`
- JAAD images under `/home/abdelrahman/Hadwa/datasets/JAAD/images/`
- JAAD keypoints under `/home/abdelrahman/Hadwa/BIU-PID-Distillation/data/jaad_keypoints_vitpose_s_extended_short_horizon.jsonl`

## How to rerun
```bash
cd /home/abdelrahman/Hadwa/BIU-PID-Distillation-clean
python3 scripts/audits/jaad_qualitative_casebook.py
```

## Output folders
- `metrics/` for selected-case metadata and auto tags
- `figures/` for compact summary plots
- `case_sheets/` for per-case PNG contact sheets

## How to read the case sheets
- The large grid shows the 16 observed frames only.
- The bottom row shows future frames for qualitative audit only and is not model input.
- The title block gives horizon, group, true label, and any archived scores/predictions available.
- `BIU v2` per-sample score fields are marked `NA` because experiment 009 did not save per-sample predictions and this task does not retrain models.
