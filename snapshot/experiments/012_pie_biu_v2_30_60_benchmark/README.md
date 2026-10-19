# Experiment 012: PIE BIU v2 vs BIU v1 benchmark (30-60, overlap 0.6)

## Purpose

Benchmark direction-aware BIU v2 features against BIU v1 on the MAIN PIE literature benchmark
(16 observed frames, TTE 30-60 frames, window overlap 0.6, event-aligned, test = set03).
This is the PIE counterpart of experiment `009_jaad_biu_v2_benchmark_comparison`.

## Protocol

- Splits (frozen, reused as-is): `/home/abdelrahman/Hadwa/BIU-PID-Distillation/data/splits_pie_wacv_pcpa_30_60_overlap06/{train,val,test}_sequences.csv`
  - train = set01+set02+set06 (2366 windows), val = set04+set05 (3619), test = set03 (3760).
  - Observed frames are recovered as `start_frame..end_frame` (asserted: consecutive, step 1, length 16 for every window).
- Feature sets: `bbox_only`, `biu_v1_only`, `biu_v1_plus_bbox`, `biu_v2_core`, `biu_v2_core_plus_bbox`.
- Models: logistic regression, random forest, XGBoost (identical hyperparameters to the JAAD 009 script, imported from it).
- Seeds: 42, 43, 44.
- Threshold rules: `fixed_0_5` and `val_f1` are the reported rules (the script also logs the other JAAD-009 rules: `val_balanced_accuracy`, `val_fpr_le_0p1`, `val_precision_ge_0p8`). All threshold selection uses VALIDATION data only; test data is never used for selection.

## Inputs used (all read-only)

- BIU v2 extractor: `src/biu_v2_extractor.py` (`extract_biu_v2(coords, conf, bbox_sequence)`)
- BIU v2 feature dictionary: `experiments/008_biu_v2_design_spec/metrics/biu_v2_feature_dictionary.csv`
  (`biu_v2_core` = `original_anatomical_biu` + `lower_body_commitment_biu` + `torso_center_commitment_biu` group prefixes)
- Archived bbox / BIU v1 feature tables: `/home/abdelrahman/Hadwa/BIU-PID-Distillation/data/processed/features_wacv_pcpa_30_60_overlap06/{bbox,biu}_{train,val,test}.csv`
  (reused, NOT rebuilt; QC metadata columns excluded per the archive trainer's `FEATURE_METADATA_COLUMNS`, giving bbox=53 and biu=72 features, matching archived `feature_count`)
- PIE keypoints JSONL: `/home/abdelrahman/Hadwa/pedestrian-intent-biu/data/data_keypoints_all.jsonl`
  (streamed; indexed by `(set, video, pid, frame_idx)`)
- Per-frame PIE bboxes (used only as low-motion direction-estimation fallback inside `extract_biu_v2`):
  parsed from `/home/abdelrahman/Hadwa/MSLSTM-PID/PIE/annotations/annotations` via the archive's
  `src/features/bbox_features.py:load_pie_bbox_annotations` (imported read-only via importlib).
- Sanity anchor: `/home/abdelrahman/Hadwa/BIU-PID-Distillation/results/wacv_pcpa_30_60_overlap06/lightweight_baseline_metrics.csv`
  (random_forest `bbox` test F1 = 0.857564; random_forest `biu_bbox` test F1 = 0.866359; seed 42, validation-F1 threshold).

## How to rerun

```bash
cd /home/abdelrahman/Hadwa/BIU-PID-Distillation-clean
python3 scripts/experiments/pie_biu_v2_30_60_benchmark.py
```

## Outputs

- `metrics/pie_biu_v2_benchmark_results.csv` — per feature set / model / seed / threshold rule.
- `metrics/pie_biu_v2_benchmark_results_mean_std.csv` — mean±std over seeds.
- `metrics/pie_biu_v2_vs_v1_delta_table.csv` — BIU v2 vs BIU v1 deltas (fixed_0_5 and val_f1).
- `metrics/pie_biu_v2_sanity_anchor_comparison.csv` — comparison against the published archive numbers.
- `metrics/pie_biu_v2_missing_keypoint_rows.csv` — windows dropped for missing keypoints.
- `metrics/pie_biu_v2_extraction_coverage.csv` — keypoint/bbox coverage summary.
- `metrics/pie_biu_v2_feature_set_definitions.csv` — feature set sizes and exact column lists.
- `predictions/{feature_set}__{model}__seed{seed}.csv` — per-sample val+test probabilities (45 files).
- `report.md` — headline findings.

## Final conclusion

See `report.md`.
