# Experiment 016: JAAD scene-context tabular features (30_60)

## Purpose

Build a lightweight tabular CONTEXT feature family for the JAAD `30_60` event
benchmark from JAAD's existing scene annotations (no image processing, no new
extraction) and benchmark whether it strengthens the long-horizon models
against the existing `bbox_only` and `biu_bbox` references.

## Annotation sources found and used (read-only)

- `/home/abdelrahman/Hadwa/datasets/JAAD/annotations_attributes/video_*_attributes.xml`
  — per-pedestrian attributes: `age`, `crossing`, `crossing_point`,
  `decision_point`, `designated`, `gender`, `group_size`, `id`,
  `intersection`, `motion_direction`, `num_lanes`, `old_id`, `signalized`,
  `traffic_direction` (346 files).
- `/home/abdelrahman/Hadwa/datasets/JAAD/annotations_traffic/video_*_traffic.xml`
  — per-video `road_type` plus per-frame `ped_crossing` (crosswalk in scene),
  `ped_sign`, `stop_sign`, `traffic_light` (346 files).

Join key: `(video_id, ped_id)` for attributes, `video_id` (+ observed frame
ids) for traffic. Coverage is 100% for all 3327 benchmark windows
(333 unique tracks, 186 videos).

## Benchmark inputs (read-only, archive repo)

- Splits: `/home/abdelrahman/Hadwa/BIU-PID-Distillation/data/jaad_splits_event_16_30_60_overlap08_default/{train,val,test}.csv`
- Features: `/home/abdelrahman/Hadwa/BIU-PID-Distillation/data/jaad_features_event_16_30_60_overlap08_default/{split}_{bbox,biu_bbox}.csv`

## Context columns (20) and their source fields

See `metrics/context_feature_definitions.csv` for the machine-readable table.

| column | source | field | encoding |
|---|---|---|---|
| attr_age_ordinal | attributes | age | ordinal child=0, young=1, adult=2, senior=3 |
| attr_gender_female / attr_gender_male | attributes | gender | one-hot (n/a -> 0,0) |
| attr_group_size | attributes | group_size | numeric |
| attr_designated_crossing | attributes | designated | 1 if D else 0 (scene infrastructure: designated crossing point) |
| attr_intersection | attributes | intersection | 1 if yes else 0 |
| attr_num_lanes | attributes | num_lanes | numeric |
| attr_signalized_s / attr_signalized_ns | attributes | signalized | one-hot S / NS (n/a -> 0,0) |
| attr_traffic_oneway | attributes | traffic_direction | 1 if OW else 0 (TW) |
| attr_motion_lateral / attr_motion_longitudinal | attributes | motion_direction | one-hot LAT / LONG (n/a -> 0,0) — FLAGGED, see caveat |
| traffic_road_type_street / traffic_road_type_parking_lot | traffic | road_type | one-hot (garage = implicit baseline) |
| traffic_ped_crossing_frac | traffic | frame/ped_crossing | fraction of OBSERVED window frames with a crosswalk marked in scene |
| traffic_ped_sign_frac | traffic | frame/ped_sign | fraction of observed frames with pedestrian sign |
| traffic_stop_sign_frac | traffic | frame/stop_sign | fraction of observed frames with stop sign |
| traffic_light_red_frac / traffic_light_green_frac / traffic_light_present_frac | traffic | frame/traffic_light | fraction of observed frames with red / green / any light state |

All attribute-derived columns are static per pedestrian track and `road_type`
is static per video — this is inherent to JAAD's annotation granularity and is
stated explicitly. Per-frame traffic flags are aggregated over the observed
window frames only (never over future frames).

## Leakage policy

- EXCLUDED as features: `crossing` (label attribute), `crossing_point`,
  `decision_point` (future event frame indices), `id`, `old_id`
  (identifiers). Asserted programmatically in the script; full raw-field
  disposition table in `metrics/leakage_audit_raw_field_disposition.csv`.
- FLAGGED as future-informed: `motion_direction` is annotated by the JAAD
  annotators over the WHOLE track (including frames after the observation
  window). Empirically it is a near-label proxy at 30_60: all 999 train
  positives have `motion_direction=LAT` and single-feature train ROC-AUC is
  0.957. Every feature set is therefore benchmarked both WITH motion
  (`*_context`) and WITHOUT it (`*_context_no_motion`). Only the `no_motion`
  variants should be quoted as honest long-horizon results.
- Labels are constant per track in this benchmark (0 of 333 tracks have mixed
  window labels) and videos are split-disjoint (0 of 186 videos appear in
  more than one split). Static context therefore acts as a track-level scene
  prior that generalizes across videos, not as memorization.

## Models / protocol (identical to experiment 009 conventions)

- logistic_regression (median-impute + standardize, class_weight=balanced),
  random_forest (300 trees, min_samples_leaf=2, balanced_subsample),
  xgboost (250 est, depth 4, lr 0.05, subsample/colsample 0.8,
  scale_pos_weight).
- Seeds 42/43/44; thresholds: fixed 0.5 and validation-F1 (validation-only
  selection, never test); test metrics reported as mean/std over seeds.

## How to rerun

```bash
cd /home/abdelrahman/Hadwa/BIU-PID-Distillation-clean
MPLCONFIGDIR=/tmp/mplconfig python3 scripts/experiments/jaad_context_features.py
```

## Outputs

- `metrics/jaad_context_feature_table.csv` — per-window context vector (3327 rows).
- `metrics/context_feature_definitions.csv` — column -> source field -> encoding.
- `metrics/leakage_audit_raw_field_disposition.csv` — every raw annotation field and its disposition.
- `metrics/feature_set_summary.csv` — feature counts per benchmarked set.
- `metrics/jaad_context_benchmark_results_per_seed.csv` — all per-seed results.
- `metrics/jaad_context_benchmark_results_mean_std.csv` — mean/std over seeds.
- `predictions/{feature_set}__{model}__seed{seed}.csv` — per-sample val+test probabilities (72 files).
- `report.md` — headline comparison and conclusion.

## Final conclusion

Context helps at 30_60. Honest (no-motion) variants: `bbox_plus_context_no_motion`
best test F1 0.801 (xgb, val-F1-selected) vs `bbox_only` reference 0.711
(logreg). See `report.md` for full numbers and caveats.
