# Saved BIU-v2 XGBoost runtime audit

Frozen before timing, 2026-09-13. No training or performance selection.

Models: all three PIE Exp096 v2 core (135-D) and Exp116 pie_road (195-D) seeds 42/43/44. Use the same 100 observation IDs already used by the AIBThings RF timing study, in their original order. No choice based on errors or latency.

First reproduce BBox53, BIU-v2 56, early/late contrasts21 and ego5 from the original 16-frame data. Reproduce road60 summaries from the saved road-relative 16x20 sequences. Verify each feature against the frozen Exp116 cache and each model probability against its recorded predictions. The same core-column ordering must match Exp096. Input/output parity is required before timing.

Single CPU worker for XGBoost, BLAS and OpenMP, matching the old RF CPU timing policy; change only the in-memory worker count and verify probability parity. Preserve original saved models. Document current hardware/software; historical timing is not a controlled speed comparison.

200 warm-up window iterations, then 2,000 measured window iterations for shared feature components and fusion. Cached pipeline inference: 200 warm-ups then 2,000 iterations at batch one, and 20 warm-ups then 200 iterations at batch100, for every model. A separate seed42 full sequential workload measures actual single-window response time, with 100 warmups and 1,000 repetitions per configuration. Rotate all 100 windows. Fixed loop counts; do not select fastest runs.

Report:
- Recomputed BBox53, BIU56, temporal21, ego5, road summaries60, vector/name packing.
- Ready-feature median-imputer + XGBoost prediction, batch1 and batch100. Batch total and effective per-window cost are distinct.
- Effective AIBThings-style cost: BIU + temporal + ego + packing + amortized prediction; BBox53 is ready. Road model additionally includes road-summary construction from ready geometry.
- Actual single-window complete core feature construction (including BBox53), packing and prediction. Road variant additionally summarizes available geometry.

Excluded from all of these: pedestrian detection, pose estimation, road segmentation, mask-to-road-geometry construction, decoding/I/O, loading and training. Road cache has derived geometry, not masks: never call its measured cost mask-to-decision or all-road-feature extraction. Any separate mask-to-geometry measurement must be labeled as a diagnostic with its actual input source.
