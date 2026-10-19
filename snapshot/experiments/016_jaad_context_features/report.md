# Report: JAAD scene-context features at 30_60

## Question

Does a lightweight tabular context family built from JAAD's existing scene
annotations (traffic XML + pedestrian attribute XML, no image processing)
strengthen the long-horizon (30_60) models beyond the `bbox_only`
(F1 ~ 0.711) and `biu_bbox` (F1 ~ 0.697) val-F1-selected references?

## Answer

Yes — context clearly helps at 30_60, with one mandatory exclusion.

1. `motion_direction` must be excluded. It is annotated over the whole track
   (future included) and is a near-label proxy: all 999 train positives are
   `LAT`, single-feature train ROC-AUC 0.957. Feature sets including it reach
   test F1 ~ 0.89-0.90, which we classify as leakage-inflated, not a valid
   long-horizon result.
2. With motion excluded (`*_no_motion`, the honest variants), context still
   adds ~ +0.08-0.09 F1 over the references.

## Headline test numbers (mean over seeds 42/43/44, val-F1-selected threshold)

Source: `metrics/jaad_context_benchmark_results_mean_std.csv`.

| feature_set | model | test F1 | precision | recall | PR-AUC | ROC-AUC | bal. acc |
|---|---|---|---|---|---|---|---|
| bbox_only (reference) | logreg | 0.711 | 0.575 | 0.932 | 0.714 | 0.743 | 0.648 |
| biu_bbox (reference) | logreg | 0.697 | 0.574 | 0.889 | 0.693 | 0.742 | 0.640 |
| biu_bbox (reference, best model) | xgb | 0.707 | 0.567 | 0.940 | 0.704 | 0.738 | 0.639 |
| context_no_motion_only | logreg | 0.774 | 0.677 | 0.904 | 0.740 | 0.794 | 0.753 |
| bbox_plus_context_no_motion | xgb | 0.801 | 0.765 | 0.840 | 0.850 | 0.875 | 0.801 |
| biu_bbox_plus_context_no_motion | rf | 0.789 | 0.684 | 0.933 | 0.819 | 0.868 | 0.768 |
| context_only (motion included, FLAGGED) | logreg | 0.893 | 0.816 | 0.986 | 0.878 | 0.914 | 0.891 |
| bbox_plus_context (motion included, FLAGGED) | logreg | 0.893 | 0.819 | 0.981 | 0.919 | 0.938 | 0.890 |
| biu_bbox_plus_context (motion included, FLAGGED) | logreg | 0.903 | 0.847 | 0.968 | 0.913 | 0.937 | 0.903 |

Fixed-0.5 thresholds show the same ordering (e.g. `bbox_plus_context_no_motion`
xgb F1 0.789 vs `bbox_only` xgb 0.704); see the mean/std CSV.

Reference agreement check: our retrained `bbox_only` logreg val-F1-selected
test F1 = 0.7108 and `biu_bbox` logreg = 0.6972, matching the quoted
references (0.711 / 0.697) and experiment 009's archived results exactly.

## Deltas (honest variants vs references, val-F1-selected, best model per set)

- `context_no_motion_only` 0.774 vs `bbox_only` 0.711: **+0.063 F1** — the
  static scene prior alone already beats window bbox dynamics at this horizon.
- `bbox_plus_context_no_motion` 0.801 vs `bbox_only` 0.711: **+0.090 F1**,
  PR-AUC 0.850 vs 0.725, ROC-AUC 0.875 vs 0.744.
- `biu_bbox_plus_context_no_motion` 0.789 vs `biu_bbox` 0.707: **+0.082 F1**.
- Context helps `bbox` slightly more than `biu_bbox`; `bbox_plus_context_no_motion`
  is the best honest configuration overall.

## What drives the gain (single-feature train ROC-AUC)

`attr_intersection` 0.852, `attr_signalized_ns` 0.823,
`attr_designated_crossing` 0.807, `traffic_ped_crossing_frac` 0.688,
`attr_group_size` 0.657. These are genuine static scene/infrastructure
properties: JAAD pedestrians standing at unsignalized intersections with
designated crossings are far more likely to be crossers.

## Caveats (stated plainly)

- All context columns are static per track (attributes) or per video
  (road_type); only the traffic-frame fractions vary across windows, and only
  over observed frames. Window labels are constant per track in this
  benchmark (0/333 mixed tracks), so context gains come from a track-level
  scene prior, NOT from temporal dynamics. The improved score partly reflects
  JAAD's sampling bias (who gets annotated where), and this should be
  disclosed if quoted in a paper.
- Videos are split-disjoint (0/186 videos in more than one split), so the
  prior generalizes across videos rather than memorizing them.
- The `*_context` (motion-included) rows are retained for completeness but are
  leakage-inflated and must not be quoted as benchmark results.
- This is a diagnostic/exploratory experiment relative to the current
  AIBThings paper scope (BIU v1, fixed 0.5); fixed-0.5 numbers are in the
  mean/std CSV and show the same conclusion.

## Recommendation

If context is adopted, use `bbox_plus_context_no_motion` and document the
static-per-track nature of the features and the exclusion of
`motion_direction`, `crossing`, `crossing_point`, `decision_point`.
