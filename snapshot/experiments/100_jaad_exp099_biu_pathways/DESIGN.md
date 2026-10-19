# Exp100: Separate BIU pathways inside the Exp099 model family

Approved: historical v2-core-56, not v2-full-63. This is the JAAD counterpart
of the Exp096 pathway separation, retaining Exp099 categorical vehicle action.
Do not modify any parent experiment or start a further experiment afterward.

## Frozen arms

Shared non-pose block: BBox-53 + six existing BBox early/late/difference
summaries + Exp099's 15 categorical vehicle-state summaries = 74 columns.

| Arm | Inputs | Dimension | Fits |
|---|---|---:|---|
| v1 | shared + v1-72 + three existing pose-quality temporal summaries | 149 | three fresh |
| v2 | shared + v2-core-56 + same three pose-quality summaries + twelve existing v2 temporal summaries | 145 | three fresh |
| combined | exact Exp099 augmented input, unchanged column order | 197 | reuse three Exp099 models |

The shared pose-quality summaries are existing Exp077 features, not a new
confidence hypothesis. V1 keeps its frozen validity counters. V2 keeps its
frozen BBox direction fallback. No seven extra full-63 metadata columns are
added. No raw-pose model or new ego feature is introduced. Vehicle action is
annotated categorical state, not physical speed or automatically estimated ego.

This is a pathway comparison, NOT a pure representation-only comparison:
v2 carries its historical twelve additional temporal summaries. Matching the
PIE Exp096 design does not make PIE and JAAD ego modalities interchangeable.

## Recovering v2 correctly

Exp077's deduplicated combination contains 72 v1 features and only 36 unique
v2 columns, with overlapping names resolved in favor of v1. A v2-only arm must
not inherit those v1 values. Recompute the complete 56-D core from the exact
SHA-pinned primary/supplemental JAAD pose JSONLs and observation BBoxes, using
the historical Exp076/077 extractor call, float64 parsing and defaults. This
includes its historical image_size=(1920,1080) normalization convention rather
than silently switching to Exp089's full-63 adapter. No pose estimator is run.

Require all 36 existing non-overlapping v2 columns to reproduce the parent cache
within rtol1e-9/atol1e-8. Recompute v1 and compare all 72 columns as an additional
pose/frame parity check. Preserve overlapping v1/v2 names in separate namespaces.
Only observed frame IDs may enter extraction; never use event BBox or future pose.
Cache reconstructed features only in Exp100. Hash all input sources and code.

## Protocol, reuse and evaluation

- Clean JAAD behavior-pedestrian default split; 1552 train/274 validation/1501
  test windows; 16 consecutive observations, TTE30--60, overlap0.8.
- Seeds42/43/44, fixed0.5; exact Exp077/099 one-stage 250-tree XGBoost recipe,
  depth4, LR0.05, subsample/column sample0.8, L2=1, four CPU threads, training
  negative/positive class weight, median imputation. No tuning, oversampling,
  threshold adjustment, early stopping or train+validation refit.
- Authenticate Exp099 models, feature contract, cohorts and saved validation
  predictions. Require exact float32 validation-probability reproduction for
  combined-model reuse. Reuse test combined predictions, not test inference.
- Train six new models. All nine validation rows must complete before test
  features open. All arms advance regardless of score; no arm selection.
- Preserve exact population, observation identities, labels and parent ordering;
  fail closed on missing/conflicting assets. Verify split video/track disjointness.
- Report per-seed, mean/sample SD, paired v2-v1, combined-v1 and combined-v2
  deltas, fixed0.5 error transitions, and high-confidence FP>=.9/FN<=.1 counts.
- PR-AUC means average precision, consistent with the historical runners.
- Use 5000 paired pedestrian-track bootstrap replicates (seed42), seed-ensemble
  probabilities and 95% percentile intervals for F1, PR-AUC and ROC-AUC.
  Report seed consistency separately. These descriptive intervals are not
  multiplicity adjusted and do not establish universal BIU-version dominance.
- Retrospective matched ablation on already exposed JAAD test data, not pristine
  confirmation. Freeze before test and make no post-result experimental changes.
- Save persistent logs, source/reuse audits, all probabilities and metrics.
  Publish EXPERIMENT_COMPLETE.json only after all artifact/metric checks pass.
