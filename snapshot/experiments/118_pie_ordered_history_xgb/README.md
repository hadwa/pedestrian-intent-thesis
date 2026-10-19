# Exp118: Ordered Frame History for XGBoost

Completed training/validation only. No set03 evaluation. PIE 16 observed frames,
TTE 30--60, seeds 42/43/44, fixed threshold .5. No pose extraction or neural
training. Existing experiment outputs unchanged.

## Answer

We had tried early/late XGBoost and a full ordered-cue TCN, but did not find this
exact XGBoost addition in prior reports/scripts. This experiment appends ordered
per-frame measurements to the matched Exp096 BIU-v2-core extended XGBoost.
It did not provide a convincing validation gain. This is not evidence that
XGBoost cannot use order, or that all temporal representations are ineffective.

## Three-Seed Validation Results

| Input | Dimensions | F1 mean +/- SD | AP mean +/- SD | ROC-AUC mean +/- SD |
| --- | ---: | ---: | ---: | ---: |
| Existing summaries | 135 | 0.712828 +/- 0.003830 | 0.782598 +/- 0.004021 | 0.919773 +/- 0.000396 |
| Summaries + ordered measurements | 471 | 0.710975 +/- 0.005190 | 0.783119 +/- 0.003451 | 0.918578 +/- 0.001504 |

Mean changes: F1 -0.001853, AP +0.000521, ROC-AUC -0.001195. F1 improves
in 1/3 seeds, AP in 2/3, ROC-AUC in 1/3. The tiny AP change does not establish
a useful gain. The frozen promising criterion fails; no test evaluation follows.

Mean FP: 286.67 -> 286.67; mean FN: 176.33 -> 178.67.
High-confidence FP (p>=.9): 111.33 -> 116.33.
High-confidence FN (p<=.1): 87.33 -> 95.67.
New fits took 1.99, 2.03 and 2.17 seconds, totaling approximately 6.19 seconds.
Baseline fit time zero in the CSV means reused, not an instantaneous fit.

## Scope and Limitations

The added block is 16 behavioral and 5 BBox channels x 16 frames, ordered
time-major. No added confidence, OBD, road, or other context channels. This is
not flattened raw 17-joint XY and does not repeat BIU56 at every frame. Frozen
BIU56 is retained and the cached frame-level measurements are an additional block.
Feature growth also changes the pool available to column subsampling despite
identical estimator settings. This comparison does not isolate ordering from all
effects of feature expansion. Validation has prior development exposure.

## Sources and Verification

- Baseline models and saved validation predictions:
  `experiments/096_pie_exp072_matched_biu_ablation/models/v2_seed{42,43,44}.joblib`
  and `predictions/validation_predictions.csv`.
- Summary train/validation inputs: `experiments/072_pie_joint_biu_temporal_ego_confirmation/metrics/prepared_{train,validation}.csv`.
- Ordered measurements, identities, equations and provenance:
  `experiments/040_pie_temporal_biu_sequence/metrics/prepared_sequences.npz`,
  `prepared_metadata.csv`, `sequence_contract.json`, `source_provenance.json`,
  and `scripts/sequence_core.py`.
- Cache hashes checked against Exp040 preparation marker; summary hashes checked
  against Exp096. All 5985 train/validation sequence identities matched exactly;
  labels/track/video parity, 16-frame and TTE checks passed. Train/val tracks
  disjoint. Baseline model parameters matched and validation probabilities replayed
  within 1e-7 with identical decisions before any new fit.
- `metrics/FROZEN.json`: hashes, columns, source paths, environment, replay errors.
- `metrics/per_seed_results.csv`, `summary_results.csv`, `paired_deltas.csv`:
  exact results. `predictions/`: six validation files. `models/`: three new fits.
- `metrics/INDEPENDENT_VERIFICATION.json`: independent saved-metric verification.

## Commands

Run from `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean`:

```bash
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -u experiments/118_pie_ordered_history_xgb/scripts/run.py >> experiments/118_pie_ordered_history_xgb/run.log 2>&1
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/118_pie_ordered_history_xgb/scripts/verify.py
```

Initial preflight attempts stopped before freezing/fitting because Python NaN
parameter values do not compare equal to themselves. Comparison serialization
was corrected for classifier and imputer settings, then all checks passed.
The retained log includes both failed preflight attempts and the successful run.
No experimental setting changed after fitting. The runner refuses overwriting
a frozen run. Nothing was committed or pushed by this task.
