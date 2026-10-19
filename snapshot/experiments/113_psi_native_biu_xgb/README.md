# Exp113: PSI-native BBox versus BIUs

Matched PSI-trained XGBoost validation study. A: BBox53; B: BBox53+72-D BIU-v1;
C: BBox53+63-D full BIU-v2. No zero-shot transfer, roads, ego inputs, offset
correction, neural training, or test evaluation. See DESIGN.md for frozen settings.
All three seeds (42/43/44) are retained; no model selection from test results.

## Completed validation results

Nine fresh fits completed and independently verified. Test remains closed.
Mean +/- sample standard deviation across seeds42/43/44, fixed predictor .5:

| Variant | Accuracy | Crossing F1 | Macro F1 | PR-AUC (AP) | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| A: BBox53 | .726243 +/- .002325 | .822177 +/- .001555 | .613827 +/- .004693 | .890574 +/- .001443 | .724053 +/- .001848 |
| B: BBox53 + v1 | .725908 +/- .000549 | .819046 +/- .001426 | .626987 +/- .006022 | .905537 +/- .002223 | .744486 +/- .004580 |
| C: BBox53 + full v2 | .725551 +/- .001647 | .817970 +/- .000766 | .630226 +/- .003661 | .914416 +/- .000688 | .759855 +/- .001901 |

| Paired comparison | Crossing F1 delta | Macro F1 delta | AP delta | ROC-AUC delta |
|---|---:|---:|---:|---:|
| A to B | -.003130 | +.013160 | +.014963 | +.020434 |
| A to C | -.004207 | +.016399 | +.023843 | +.035803 |
| B to C | -.001076 | +.003239 | +.008879 | +.015369 |

Both BIUs improve macro-F1, AP and ROC-AUC over A on every seed, satisfying the
predeclared descriptive macro-F1/AP consistency criterion. Crossing-positive F1
decreases on every seed versus A. v2 improves AP and ROC-AUC over v1 on every
seed, but its macro-F1 gain is not consistent (seed43 decreases), and its crossing
F1 is not consistently better. No universally best representation is claimed.

Mean FP/FN: A2288.67/1796.33; B2106.67/1983.33; C2056.67/2038.67. BIUs trade
some crossing recall for improved negative-class discrimination. Fractional counts
are per-seed means, not pooled counts. All14922 validation windows have complete
pose, so the complete-pose secondary table matches the full validation table.

**Important class-imbalance baseline:** validation is11240 positive /3682 negative
(75.325%). Always predicting CROSS would give accuracy.753250, crossing F1.859262,
macro-F1.429631, AP.753250 and ROC-AUC.5. Thus our models do not beat that trivial
rule on accuracy or positive F1; they do improve macro-F1 and ranking substantially.
Do not present .818 crossing F1 alone as strong PSI performance. Macro-F1 weights
both classes equally; positive F1 does not. No threshold was adjusted to reverse
this trade-off. The fixed class-weighting recipe is identical across A/B/C.

Training:26880 windows,18799 positive,8081 negative,108 videos/174 tracks.
Validation:14922 windows,11240 positive,3682 negative,35 videos/77 tracks.
Training has585 all-zero/missing-pose windows (2.176%);26295 complete-pose windows.
Validation has no missing pose. v2 BBox fallback occurs898 times in training
(3.341%) and212 times in validation (1.421%). This makes v2 a frozen pose/BBox
behavioral representation, not a mathematically pure pose transformation.
Mean-vote ties at.5:5302 training and1827 validation windows. A tie can arise from
all-not-sure votes or mixed votes; these counts do not identify the former alone.

Conclusion: under this recovered PSI-native validation task, BIUs improve ranking
and macro-F1 beyond BBox, with v2 showing the stronger ranking result. They do not
improve positive F1 or accuracy. These overlapping-window validation results are
not independent statistical confirmation or held-out test evidence. No PSI test
result exists for Exp113 yet. No further model, tuning or test run was launched.

Verification: four unit tests pass; nine saved model predictions and metrics
reproduce, all features share identical BBoxes, IDs/horizons are checked, source
hashes are unchanged, and imputer medians equal training-only medians. See
`VERIFIED.json`, `metrics/per_seed_results.csv`, `metrics/summary_results.csv`,
`metrics/paired_deltas.csv` and `metrics/pose_coverage.csv`.

## Protocol and interpretation

15 consecutive observed frames predict binary human-estimated intent at the next
raw frame (frame16). This is not 16 observed frames/TTE30--60. The official PSI
mean-vote label gives cross/not_sure/not_cross weights1/.5/0; mean>=.5 is positive.
The annotation cutoff .5 is conceptually distinct from our predictor cutoff .5.
The official extension starts unlabeled frames with not_sure, forward fills, and
truncates already-crossed tails. We preserve these rules, including positive ties,
rather than dropping uncertainty to manufacture easier binary labels.

Official database and sampling behavior are preserved (including the strict
track length >16 rule and historical gap splitting). Unit tests compare our
window range with the pinned native sampler. This tests engineered representations
under the native task, not numerical reproduction of its neural baseline.
The full source cohort is retained regardless of pose-cache availability. Missing
pose is zero-filled and audited; complete-pose validation is secondary, using
the same fitted models and saved probabilities. Ground-truth BBoxes are inputs,
so this is not detector-inclusive deployment evaluation.

Native label construction legitimately consults annotations/full tracks for
cohort definition. Neither full-track geometry nor future annotation values enter
the observed feature vector; all feature extractors receive exactly15 frames.

## Sources

- `official_code/baseline/`: selected unmodified files from
  [PSI-Intent-Prediction](https://github.com/PSI-Intention2022/PSI-Intent-Prediction),
  revision63c584b415e9b147c06d7aad9819d0df39feee91.
- `official_code/dataset/`: annotation extension from
  [PSI-Dataset](https://github.com/PSI-Intention2022/PSI-Dataset),
  revisionbcefb1137ba12737ba2936bf50254c634d400f7a.
- `recovery.json`: SHA-256 and member paths for all143 official training/validation
  annotation files, code hashes and the chosen TrainVal split. Both archive split
  copies agree on train/val; test disagreement is not silently resolved here.
- Existing `data/psi_vitpose_s_all_tracks.jsonl`, SHA
  31d8f45a7f8b61803b2dde8f981ce012086ac9905455d7aaac743bc1ec2a3fc7.
  Cache naming/history is not cryptographic proof of its original estimator
  checkpoint. Exported static labels are ignored completely.
- Frozen existing extractors: Exp076 BBox53 and Exp089 72-D/63-D BIU adapters,
  backed by `src/features/biu_features.py` and `src/biu_v2_extractor.py`.
  Equations are unchanged; aggregations use15 instead of16 frames. v2 retains
  its BBox direction fallback. Confidence is not calibrated or newly added.
- Paper already stored: `papers/Jing_2026_PSI_arxiv_v3.pdf`, section4,
  [arXiv v3](https://arxiv.org/abs/2112.02604v3). No new paper download.

## Execution

Working directory: `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean`.
Commands (Python is the existing biu-pid-distill environment):

```bash
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -u experiments/113_psi_native_biu_xgb/recover.py > experiments/113_psi_native_biu_xgb/recovery_resume.log 2>&1
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/113_psi_native_biu_xgb/test_run.py
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -u experiments/113_psi_native_biu_xgb/run.py prepare > experiments/113_psi_native_biu_xgb/prepare.log 2>&1
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -u experiments/113_psi_native_biu_xgb/run.py train > experiments/113_psi_native_biu_xgb/train.log 2>&1
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/113_psi_native_biu_xgb/verify.py
```

Recovery resumed after the recorded user pause; previously recovered members were
CRC/size checked and reused. No videos or test annotation files downloaded.
The old PAUSED.md is historical, not the current run status. No expensive model
retraining or pose extraction is needed. XGBoost uses four CPU threads, preserving
the fixed recipe rather than changing algorithms merely to use a GPU.

Large source annotations, caches and model files stay local and ignored by git.
This new experiment is not yet committed or pushed.
