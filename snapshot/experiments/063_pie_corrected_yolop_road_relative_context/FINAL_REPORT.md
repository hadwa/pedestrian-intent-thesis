# Exp063 Final Report

## Outcome

**Verified validation-only negative result.** Neither corrected YOLOP candidate
passed the frozen Exp062 validation gate. `VALIDATION_ONLY_COMPLETE` was written,
set03 was never enumerated or accessed, and no final-test result exists.

This means Exp063 does **not** justify rerunning Exp053 or another full
five-stream MSLSTM fusion merely with this YOLOP substitution. Exp052 has no
YOLOP stream and therefore has nothing to rerun for this hypothesis.

## Controlled Setup

- PIE benchmark: 16 consecutive observed frames, TTE 30--60.
- Train: set01/set02/set06, 2,366 windows.
- Validation: set04/set05, 3,619 windows.
- Forbidden during development: set03.
- Anchors: authenticated Exp062 RF BIU-v1+BBox and XGBoost BIU-v2-core+BBox
  grouped-OOF/full-validation probability bundles.
- Context corrector: exact hashed Exp062 5,932-parameter ego+scene bounded
  residual model.
- Seeds: 42, 43, 44; fixed threshold: 0.5.
- H: historical flawed YOLOP 20-D extraction, reused from Exp062.
- G: corrected official masks with historical 9 drivable + 9 lane + 2 count
  layout.
- R: same corrected masks summarized as frozen 20-D YOLOP+BBox road-relative
  geometry.

YOLOP extraction used the pinned 7,940,846-parameter checkpoint with SHA-256
`620e7495328cfc8e04807c14cbb8f8df9916953359d190cbf494b2d049f03b0c`.
It processed 31,131 unique frames and 46,210 pedestrian-frame records across 34
train/validation videos in 840.31 seconds on an RTX 3070 Laptop GPU. All 5,985
windows were retained. The full cache occupies approximately 4.2 MB because
full-resolution masks were not persisted.

## Mean Validation Results

All values below are three-seed means. H is the historical Exp062 corrected
model; G and R are fresh Exp063 corrected models.

| Anchor | Representation | F1 @0.5 | PR-AUC | ROC-AUC | Harmonic | High-conf. errors |
|---|---|---:|---:|---:|---:|---:|
| RF BIU-v1+BBox | H historical | 0.696877 | 0.705828 | 0.892356 | 0.701310 | 65.33 |
| RF BIU-v1+BBox | G corrected grid | 0.698698 | 0.700691 | 0.891557 | 0.699681 | 50.67 |
| RF BIU-v1+BBox | R road-relative | **0.704158** | **0.705962** | **0.893083** | **0.705052** | 74.67 |
| XGB BIU-v2+BBox | H historical | **0.672888** | 0.711842 | 0.884767 | 0.691755 | 222.00 |
| XGB BIU-v2+BBox | G corrected grid | 0.670759 | **0.724440** | **0.889854** | **0.696469** | **215.67** |
| XGB BIU-v2+BBox | R road-relative | 0.671833 | 0.717324 | 0.888164 | 0.693833 | 222.00 |

The unchanged tree anchors themselves were:

| Anchor | F1 @0.5 | PR-AUC | ROC-AUC | High-conf. errors |
|---|---:|---:|---:|---:|
| RF BIU-v1+BBox | 0.697222 | 0.701152 | 0.891349 | 50.33 |
| XGB BIU-v2+BBox | 0.666941 | 0.714806 | 0.883470 | 225.33 |

## Controlled Deltas

| Comparison | Anchor | Delta F1 | Delta PR-AUC | Delta ROC-AUC | Delta high-conf. errors |
|---|---|---:|---:|---:|---:|
| H -> G | RF | +0.001821 | -0.005137 | -0.000799 | -14.67 |
| H -> G | XGB | -0.002129 | +0.012598 | +0.005087 | -6.33 |
| G -> R | RF | +0.005459 | +0.005271 | +0.001526 | +24.00 |
| G -> R | XGB | +0.001074 | -0.007116 | -0.001691 | +6.33 |
| H -> R | RF | +0.007281 | +0.000134 | +0.000727 | +9.33 |
| H -> R | XGB | -0.001055 | +0.005482 | +0.003397 | 0.00 |

Corrected grid improved XGB ranking relative both to H and to the raw XGB
anchor, but it did not preserve/improve F1 and remained far below the absolute
gate. Road-relative improved RF F1 and harmonic, but the mean gain was driven
mainly by seed 42: G -> R F1 deltas were +0.014793, +0.001586, and 0.000000;
PR-AUC deltas were +0.015930, -0.000762, and +0.000645. It also increased mean
RF high-confidence errors by 24 relative to G.

Saved-prediction transitions reinforce that finding. For RF G -> R, the mean
counts were 46 source-wrong cases corrected versus 35 source-correct cases
harmed, but seed 42 alone accounted for 130 corrected and 102 harmed; seed 43
had only 7/2 and seed 44 1/1. This is not stable evidence for a general
representation improvement.

## Representation Audit

| Representation | Split | Nonzero fraction | Road valid | Lane valid |
|---|---|---:|---:|---:|
| Corrected grid | train | 0.4185 | n/a | n/a |
| Corrected grid | validation | 0.4380 | n/a | n/a |
| Road-relative | train | 0.7786 | 0.8142 | 0.8292 |
| Road-relative | validation | 0.8237 | 0.8492 | 0.8919 |

Correct official postprocessing produced plausible sparse lane occupancy and
nontrivial drivable occupancy instead of the historical near-constant
channel-zero values. However, better semantic masks did not by themselves make
the small residual learner reliably protect and improve the strong tree anchor.
The nearby-vehicle count channel remained identically zero, as it was in the
historical cache; the normalized pedestrian-count channel was nonzero. This is
reported as a representation limitation, not changed post hoc.

## Frozen Gate Decision

All four candidates failed the absolute requirements F1 > 0.707526, PR-AUC >
0.748513, ROC-AUC > 0.901868, and harmonic > 0.727414. Additional failures:

- G+RF failed required relative harmonic/gain and slightly exceeded anchor
  high-confidence errors.
- R+RF satisfied the relative performance gain but failed the high-confidence
  error condition.
- R+XGB failed the required relative harmonic/gain.
- G+XGB satisfied all relative/stability/high-confidence conditions but failed
  all four absolute conditions.

Therefore the frozen decision is no winner and no set03 evaluation.

## Thesis-Safe Conclusion

Correcting the historical YOLOP postprocessing was necessary for semantic
validity, but it did not solve the predictive fusion problem. The corrected
grid offered an XGB ranking improvement, while road-relative geometry offered
an RF F1 improvement, yet neither effect was simultaneously strong, stable,
and low-confidence-error-safe across seeds. The evidence does not support
claiming that corrected or road-relative YOLOP context improves the current
PIE predictor overall.

The controlled rerun rule is therefore **not met**. Do not rerun Exp052, and do
not reproduce Exp053 with only a YOLOP swap. Any subsequent context experiment
requires a separately approved hypothesis addressing the correction learner or
context/anchor interaction rather than assuming that mask correction alone is
sufficient.

## Evidence Paths

- `metrics/extraction_manifest.json`
- `metrics/representation_diagnostics.csv`
- `metrics/validation_per_seed_results.csv`
- `metrics/validation_summary_results.csv`
- `metrics/paired_deltas.csv`
- `metrics/error_transitions_fixed_0p5.csv`
- `predictions/validation_*.csv`
- `VALIDATION_COMPLETE.json`
- `VALIDATION_ONLY_COMPLETE`
- `EXPERIMENT_COMPLETE`
