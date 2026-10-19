# Exp089: Standardized JAAD PedCMT BIU Transfer

Controlled A--E PedCMT comparison on the existing JAAD-beh 16-frame,
TTE30--60 benchmark. See `DESIGN.md` for the frozen specification.

## Completion

The detached CUDA pipeline completed all 15 frozen runs (A--E, seeds 42/43/44)
and one gated final evaluation on 1,501 test windows. `EXPERIMENT_COMPLETE.json`
is the terminal marker. No threshold, model, feature, or checkpoint rule was
changed after test access.

## Fixed-0.5 results

Values are mean +/- sample standard deviation across seeds. ROC-AUC and PR-AUC
use raw probabilities and therefore do not depend on the classification
threshold.

| Variant | Inputs | F1 | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|
| A | BBox + vehicle action | 0.6734 +/- 0.0566 | 0.6891 +/- 0.0246 | 0.6256 +/- 0.0207 |
| B | A + raw pose | 0.6767 +/- 0.0588 | 0.6639 +/- 0.0488 | 0.6061 +/- 0.0815 |
| C | A + BIU-v1 | 0.5972 +/- 0.0625 | 0.6065 +/- 0.0190 | 0.5713 +/- 0.0142 |
| D | A + BIU-v2 | 0.6373 +/- 0.0270 | 0.6879 +/- 0.0346 | 0.6610 +/- 0.0630 |
| E | A + BIU-v1/v2 union | 0.6535 +/- 0.0570 | **0.7099 +/- 0.0291** | **0.6762 +/- 0.0334** |

The primary BIU-union deltas are:

| Comparison | F1 | ROC-AUC | PR-AUC | Favorable seeds (F1/ROC/PR) |
|---|---:|---:|---:|---:|
| A -> E | -0.0199 | +0.0209 | +0.0506 | 1/3, 2/3, 2/3 |
| B -> E | -0.0232 | +0.0460 | +0.0701 | 2/3, **3/3**, 2/3 |

Variant D also improves PR-AUC over B by 0.0548 on mean, favorable on all three
seeds. These results establish useful BIU ranking information relative to both
baselines, but not a fixed-0.5 F1 improvement.

## Clustered ensemble analysis

The three-seed ensemble uses 151 pedestrian-track clusters. E improves ensemble
ROC-AUC/PR-AUC by +0.0587/+0.0788 over A and +0.0427/+0.0356 over B. The 95%
track-cluster bootstrap intervals cross zero for all four ranking deltas, so the
standardized JAAD evidence is positive but not statistically decisive.

At fixed 0.5, E reduces ensemble false positives by 99 versus A and 134 versus B,
but introduces 122 and 126 additional false negatives. This operating-point
tradeoff explains why better ranking did not become better F1.

## Threshold diagnostic

The PedCMT paper reports F1 without an explicit threshold. Its released PIE and
JAAD code applies `numpy.round` to sigmoid probabilities, which is effectively a
0.5 decision threshold (apart from the measure-zero exact-0.5 tie). Exp089's
fixed 0.5 therefore follows the released implementation lineage.

As a clearly secondary post-hoc diagnostic, an F1 threshold was selected from
each seed's saved validation probabilities only and then applied unchanged to
its saved test probabilities:

| Variant | Fixed-0.5 test F1 | Validation-threshold test F1 | Delta |
|---|---:|---:|---:|
| A | 0.6734 | 0.6976 | +0.0242 |
| B | 0.6767 | 0.6998 | +0.0230 |
| C | 0.5972 | 0.6604 | +0.0632 |
| D | 0.6373 | 0.6605 | +0.0232 |
| E | 0.6535 | 0.6803 | +0.0268 |

Threshold selection helps all variants on mean, but E remains below A and B.
The E thresholds range from 0.0936 to 0.4118, also showing substantial seed-level
calibration instability. Fixed 0.5 is therefore part of the F1 shortfall, not
its main cause. This diagnostic does not replace the predeclared primary result.

## Conclusion

On standardized JAAD, the BIU-v1/v2 union improves probability ranking over
BBox+vehicle-action and over the same model with raw pose, but it does not improve
fixed-0.5 F1. The safe conclusion is that BIUs add useful ranking information;
the current PedCMT training/calibration does not convert that information into a
stable operating-point gain.

## Analysis artifacts

- `metrics/verified_per_seed_results.csv`
- `metrics/complete_paired_deltas.csv`
- `metrics/complete_paired_summary.csv`
- `metrics/complete_error_transitions_fixed_0p5.csv`
- `metrics/ensemble_metrics.csv`
- `metrics/ensemble_cluster_bootstrap.csv`
- `metrics/validation_threshold_diagnostic.csv`
- `metrics/validation_threshold_summary.csv`

The detached pipeline writes persistent output to `logs/pipeline.log`. It
creates `EXPERIMENT_COMPLETE.json` only after preparation, all 15 checkpoints,
and final evaluation succeed.

```bash
PY=/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python
$PY experiments/089_pedcmt_standardized_jaad_biu_transfer/scripts/run_experiment.py prepare
$PY experiments/089_pedcmt_standardized_jaad_biu_transfer/scripts/run_experiment.py smoke --device cuda
$PY experiments/089_pedcmt_standardized_jaad_biu_transfer/scripts/run_experiment.py train-all --device cuda
$PY experiments/089_pedcmt_standardized_jaad_biu_transfer/scripts/run_experiment.py evaluate --device cuda
```
