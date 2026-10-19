# Exp091 Results

## Outcome

**Validation-only negative under the predeclared gate.** Natural-distribution
training retained a strong BIU ranking gain, but it did not restore fixed-0.5
F1 or accuracy and did not eliminate high-confidence errors. The final JAAD
test cohort was never loaded.

Values are means across seeds 42, 43 and 44 on the same 274-window validation
cohort. `C balanced` is the saved Exp090 result; only `N natural` was trained by
Exp091.

| Variant | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | Log loss | ECE | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A frozen parent | 0.7725 | 0.7730 | 0.8902 | 0.8250 | 0.7846 | 0.8120 | 0.1964 | 0.5799 | 0.2062 | 43.7 | 18.7 |
| C Exp090 balanced | 0.7251 | 0.8164 | 0.7196 | 0.7648 | 0.8180 | 0.8874 | 0.1913 | 0.5526 | 0.1771 | 27.7 | 47.7 |
| N Exp091 natural | 0.6971 | 0.7812 | 0.7118 | 0.7447 | 0.7971 | 0.8715 | 0.1832 | 0.5298 | 0.1658 | 34.0 | 49.0 |

Relative to A, N changed mean PR-AUC by `+0.0596` but F1 by `-0.0803` and
accuracy by `-0.0754`. Relative to Exp090 C, natural sampling reduced PR-AUC
by `0.0159` and F1 by `0.0201`, although Brier, log loss and ECE improved.

## Calibration Diagnostic

The track-group-cross-fitted intercept changed N as follows:

| Policy | Accuracy | Precision | Recall | F1 | Brier | Log loss | ECE | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw fixed 0.5 | 0.6971 | 0.7812 | 0.7118 | 0.7447 | 0.1832 | 0.5298 | 0.1658 | 34.0 | 49.0 |
| Cross-fitted intercept | 0.6959 | 0.7633 | 0.7392 | 0.7511 | 0.1968 | 0.5764 | 0.1476 | 39.0 | 44.3 |

Calibration recovered only `0.0063` mean F1 and worsened both proper scoring
rules, Brier and log loss. Fold-specific intercepts also alter global ranking,
so the cross-fitted ROC/PR values are diagnostic rather than evidence of a
representation change.

## Interpretation

Removing artificial window duplication did not solve the problem. The natural
branch still applied mean absolute logit corrections of `1.18--1.44` and
changed 46, 46 and 49 parent true positives into false negatives across the
three seeds. Its average correction was positive, so the failure is not a
simple global class-prior offset: large sample-dependent corrections are
overriding correct anchor decisions.

The result does not justify final-test access or another sampler search. It
supports retaining Exp090's conclusion: BIU union contains useful JAAD ranking
information, but this direct correction architecture does not convert it into
a reliable fixed-threshold gain.

## Traceability

- `metrics/source_and_sampling_audit.json`
- `metrics/validation_per_seed_results.csv`
- `metrics/validation_summary_results.csv`
- `metrics/crossfit_intercepts.csv`
- `metrics/reliability_bins.csv`
- `metrics/validation_gate_report.json`
- `predictions/validation_calibration_predictions.csv`
- `logs/N_seed*.history.csv`

Commands used:

```bash
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/091_pedcmt_natural_sampling_calibration/scripts/run_experiment.py prepare
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/091_pedcmt_natural_sampling_calibration/scripts/run_experiment.py smoke --device cpu
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/091_pedcmt_natural_sampling_calibration/scripts/run_experiment.py train-all --device cpu
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/091_pedcmt_natural_sampling_calibration/scripts/run_experiment.py analyze
```
