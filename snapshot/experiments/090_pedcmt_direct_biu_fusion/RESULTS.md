# Exp090 Results

## Outcome

**Validation-only negative under the frozen gate.** The gate failed on fixed-0.5
F1 and accuracy non-inferiority, so the 1,501-window final JAAD cohort was not
loaded or evaluated by Exp090. No Exp090 final-test predictions or
`EXPERIMENT_COMPLETE.json` exist.

## Validation Results

Values are mean +/- sample standard deviation across seeds 42, 43, and 44 on
the frozen 274-window validation cohort.

| Variant | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FP | FN | High-conf errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A frozen BBox+ego | 0.7725 +/- 0.0970 | 0.7730 +/- 0.0344 | 0.8902 +/- 0.1546 | 0.8250 +/- 0.0889 | 0.7846 +/- 0.0929 | 0.8120 +/- 0.0625 | 43.7 | 18.7 | 0.0 |
| B A+raw pose residual | 0.6168 +/- 0.0132 | 0.7616 +/- 0.1503 | 0.6275 +/- 0.1819 | 0.6607 +/- 0.0724 | 0.6933 +/- 0.0349 | 0.7835 +/- 0.0263 | 41.7 | 63.3 | 0.7 |
| C A+BIU union residual | 0.7251 +/- 0.0404 | 0.8164 +/- 0.0454 | 0.7196 +/- 0.0223 | 0.7648 +/- 0.0318 | 0.8180 +/- 0.0630 | 0.8874 +/- 0.0491 | 27.7 | 47.7 | 1.3 |

High-confidence errors are `FP probability >= 0.9` plus
`FN probability <= 0.1`.

## Paired Mean Deltas

| Comparison | Accuracy | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|
| A -> B | -0.1557 | -0.1643 | -0.0913 | -0.0285 |
| A -> C | -0.0474 | -0.0602 | +0.0335 | +0.0754 |
| B -> C | +0.1083 | +0.1041 | +0.1247 | +0.1039 |

C improved PR-AUC over both A and B in every seed. It also learned genuinely
signed corrections in every seed (roughly 53--57% positive and 43--47%
negative). It failed the gate because its fixed-0.5 F1 and accuracy were not
within 0.005 of A.

## Grounded Interpretation

C materially outperformed the raw-pose residual B on every mean metric. This
supports the narrow representation claim that the BIU union was more useful
than the raw pose sequence to this exact direct-fusion branch.

C did not improve the complete frozen predictor at threshold 0.5. Relative to
A, it reduced mean FP from 43.7 to 27.7 and raised precision, but increased mean
FN from 18.7 to 47.7 and reduced recall. Per-seed transitions show the same
failure mode in seeds 43 and 44: C removed 32/27 FPs but changed 46/45 parent
TPs into FNs. The better BCE and ranking metrics therefore did not translate
into better fixed-threshold classification.

This result does not justify changing the threshold, correction bound, loss,
or checkpoint rule inside Exp090. The frozen design requires stopping this
neural line after a failed gate.

## Traceability

- Parent/source audit: `metrics/source_and_parent_audit.json`
- Per-seed validation metrics: `metrics/validation_per_seed_results.csv`
- Validation summary: `metrics/validation_summary_results.csv`
- Gate decision: `metrics/validation_gate_report.json`
- Branch histories: `logs/B_seed*.history.csv`, `logs/C_seed*.history.csv`
- Saved validation probabilities/corrections: `predictions/validation_*.csv`

Commands used:

```bash
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/090_pedcmt_direct_biu_fusion/scripts/run_experiment.py prepare --device cpu
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/090_pedcmt_direct_biu_fusion/scripts/run_experiment.py smoke --device cpu
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/090_pedcmt_direct_biu_fusion/scripts/run_experiment.py train-all --device cuda
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/090_pedcmt_direct_biu_fusion/scripts/run_experiment.py gate
```
