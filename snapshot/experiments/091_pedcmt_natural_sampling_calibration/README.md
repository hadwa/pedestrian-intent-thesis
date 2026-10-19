# Exp091: PedCMT Natural Sampling and Calibration

This validation-only experiment tests whether Exp090's artificially balanced
correction-branch training distribution caused its conservative CROSS
probabilities. It reuses all Exp090/Exp089 artifacts and changes only the
training sampler to the natural JAAD window distribution.

An intercept-only, track-group-cross-fitted calibration analysis is secondary.
The final JAAD test cohort is not accessible through this runner.

## Result

The frozen validation gate failed. Natural sampling preserved a mean PR-AUC
gain over the parent (`0.8715` versus `0.8120`) but reduced fixed-0.5 F1
(`0.7447` versus `0.8250`). Cross-fitted intercept calibration raised F1 only
to `0.7511` while worsening Brier score and log loss. See `RESULTS.md`.

Commands:

```bash
PY=/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python
$PY experiments/091_pedcmt_natural_sampling_calibration/scripts/run_experiment.py prepare
$PY experiments/091_pedcmt_natural_sampling_calibration/scripts/run_experiment.py smoke --device cuda
$PY experiments/091_pedcmt_natural_sampling_calibration/scripts/run_experiment.py train-all --device cuda
$PY experiments/091_pedcmt_natural_sampling_calibration/scripts/run_experiment.py analyze
```
