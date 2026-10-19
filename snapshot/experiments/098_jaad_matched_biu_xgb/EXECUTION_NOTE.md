# Exp098 execution and verification note

The frozen runner completed all 12 trainings, validation, final test inference,
5,000-replicate track bootstrap and report generation. Its last verification
step stopped because it parsed shortest-round-trip float32 CSV decimals as
float64 and required agreement at 1e-12. The largest observed test Brier-score
difference was 5.94e-10. This is serialization precision, not changed decisions.

The original XGBoost probabilities are float32. Casting the saved decimals back
to float32 reproduces the original values. The independent verifier applies
that restoration and repeats every saved metric check for all 24 validation/test
prediction files. It authenticates all frozen code, sources, models and outputs
before verification. A synthetic round-trip regression test covers this behavior.

No design, runner, source cache, model, probability file or benchmark metric was
changed. No model was retrained. The original failed verification remains in
logs/run.log; metrics/saved_prediction_verification.json records the independent
checks. EXPERIMENT_COMPLETE.json is published only after these checks succeed.

Reproduce final verification with:

```bash
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/098_jaad_matched_biu_xgb/scripts/verify_saved_outputs.py
```
