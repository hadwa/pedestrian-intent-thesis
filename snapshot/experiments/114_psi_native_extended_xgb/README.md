# Exp114: PSI-native extended XGBoost

This train/validation-only experiment originally blocked before preparation.
The original full-cohort design and `BLOCKED.json` remain as history. The user
subsequently approved `APPROVED_COHORT_AMENDMENT.md`: exclude exactly the 585
training windows without PSI2 speed inputs, then compare S187 (the complete
PSI-adapted Exp075 source block) against R247 (S187 plus 60 TwinLiteNet+ road
summaries) on identical amended rows with the same recipe and seeds42/43/44.

The source preflight found no PSI2 CV/speed member for three selected training
videos. They contribute 585 windows, including 301 windows from `video_0093`,
which also has no PSI1 CV member. Dropping those windows, zero filling speed, or
substituting driving-decision labels would violate the frozen matched design.
That original full-cohort run correctly stopped. The amended complete-input run
then completed with 26,295 training and unchanged 14,922 validation windows.
R247 improved mean macro-F1 from .609567 to .634889 and mean AP from .895862 to
.902144; all three paired seed deltas were positive for both metrics. See
`AMENDED_FINAL_REPORT.md`. This must not be described as the Exp113 population.

See `SOURCE_PREFLIGHT.md`, `metrics/source_preflight.json`, and `BLOCKED.json`
for the exact evidence. Run the local verifier with:

```bash
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/114_psi_native_extended_xgb/scripts/preflight.py
```

The historical preflight command still exits nonzero by design because its
full-cohort premise remains blocked. Amended completion is independently
verified in `INDEPENDENT_VERIFICATION.json`. No test data were accessed.
