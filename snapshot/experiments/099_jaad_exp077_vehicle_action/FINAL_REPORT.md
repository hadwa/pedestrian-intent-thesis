# Exp099 Final Report

Clean JAAD: 16 observations, TTE30--60, fixed0.5, seeds42/43/44.
A is the authenticated Exp077 182-D one-stage baseline (three saved models reused).
B adds 15 fixed categorical vehicle-state summaries (three fresh 197-D models).

| Arm | F1 mean +/- SD | PR-AUC mean +/- SD | ROC-AUC mean +/- SD |
|---|---:|---:|---:|
| A | 0.707276 +/- 0.000481 | 0.713266 +/- 0.005142 | 0.752891 +/- 0.000400 |
| B | 0.719125 +/- 0.003319 | 0.755670 +/- 0.005050 | 0.777743 +/- 0.002805 |

## B minus A

- f1: +0.011850; favorable seeds 3/3.
- pr_auc: +0.042404; favorable seeds 3/3.
- roc_auc: +0.024852; favorable seeds 3/3.

All inputs except the categorical vehicle-state block are unchanged. The new
block is exact JAAD vehicle annotation, not measured speed or automatically
inferred ego motion. State occupancies and first/last one-hot codes use only
observation frames. No model or threshold was selected using test results.

This is a retrospective matched ablation on previously exposed test data, not
untouched confirmation. Track-bootstrap intervals are descriptive and based on
seed-ensemble probabilities. No subsequent experimental change is authorized.
