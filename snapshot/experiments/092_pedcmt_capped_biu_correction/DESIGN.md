# Exp092 Design

Exp092 evaluates one fixed post-training correction cap. It performs no model
training and does not alter Exp090 or Exp091.

## Frozen Rule

```text
final_logit = frozen_Exp089_A_logit + 0.10 * frozen_Exp091_N_correction
```

The `0.10` scale was selected from Exp091 validation-only sensitivity analysis
as a round conservative value inside its observed `0.01--0.12` admissible
range. It is not changed after validation or test access.

## Validation Gate

At fixed threshold 0.5 and averaged over seeds 42/43/44, N10 must:

- improve PR-AUC over A by at least 0.005;
- keep F1 within 0.005 of A;
- keep accuracy within 0.005 of A;
- not increase high-confidence FP+FN.

The final JAAD cohort remains inaccessible until this gate passes. If it
passes, the exact frozen rule is evaluated once without further modification.
