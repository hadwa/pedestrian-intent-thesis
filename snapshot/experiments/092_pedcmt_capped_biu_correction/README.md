# Exp092: PedCMT Capped BIU Correction

This saved-model experiment tests whether limiting the Exp091 natural BIU
branch to ten percent of its learned correction preserves the strong PedCMT
parent while retaining BIU ranking information.

No model is retrained. See `DESIGN.md` for the frozen validation gate and
`RESULTS.md` for the outcome.

## Result

The validation gate passed and the one-time final evaluation improved mean F1
from `0.6734` to `0.6967`, ROC-AUC from `0.6891` to `0.7225`, and PR-AUC from
`0.6256` to `0.6633`. All values use fixed threshold 0.5 where applicable.
