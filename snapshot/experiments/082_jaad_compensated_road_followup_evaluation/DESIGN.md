# Exp082 Design

Exp082 is a validation-selected follow-up evaluation of the frozen Exp081
Variant D. It does not reinterpret or overwrite Exp081's `no_advance` result.

## Frozen comparison

- A: the exact Exp081 182-D portable anchor.
- D: A plus the exact Exp081 60-D camera-motion-compensated road summary.
- Seeds: 42, 43, and 44.
- Models: reuse the six SHA-authenticated Exp081 models; no retraining.
- Cohort: clean JAAD test, 1,501 windows, 16 observed consecutive frames,
  TTE 30--60.
- Threshold: fixed 0.5.

The comparison was selected because validation showed a large F1 and ROC-AUC
gain and fewer high-confidence errors, despite a mean PR-AUC loss. Test results
will be reported for all frozen metrics without changing the model or selecting
among seeds after test access.

## Interpretation

This is follow-up evidence selected using validation results. It is not a
retroactive pass of Exp081's preregistered gate. JAAD test was historically
exposed in earlier repository work, which limits claims of pristine external
confirmation.
