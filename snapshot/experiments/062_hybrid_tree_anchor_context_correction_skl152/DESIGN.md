# Exp062 Design Binding

The scientific design inherited unchanged from Exp061 is
`docs/superpowers/specs/2026-08-25-exp061-hybrid-anchor-context-correction-design.md`.
The bounded runtime remediation is approved in
`.superpowers/sdd/2026-08-25-exp061-hybrid-tree-context-correction/exp062-remediation-brief.md`.
These files jointly bind Exp062. Implementation, training commands,
documentation, and later audit artifacts cannot override, relax, reinterpret,
or extend them.

Exp062 compares two fresh BBox+BIU tree-anchor families with the same compact,
bounded ego-motion and YOLOP context correction. The primary benchmark is PIE
with 16 observed frames, TTE 30--60, three fixed seeds, and a fixed 0.5
threshold. Its context pathway is a selective correction of a non-neural tree
predictor, not an end-to-end neural replacement.

## Frozen Validation Gate

The validation gate is frozen before training. A corrected candidate is eligible
only when all ten conditions below hold against its own fresh anchor:

1. Mean validation F1 is strictly greater than `0.7075263341950917`.
2. Mean validation PR-AUC is strictly greater than `0.7485129299280930`.
3. Mean validation ROC-AUC is strictly greater than `0.9018675548671817`.
4. Mean harmonic F1/PR-AUC is strictly greater than `0.7274142468351926`.
5. Mean harmonic improvement is at least `0.005`.
6. Neither mean F1 nor mean PR-AUC is more than `0.002` below the anchor.
7. At least one of mean F1 or mean PR-AUC improves by strictly more than
   `0.005`.
8. No seed's harmonic-score delta is below `-0.005`.
9. Mean total high-confidence errors do not exceed the anchor.
10. Provenance, parity, OOF, temporal, and leakage invariants all pass.

Set03 can be accessed exactly once only after an authenticated
`FROZEN_FINAL_MODEL.json` is created by the final-freeze contract. That contract
recomputes and requires the eligible gate decision, records the selected anchor
family, and requires a concrete authenticated candidate manifest. The manifest
canonically binds the selected anchor, exact seeds 42/43/44, checkpoint hashes,
candidate and anchor validation summaries, config, source manifest, and
validation artifacts. Freeze verifies that manifest against its function inputs
and hashes the manifest into `FROZEN_FINAL_MODEL.json`.

The frozen package versions are:

```text
python==3.10.20
numpy==1.26.3
scipy==1.13.1
pandas==2.3.3
scikit-learn==1.5.2
joblib==1.5.1
threadpoolctl==3.6.0
xgboost==3.2.0
torch==2.11.0+cu128
pytest==9.1.1
```
