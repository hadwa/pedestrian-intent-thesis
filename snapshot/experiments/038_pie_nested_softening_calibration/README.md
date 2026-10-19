# Exp038: PIE Nested Softening Calibration

## Status

`COMPLETED - NO ADVANCE`.

Exp038 tests one confirmatory hypothesis: nested softening-only temperature
scaling can preserve Exp037's strong `joint_temporal_ego_xgb` discrimination
while reducing its high-confidence false-positive and false-negative tails.

The experiment reuses Exp037 features and raw development OOF predictions. It
keeps threshold `0.5`, model seeds, estimators, labels, splits, and features
unchanged. Set06 is conditional confirmation; set03 is opened only after an
advancing set06 lock and remains test-informed diagnostic evidence.

## Research Basis

Guo et al., “On Calibration of Modern Neural Networks,” ICML 2017, motivates
temperature scaling as a simple post-hoc calibration method. The paper does not
validate PIE, tree ensembles, or Exp038's additional `T >= 1` softening
constraint. Official source and local PDF provenance are recorded in
`papers/README.md`.

The full frozen design is in
`docs/superpowers/specs/2026-08-15-pie-nested-softening-calibration-exp038-design.md`.

## Outcome

The nested softening temperatures were between `1.653449` and `1.932497` for
`joint_temporal_ego_xgb`. Fixed-0.5 F1 and the entire confusion matrix remained
exactly invariant. High-confidence FPs fell from 132 to 39 and high-confidence
FNs from 110 to 59, reducing the combined tail from 242 to 98 (`59.5%`). Brier
score improved from `0.091914` to `0.088572`, log loss from `0.337000` to
`0.295637`, and 10-bin ECE from `0.052617` to `0.037596`.

Development did not advance because pooled PR-AUC changed from `0.812923` to
`0.807927`, a loss of `0.004996` versus the frozen noninferiority allowance of
`0.002`. Each outer set used a different leakage-free temperature, so pooled
cross-set ordering could change even though ranking within every set remained
invariant. Every other preregistered gate passed.

Consequently set06 and set03 were not materialized. Exact results, temperatures,
bootstrap intervals, and gate evidence are in `RESULTS.md` and `metrics/`.
