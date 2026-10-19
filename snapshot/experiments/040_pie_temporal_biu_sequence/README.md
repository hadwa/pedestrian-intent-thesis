# Exp040: PIE Temporal BIU Sequence Diagnosis

## Status

`not_promising` development-only diagnostic. This experiment does not open or
score `set03`, does not alter any benchmark output, and is not an independent
test claim. `set06` was already opened by Exp039, so the five-set LOSO result
is useful for development diagnosis but is not a fresh confirmation.

## Protocol

- PIE standard anticipatory benchmark, 16 observed frames, TTE `30_60`.
- Authorized development sets: `set01`, `set02`, `set04`, `set05`, `set06`.
- Fixed threshold `0.5`; seeds `42,43,44`; no threshold selection.
- Exact cached keypoints, BBoxes, OBD, labels, and split rows reused from the
  authenticated Exp037/Exp039 sources.
- Candidate: a frozen 16-frame residual temporal convolutional model over 26
  BIU-v2/BBox/OBD channels. No RGB or road-scene context was used.

## Pooled LOSO Results

| Arm | PR-AUC | F1 @ 0.5 | High-confidence FP+FN |
|---|---:|---:|---:|
| Joint temporal-ego XGB anchor | 0.827034 | 0.786775 | 244 |
| Temporal BIU TCN candidate | 0.737731 | 0.742143 | 421 |
| BIU-v2/BBox XGB anchor | 0.759168 | 0.738162 | 297 |

The candidate is lower than the joint anchor by `0.089303` PR-AUC and
`0.044632` F1, and its high-confidence error count is higher by `177`.
The frozen promotion gate therefore failed all required improvement/tail
checks. Exact fold and calibration metrics are in `metrics/development_results.csv`
and `metrics/development_per_set.csv`.

## Reproducibility

- Pre-registration: `pre_registration.json`
- Sequence contract: `metrics/sequence_contract.json`
- Prepared cohort/provenance: `metrics/prepared_metadata.csv`,
  `metrics/source_provenance.json`
- OOF predictions: `metrics/sequence_oof_predictions.csv`
- Anchor predictions: `metrics/anchor_oof_predictions.csv`
- Gate and lock: `metrics/gate_checks.json`, `metrics/development_lock.json`
- Final report: `RESULTS.md`
- Verification: `metrics/verification_report.json` (`PASS`)

## Research Basis

The temporal-convolution design was motivated by Bai, Kolter, and Koltun,
“An Empirical Evaluation of Generic Convolutional and Recurrent Networks for
Sequence Modeling.” The local paper and SHA-256 are recorded in
`papers/README.md`; that paper is methodological motivation only and is not
evidence of PIE performance.
