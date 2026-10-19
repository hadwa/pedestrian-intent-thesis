# Exp083 Protocol And Provenance Audit

## Classification

Exp083 is a fresh, controlled, published-protocol-aligned comparison. It is
not claimed row-identical to PCAP-SL because that paper does not publish its
sample IDs, generated-window counts, or classification threshold.

## Frozen Protocol

- Dataset: JAAD 2.0 through the official local JAAD/PCPA API lineage.
- Cohorts: `sample_type='beh'` and `sample_type='all'`, trained separately.
- Observation: 16 consecutive official track observations.
- TTE: 30--60 frames from the final observation to the official endpoint.
- JAAD sample overlap: 0.8, giving a three-observation window stride.
- Splits: official train, validation, and test video memberships.
- Primary threshold: fixed 0.5; validation-selected F1 threshold is secondary.
- Seeds: 42, 43, and 44.

Two documented JAAD_all training windows on `video_0098:0_98_544` cross a
134-frame raw-annotation gap. They are retained because they are consecutive
official track observations; see `PROTOCOL_AMENDMENT.md`.

## Frozen Predictor

The model is Exp082-D: the exact 182-D Exp077 portable BBox/pose/BIU anchor
plus the exact 60-D Exp081 camera-motion-compensated road summary, followed by
one 250-tree XGBoost classifier. The frozen parameters are recorded in
`pre_registration.json`. Separate models are fitted for JAAD_beh and JAAD_all.

## Authenticated Sources

- JAAD API: `/home/abdelrahman/Hadwa/datasets/JAAD/jaad_data.py`, SHA-256
  `2523a3631fa0e3d3f9bb6d2e9a20b28905f30bda6aed106376b9a25b6e1a7d8c`.
- PCPA benchmark paper: `papers/Kotseruba_2021_PCPA_Benchmark.pdf`, SHA-256
  `3346fa5d35cc3726b5afe740c85d6e108679b0f943edb0cca67c94242f289f73`.
- Exp081 extraction: SHA-256
  `e0107d6614797d69b877b3c0221911d39f2856f555a16ac89b9244ef7bad4e43`.
- Exp081 model assembly: SHA-256
  `dd00565a645b9ac842ad080dd48900744e220dee28c7fb2e79d2f0712749077d`.
- TwinLiteNet+ Large checkpoint: SHA-256
  `5605d1a8ce762c1b6f7d23f4c2e7ec9384bfd6e6887b040b43e1f8902a817e22`.

Exact pose and RGB identity coverage was 100%. Shared development rows matched
Exp077 anchor features within `9.1e-13` and Exp081 road features exactly. The
pose-cache sources and their hashes are recorded in `INPUT_COVERAGE_AUDIT.md`.

## Development Gate Amendment

Before training and test access, the population-dependent 128-row lexical
prefix quality sample was replaced by complete-split evaluation while keeping
the original 0.95 transition and 0.5 displacement-reduction thresholds. See
`QUALITY_GATE_AMENDMENT.md`. This changed no predictor input or extracted
value.

