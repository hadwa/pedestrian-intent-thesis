# Exp083 Frozen Design

The authoritative approved design is
`docs/superpowers/specs/2026-08-29-exp083-jaad-pcpa-aligned-exp082-design.md`,
SHA-256 `c9b0d34659d2a7084df950365475c4154b9ba4cb1836f701eb8c9f5bf3d4f117`.
The subsequently approved raw-frame continuity clarification is recorded in
`PROTOCOL_AMENDMENT.md` and must be read with that original specification.
The pre-training development-context quality sampling correction is recorded
in `QUALITY_GATE_AMENDMENT.md`; it preserves the frozen thresholds while
replacing a population-dependent 128-row lexical prefix with complete-split
evaluation.

Exp083 trains the exact 242-D Exp082-D XGBoost design separately on official
PCPA-style `JAAD_beh` and `JAAD_all`. It freezes 16 consecutive observed
frames, TTE 30--60, JAAD overlap 0.8, fixed threshold 0.5, and seeds 42/43/44.
The result is published-protocol-aligned, not claimed row-identical to
PCAP-SL. New perception extraction requires a reported coverage stop first.
