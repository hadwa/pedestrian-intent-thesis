# Exp048 Frozen Design

The authoritative design is
`docs/superpowers/specs/2026-08-17-mslstm-practical-behavior-replacement-design.md`.

The target reference is fixed from the detailed thesis description:

`BBox + GT Action + GT Look + Ego Speed + Ego Acceleration + GT Static Context + YOLOP`

Variants A-E, native rows, augmentation, 15 epochs, seeds 42/43/44,
validation-F1-at-0.5 checkpoint selection, fixed-0.5 reporting, and final set03
evaluation are immutable. Repository README/ACTIVE_STREAMS omissions do not
redefine the thesis target. Historical results are reference-only; scientific
conclusions come from fresh paired A-E runs.

Before training, the F1 and PR-AUC materiality/non-inferiority margin is frozen
at `0.005`. A substantial behavioral-recovery fraction is frozen at `0.50`
when the corresponding fresh `B-C` denominator is positive.
