# Exp074 Frozen Design

- Base: exact 187-column Exp072 input.
- Addition: 20 Exp063 road-relative channels, each summarized as frames 1--8
  mean, frames 9--16 mean, and late minus early, for 60 columns total.
- Candidate: 247 columns through the exact Exp072 median-imputer/XGBoost.
- Seeds: 42, 43, 44; threshold: fixed 0.5.
- Train: set01/set02/set06; validation: set04/set05; gated test: set03.
- No context feature selection, model search, threshold search, or refit.
- YOLOP/BBox road construction is label-free. Predictor, summarizer, YOLOP,
  and pose/BIU costs are reported separately.

The complete gate is frozen in `pre_registration.json` and the approved design
spec is `docs/superpowers/specs/2026-08-27-pie-exp074-road-relative-yolop-design.md`.
