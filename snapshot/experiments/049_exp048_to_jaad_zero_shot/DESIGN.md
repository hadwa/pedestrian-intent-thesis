# Frozen Exp049 Design

Exp049 is the zero-shot JAAD transfer evaluation defined and approved in
`docs/superpowers/specs/2026-08-23-exp049-exp048-jaad-zero-shot-design.md`.
That specification is authoritative; this file records its experiment-local
freeze.

## Fixed Contract

- Source models: all Exp048 A--F checkpoints for seeds 42, 43, and 44.
- Target cohort: official JAAD `default/test.txt` only.
- Temporal task: 30 contiguous observation frames; target is the next raw
  frame's frame-level crossing behavior.
- Threshold: fixed 0.5.
- No optimization, model selection, calibration, or JAAD fitting.
- A/B retain the specified JAAD semantic annotations; C--F are the practical
  variants; D/E/F use one identical JAAD pose source.
- Exact YOLOP parity and complete 30-frame pose coverage are fail-closed audit
  gates.
- JAAD target labels and model probabilities remain inaccessible until all
  pre-inference artifacts have been authenticated.
