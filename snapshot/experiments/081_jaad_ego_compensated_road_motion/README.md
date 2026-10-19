# Exp081: JAAD Ego-Compensated Road Motion

This experiment tests whether image-derived camera-motion compensation makes
road-relative pedestrian features useful beyond the frozen Exp077 portable
anchor on the clean JAAD 16-frame, TTE 30--60 benchmark.

The experiment is pre-registered and phase-gated. The designated JAAD test
cohort must remain sealed until the validation advance gate passes. The visual
motion estimate is an image-space proxy, not physical ego speed.

See `DESIGN.md`, `PROVENANCE_AUDIT.md`, and `pre_registration.json` for the
frozen design and authenticated sources.

## Frozen Variants

- A: exact Exp077 182-D portable anchor.
- B: A plus the 15-D image-space visual ego-motion proxy.
- C: A plus the exact 60-D uncompensated Exp063 road representation.
- D: A plus the matched 60-D camera-motion-compensated road representation.

Only D can advance through the frozen validation gate. If it advances, final
test evaluation compares frozen A and D; B and C remain validation controls.

## Execution

Use Python from `/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python`.
Run one phase per command and redirect stdout/stderr into `logs/` for persistent
progress. There is deliberately no command that crosses the validation gate
automatically.

```bash
python scripts/run_experiment.py audit
python scripts/run_experiment.py extract-development
python scripts/run_experiment.py prepare
python scripts/run_experiment.py validate
python scripts/run_experiment.py latency
python scripts/run_experiment.py verify-gate
```

If `verify-gate` prints `no_advance`, run `report` and `verify`, then stop. If it
prints `advance`, run `extract-test`, `evaluate`, `report`, and `verify` in that
order.

The extraction cache contains only compact arrays. Full-resolution TwinLite
masks are never persisted. Test data is not enumerated, decoded, or hashed by
the development phases.
