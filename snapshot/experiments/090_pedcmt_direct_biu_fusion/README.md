# Exp090: Direct BIU Fusion Into a Frozen PedCMT Anchor

This is a controlled, validation-gated follow-up to Exp089. It tests whether a
direct signed residual can use BIU-v1/v2 union features more effectively than a
matched raw-pose branch while preserving the exact seed-matched Exp089
BBox+ego prediction as the baseline.

See [DESIGN.md](DESIGN.md) for the predeclared architecture, training contract,
and validation gate. Final-test evaluation is forbidden unless that gate passes.

## Lineage

- Parent experiment: `experiments/089_pedcmt_standardized_jaad_biu_transfer/`
- Parent architecture source: `experiments/084_pedcmt_biu_transfer/scripts/pedcmt_model.py`
- Reused inputs: Exp089 train/validation caches and training-only statistics
- Variants trained: B raw pose residual and C BIU-union residual
- Seeds: 42, 43, 44

## Commands

```bash
python3 experiments/090_pedcmt_direct_biu_fusion/scripts/run_experiment.py prepare --device cpu
python3 experiments/090_pedcmt_direct_biu_fusion/scripts/run_experiment.py smoke --device cpu
python3 experiments/090_pedcmt_direct_biu_fusion/scripts/run_experiment.py train-all --device cuda
python3 experiments/090_pedcmt_direct_biu_fusion/scripts/run_experiment.py gate
python3 experiments/090_pedcmt_direct_biu_fusion/scripts/run_experiment.py evaluate --device cuda
```

The last command fails closed unless `VALIDATION_GATE_PASSED.json` exists.

## Status

Training and validation are complete. The frozen validation gate failed on
fixed-0.5 F1 and accuracy non-inferiority, so final-test evaluation was not
performed. See [RESULTS.md](RESULTS.md) and
`metrics/validation_gate_report.json`.
