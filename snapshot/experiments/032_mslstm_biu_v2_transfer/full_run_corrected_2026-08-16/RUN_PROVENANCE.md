# Corrected Exp032 Run Provenance

## Scientific status

This is a controlled diagnostic transfer study under the recovered local
MSLSTM next-frame task. It is not the 16-frame/TTE 30--60 thesis benchmark, and
its PIE set03 evaluation is not an independent confirmation because set03 was
inspected after Exp032 was designed.

## Frozen execution

- Observation/target: 30 observed raw frames, `pred_len=1` raw frame.
- Train: recovered balanced index, 109,934 rows, 54,967 per class.
- Validation: all 77,288 reconstructed set05/set06 rows.
- Test: all 271,078 reconstructed set03 rows.
- Variants: fresh MSLSTM control and the same model plus a 128-D late BIU-v2 branch.
- Seeds: 42, 43, 44.
- Epochs: 5.
- Batch size: 32.
- Optimizer: Adam, learning rate `1e-4`.
- Checkpoint selection: validation F1 at fixed 0.5.
- Primary reporting threshold: fixed 0.5; validation-F1 threshold is secondary.
- Device recorded by the runner in `metrics/latency_summary.csv`: `cuda`. The
  earlier handwritten CPU statement was contradicted by the generated runtime
  artifact and is superseded here.

## Authenticated inputs

- `MSLSTM-PID/checkpoints/train_index.pkl`: `585af93c504c89922cab076a637b74e1cdd72ebb705b302b0fc2a00ecafb537c`
- `MSLSTM-PID/checkpoints/scalers.pkl`: `bc775afffa00e15ad7e21208991f3c1a1070f6b4512fccd461c9c51e8f2e0451`
- `MSLSTM-PID/PIE/annotations/data_cache/pie_database.pkl`: `410e9422755c0387259d5689cac7ed18b0b0509719a63fddaba76b778cf69d6f`
- Corrected runner at launch: `fb3526b6eb1366da7d8a1ce0118758e30373ccf29384e1af78a6aaede53dcf61`

BIU cache hashes and reuse constraints are recorded in
`CACHE_REUSE_PROVENANCE.md`.

## Exact command

```bash
PYTHONUNBUFFERED=1 /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python \
  scripts/experiments/exp032_mslstm_biu_v2_transfer.py \
  --mode train \
  --epochs 5 \
  --output-root experiments/032_mslstm_biu_v2_transfer/full_run_corrected_2026-08-16
```

Detached session: `tmux` session `exp032_corrected`.
