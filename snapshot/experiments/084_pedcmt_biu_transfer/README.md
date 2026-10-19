# Exp084: PedCMT BIU Architecture Transfer

Exp084 tests whether frozen BIU-v1, BIU-v2, or their lossless union improves a
recent neural pedestrian-intention architecture relative to its BBox/ego base
and a raw-pose control. It uses the project-standard PIE 16-frame/TTE30--60
cohort and fixed threshold 0.5.

This is a controlled architecture transfer, not a numerical reproduction of the
PedCMT paper. The paper's released runner uses a different random split protocol.
The frozen scientific design is in `DESIGN.md`; no model selection uses set03.

## Variants

| ID | Inputs |
|---|---|
| A | BBox + ego speed |
| B | A + raw 17-joint pose sequence |
| C | A + 72-D BIU-v1 |
| D | A + 63-D BIU-v2 |
| E | A + 135-D source-namespaced v1/v2 union |

All extra representations enter as a context token in PedCMT's existing
cross-modal meeting. They are not hard-added to the final bottleneck.

## Commands

```bash
PY=/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python
$PY experiments/084_pedcmt_biu_transfer/scripts/run_experiment.py prepare
$PY experiments/084_pedcmt_biu_transfer/scripts/run_experiment.py smoke --device cpu
$PY experiments/084_pedcmt_biu_transfer/scripts/run_experiment.py train-all --device cuda
$PY experiments/084_pedcmt_biu_transfer/scripts/run_experiment.py evaluate --device cuda
```

The final command fails closed unless every A--E/42--44 checkpoint exists.
Per-epoch histories are flushed under `logs/`; validation and final probabilities
are stored under `predictions/`.

## References

- X. Chen et al., "PedCMT: Pedestrian Crossing Intention Prediction With Cross-
  Modality Fusion and Spatio-Temporal Multi-Task Learning," IEEE Transactions on
  Intelligent Transportation Systems, 2024, DOI `10.1109/TITS.2024.3386689`.
- Public implementation: `https://github.com/xbchen82/PedCMT`, audited at commit
  `d9be3f87d4fa0d8d935c1fbc55188cab2c079ca3`.

No paper PDF is copied into the repository because a clearly licensed open PDF
was not established. The DOI and exact public code revision are pinned instead.
