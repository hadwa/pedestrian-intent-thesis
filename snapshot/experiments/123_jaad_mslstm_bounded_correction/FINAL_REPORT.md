# Exp123: frozen MSLSTM bounded correction

Validation-only retrospective integration study. Parent=Exp122 A AP-selected checkpoint for each seed, unchanged. No test access. Three added branches use raw pose, BIU56, or BIU56+seven auxiliaries63. Cap0.10*2*tanh applies during training and inference. Epoch0 identity is eligible under AP selection.

## Three-seed means, fixed threshold0.5

| Arm | F1 | AP | ROC-AUC | Accuracy | FP | FN | Branch parameters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.6741 | 0.8216 | 0.7653 | 0.6545 | 30.7 | 64.0 | 0 |
| raw | 0.6795 | 0.8219 | 0.7668 | 0.6594 | 31.0 | 62.3 | 2372738 |
| core56 | 0.6725 | 0.8245 | 0.7667 | 0.6533 | 30.3 | 64.7 | 205313 |
| full63 | 0.6798 | 0.8232 | 0.7686 | 0.6606 | 30.3 | 62.7 | 206209 |

## Paired directions

| Comparison | F1 delta | AP delta | ROC delta | Positive seeds F1/AP/ROC |
| --- | --- | --- | --- | --- |
| core56 minus baseline | -0.0016 | +0.0030 | +0.0014 | 1/3, 2/3, 2/3 |
| full63 minus baseline | +0.0057 | +0.0017 | +0.0033 | 2/3, 2/3, 2/3 |
| raw minus baseline | +0.0054 | +0.0004 | +0.0014 | 2/3, 2/3, 2/3 |
| core56 minus raw | -0.0070 | +0.0026 | -0.0000 | 0/3, 2/3, 1/3 |
| full63 minus raw | +0.0003 | +0.0013 | +0.0018 | 1/3, 2/3, 2/3 |

## Correction reachability

Only parent probabilities approximately0.450166--0.549834 can change class under this cap. The cap limits changes to confident baseline decisions, including confidently wrong decisions.

| Seed | Baseline errors | Errors within reach | Reachable FP | Reachable FN |
| --- | --- | --- | --- | --- |
| 42 | 75 | 9 | 1 | 8 |
| 43 | 129 | 10 | 2 | 8 |
| 44 | 80 | 10 | 1 | 9 |

## Integrity and interpretation boundaries

All12 saved validation files reproduce metrics within 0; bounded-logit reconstruction and unreachable-decision preservation passed. Parent checkpoint/state immutability, replay and no-gradient checks passed. Nine branches trained; no source models were retrained or changed.

Validation AP cannot be worse than the replayed baseline AP because epoch0 identity was eligible. That is a selection property, not independent generalization evidence. F1 can still deteriorate. This design differs from jointly trained Exp122 and from uncapped-training PedCMT; do not attribute every difference to one scalar fusion weight.

No cap/threshold search, final test evaluation, new architecture, thesis modification, commit or push followed this result. Full per-seed metrics, sample SD and transitions are in `metrics/`.

## Sources and commands

Frozen sources: `metrics/source_hashes.json`; procedure: `DESIGN.md`; parent paths: `../122_jaad_mslstm_no_noise/checkpoints/A_seed{42,43,44}_ap.pt`; exact parent replay evidence: `PRETRAINING_AUDIT_PASSED.json`.

Training: `env LD_LIBRARY_PATH=/usr/lib/wsl/lib PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -u experiments/123_jaad_mslstm_bounded_correction/scripts/run.py`.

Report: `env PYTHONDONTWRITEBYTECODE=1 /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/123_jaad_mslstm_bounded_correction/scripts/report.py`.
