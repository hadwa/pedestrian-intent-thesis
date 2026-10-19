# Exp119: v2-only capped neural controls

See DESIGN.md for the frozen missing-control design. Fifteen new GPU trainings,
not a search. Existing direct-token PedCMT and PIE MSLSTM comparisons are reused.
No API-style cohort expansions, thesis changes or new pose extraction here.

Commands from repository root:

```sh
PY=/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python
env LD_LIBRARY_PATH=/usr/lib/wsl/lib PYTHONDONTWRITEBYTECODE=1 $PY experiments/119_pedcmt_v2_capped_completion/scripts/run.py --audit-only
env LD_LIBRARY_PATH=/usr/lib/wsl/lib PYTHONDONTWRITEBYTECODE=1 $PY experiments/119_pedcmt_v2_capped_completion/scripts/launch.py
tail -n 30 experiments/119_pedcmt_v2_capped_completion/logs/pipeline.log
```

CUDA is accessible outside the execution sandbox on RTX3070 Laptop GPU. The
launcher starts an independent session without tmux; host sleep/shutdown still
interrupts computation. Source hashes, per-seed histories, completion/checkpoint
hashes and per-seed predictions provide persistent progress. Completed seeds
are skipped after hash verification. TEST remains locked until all 15 complete.
Audit-only runs never train or read test arrays. No success is claimed until
EXPERIMENT_COMPLETE.json exists. FAILURE.json is a historical failure record;
its timestamp must be considered against subsequent logs/markers.
