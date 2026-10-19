# Exp051: Normalized MSLSTM Fusion

Controlled follow-up to Exp050. The only model change is parameter-free normalization of every 512-D stream context before the existing learned weighted sum.

See `DESIGN.md` for frozen invariants and interpretation. Exp050 remains unchanged; its authenticated caches are reused read-only.

```bash
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python scripts/run_experiment.py prepare
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python scripts/run_experiment.py gpu-smoke
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python scripts/run_experiment.py train-all
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python scripts/run_experiment.py evaluate
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python scripts/run_experiment.py verify
```
