> Evaluation update: portable tabular, MSLSTM and full capped-PedCMT saved-model replay now passes. See [SAVED_MODEL_EVALUATION.md](SAVED_MODEL_EVALUATION.md). Earlier references below to pending neural/tabular inference are historical; raw-data and publication gaps remain.

# Reproduction status and commands

This package is not yet an end-to-end training release. Current commands use only the Python standard library and do not import or execute any historical experiment code.

From this repository:

```sh
python tools/evidence.py verify
python tools/evidence.py summaries
```

`verify` checks all copied evidence against the recorded SHA-256 hashes. `summaries` rebuilds Exp119, Exp122 test-evaluation, and Exp124 summary rows from their saved per-seed metric CSVs, using seeds 42/43/44, arithmetic means and sample standard deviations. It compares every rebuilt numeric cell to the preserved reference summary, and writes a new timestamped directory under `runs/`. It retains the original split labels, including diagnostic rows. These checks verify aggregation, not inference or training, and do not validate that per-seed source metrics were themselves correct.

The runner resolves its root from its own location. It refuses outputs outside `runs/` and refuses overwriting an existing run. Never execute scripts under `snapshot/` yet.

## Training dependencies confirmed from source

- Exp119 loads Exp093/090 model code, Exp084/089 parent checkpoints and normalization statistics, Exp050/089 source caches, and Exp093/090 cached parent outputs. These must be regenerated or supplied with checksums before training can be reproduced.
- Exp122 imports Exp121, which imports Exp120, with Exp053/050 preprocessing dependencies. Exp053 imports BBox feature code by an absolute archive path. A byte-exact copy is preserved in `vendor/research_archive/`, but the historical import chain remains unchanged and must not be executed.
- Exp124 imports Exp116's metrics helper, reads its PSI feature caches, clones its fitted parent pipelines, and reuses its C/D model predictions. Training A/B alone is not a reconstruction of all four arms.

`audit/portability_phase1/recorded_inputs.json` inventories recorded source paths and local availability for these three experiments. Presence is not a checksum verification; original recorded hashes are retained separately. This inventory is not a complete raw-data dependency graph.

Remaining work: resolve upstream raw dataset and perception dependencies; package portable feature builders and model implementations; preserve split/seed/selection/threshold contracts; separate training outputs from reference evidence; document exact environments and dataset acquisition; test without original filesystem access; reconcile final thesis mapping and review publication contents. Full training is not authorized by these checks.

## PSI implementation update

See [PSI_REPRODUCTION.md](PSI_REPRODUCTION.md) for the new standalone cached-feature workflow. It supersedes the earlier training-dependency blockers for that route only; the raw-data chain remains unfinished.
