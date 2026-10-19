# Pedestrian intention thesis companion

Companion repository for the thesis experiment code and supporting evidence.

Repository: https://github.com/hadwa/pedestrian-intent-thesis

Start with the [thesis result-to-experiment index](thesis/14_RESULT_SOURCE_MAP.md), at the exact path cited in Appendix A.

**Availability:** code and reports are published here. Saved weights and frozen input bundles remain local and do not yet have public download links. Cloning this repository alone is not sufficient to run the saved-model evaluations.

Start with:

- [Saved-model evaluation commands](SAVED_MODEL_EVALUATION.md): tabular, MSLSTM, capped PedCMT and PSI. All 117 audited existing weight files have been exercised by inference checks; no retraining.
- [Result-to-artifact coverage](RESULT_ARTIFACT_COVERAGE.md): thesis entries and exact source files.
- [Data preparation](DATA_PREPARATION.md): input contracts, historical processing and remaining raw-data gaps.
- [Original training recipes](EXPERIMENT_RECIPES.md): documentation for readers, not tasks to execute during packaging.
- [Saved weights](SAVED_WEIGHTS.md): packaged artifacts and prediction-only limitations.
- [Thesis-link handoff](THESIS_LINK_HANDOFF.md): update the thesis session after publication.

The reference `snapshot/` preserves historical evidence. Do not execute its experiment runners directly: some retain original filesystem assumptions. Use the portable commands under `tools/`. Local model/data bundles are under `data/` and are excluded from Git; new evaluation outputs go under `runs/` without overwriting earlier runs.

Release work remaining: complete raw-data acquisition/regeneration instructions or distributable input bundles; verify a fresh environment installation; trim historical staging material; publish downloadable artifacts with checksums and tag a reproducible release. The existing inference checks do not claim raw-video reproduction or fresh training parity.
