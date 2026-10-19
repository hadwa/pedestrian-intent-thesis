# Exp102 Implementation Plan

Use executing-plans inline. No branch changes, commits, model search or edits
outside this isolated experiment folder; the other thesis session is active.

1. Audit existing results: authenticate Exp083 manifests; compare Exp031
   BBox candidate identity and recipe; record non-equivalent completed runs.
2. Write failing unit tests for arm dimensions, exact core/full slicing,
   namespaced union, complete30-row validation gate, test lock and hashes.
3. Implement scripts/run_experiment.py. Import existing frozen metric,
   extractor and estimator helpers without modifying their globals. Use only
   observation pose/BBoxes/image dimensions. Cache per split once for all,
   index behavior rows by exact frame/track keys. Preserve gap annotations.
4. Run tests. Prepare development and numerical BBox/source checks; freeze
   design/scripts/manifests and reuse audit before any fitting.
5. Run normal terminal all-phase execution with persistent logs. Reuse exact
   baseline probabilities; train only missing models on training populations.
6. Freeze all30 validation rows; then construct test features and evaluate all
   arms. Generate paired tables and track uncertainty without model changes.
7. Reproduce metrics from every saved probability file; authenticate reused
   predictions and freshly saved models. Write final report/completion marker.

Commands are logged by the runner. Unit test command:
`/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -m pytest experiments/102_jaad_api_matched_biu/scripts/test_exp102.py -q`
