# Saved-model evaluator completion

All 117 audited checkpoint files were exercised. No model was trained. See [the full guide](../../SAVED_MODEL_EVALUATION.md) for commands, exact tolerances, original input sources and limitations.

```json
{
  "checkpoint_files_audited": 117,
  "checkpoint_files_exercised": 117,
  "new_evaluation_jobs": 207,
  "previous_psi_jobs": 24,
  "models_trained": 0,
  "historical_reference_metrics_matched": true,
  "original_repositories_blocked_during_replay": true,
  "neural_validation_device": "cuda",
  "cpu_neural_replay_verified": false,
  "fresh_environment_install_verified": false,
  "reports": {
    "tabular": "runs/tabular_saved_replay/verification.json",
    "mslstm": "runs/mslstm_saved_replay/verification.json",
    "pedcmt": "runs/pedcmt_saved_replay_v2/verification.json",
    "psi": "runs/psi_saved_model_replay_v2/replay.json"
  }
}
```

The original repositories were read-only during input export and explicitly blocked during replay. Neural checks used CUDA; clean environment installation and CPU neural replay are not verified. Parent-reference PedCMT precision is documented in the guide. Historical thesis tables were not rewritten.
