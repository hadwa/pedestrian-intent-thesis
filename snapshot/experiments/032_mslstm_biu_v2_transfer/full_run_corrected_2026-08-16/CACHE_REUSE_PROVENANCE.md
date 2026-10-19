# Exp032 BIU Cache Reuse Provenance

The corrected run reuses the completed BIU-v2 cache from the interrupted
`full_run_2026-08-14` preparation. Training had not modified this cache. The
corrected balanced training index changes which cached rows are sampled for
training, not the reconstructed 681,820-row protocol or BIU feature values.

- Source cache: `../full_run_2026-08-14/cache/biu_v2_cache.npy`
- Source metadata: `../full_run_2026-08-14/cache/biu_v2_cache_meta.json`
- Cache SHA-256: `6dc6c06f91396e5db8f098a1eb7b9acdf5e3477fd3bccef28b0c780ac520d683`
- Metadata SHA-256: `575eb70feecb4b284e0bb61d21af393b586a5c23b00bb515b01a5de1f2519729`
- Copied cache and metadata hashes were verified identical before training.

This reuse avoids recomputing deterministic BIU-v2 features and does not reuse
any model checkpoint, probability, metric, or test prediction.

Post-run live-source verification recomputed the frozen BIU-v2 extractor from
the current pose source for 16 uniformly selected rows and 100 random rows
(seed `32042`). Both comparisons matched the reused float32 cache exactly
(`max_abs_error = 0.0`). See `metrics/cache_live_validation.csv`. The runner now
performs a deterministic live-sample comparison before accepting any reused
cache.
