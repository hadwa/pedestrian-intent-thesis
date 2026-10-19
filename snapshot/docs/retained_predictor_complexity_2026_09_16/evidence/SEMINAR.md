# Compact crossing predictors with explicit complexity accounting

| Predictor | Model structure / neural scalar counts | Serialized inference size | Mean / median / p95 (ms) |
|---|---|---|---|
| XGBoost behavioral (135-D) | 250 trees / 5,674–5,756 nodes | 0.351–0.353 MiB (pipeline) | 1.06 / 1.05 / 1.17 |
| XGBoost + road context (195-D) | 250 trees / 5,872–5,890 nodes | 0.360–0.361 MiB (pipeline) | 1.17 / 1.16 / 1.26 |
| MSLSTM practical baseline | 6.765M trainable scalars | 25.829 MiB (weights) | 7.75 / 7.70 / 8.30 |
| MSLSTM + BIU-v2 + 7 auxiliaries (63-D) | 6.840M trainable scalars | 26.114 MiB (weights) | 7.96 / 7.87 / 8.65 |
| MSLSTM + raw pose | 9.006M trainable scalars | 34.383 MiB (weights) | 10.44 / 10.36 / 11.09 |
| PedCMT PIE frozen parent | 1.834M frozen scalars | 7.059 MiB (weights) | 3.27 / 3.23 / 3.59 |
| PedCMT PIE complete capped predictor | 1.935M total; 0.100M trainable | 7.449 MiB (weights) | 3.48 / 3.44 / 3.78 |

Tree structure/value counts are not neural scalar-weight counts; trainability refers to fitting, and uncompressed artifact sizes and one-core CPU timings exclude upstream perception.

CPU: 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz. Batch 1; one worker/thread; 60 warm-ups + 200 measurements in each of 3 blocks per seed; 3 seeds; 32 ready PIE windows. Times are pooled over independent models, not a three-model ensemble. Parent and branch-only timings are not interchangeable with complete PedCMT latency.
