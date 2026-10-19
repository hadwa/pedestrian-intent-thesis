# Exp062 Runtime Provenance

## Remediation Boundary

Exp061 failed its unchanged `1e-7` RF validation-probability parity gate under
scikit-learn `1.7.2`. A validation-only audit proved that scikit-learn `1.5.2`
reproduces the historical Exp012 RF seed-42 vector over all 3,619 validation
rows with maximum probability delta `2.220446049250313e-16`, zero fixed-0.5
class mismatches, and maximum metric delta `2.220446049250313e-16`.

Exp062 is a fresh non-overwriting remediation. It preserves the Exp061 model
hypothesis, inputs, variants, gates, seeds, threshold, architecture, source
isolation, OOF rules, and final-access discipline. Its only scientific change
is the historical-compatible RF construction runtime. Exp061 files, markers,
partial models, logs, staged sources, and metrics remain immutable.

## Exact Runtime

The ignored `.venv/` is built with the Python `3.10.20` `biu-pid-distill`
interpreter and `--system-site-packages`. NumPy `1.26.3`, SciPy `1.13.1`,
scikit-learn `1.5.2`, joblib `1.5.1`, and threadpoolctl `3.6.0` are overlaid
from `/tmp/exp061-skl152-wheelhouse` with `pip --no-index`. Pandas `2.3.3`,
XGBoost `3.2.0`, Torch `2.11.0+cu128`, and Pytest `9.1.1` are inherited and
checked at every public runner invocation.

| Wheel | SHA-256 |
| --- | --- |
| `joblib-1.5.1-py3-none-any.whl` | `4719a31f054c7d766948dcd83e9613686b27114f190f717cec7eaa2084f8a74a` |
| `numpy-1.26.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` | `bdd2b45bf079d9ad90377048e2747a0c82351989a2165821f0c96831b4a2a54b` |
| `scikit_learn-1.5.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` | `3a686885a4b3818d9e62904d91b57fa757fc2bed3e465c8b177be652f4dd37c8` |
| `scipy-1.13.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` | `f26264b282b9da0952a024ae34710c2aff7d27480ee91a2e82b7b7073c24722f` |
| `threadpoolctl-3.6.0-py3-none-any.whl` | `43a0b8fd5a2928500110039e43a5eed8480b918967083ea48dc3ab9f13c4a7fb` |

## Interpretation Boundary

Full-vector reproduction proves version-sensitive tree construction compatible
with scikit-learn `1.5.2`; it does not independently prove that the unrecorded
historical environment reported that literal package version. Exp062 remains
subject to the unchanged six-anchor validation parity gate before OOF or neural
training. This implementation package contains no new benchmark result and
does not authorize set03 access.
