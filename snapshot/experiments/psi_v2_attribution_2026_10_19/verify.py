"""Independent aggregation checks of saved diagnostic outputs and source hashes."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
assert not (OUT / 'independent_verification.json').exists()
hashes = json.loads((OUT / 'source_hashes.json').read_text())
for name, digest in hashes.items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
members = pd.read_csv(OUT / 'membership.csv')
feature = pd.read_csv(OUT / 'feature_per_seed.csv')
blocks = pd.read_csv(OUT / 'block_per_seed.csv')
cues = pd.read_csv(OUT / 'per_seed.csv')
summary = pd.read_csv(OUT / 'summary.csv')
tested = 0
for seed in (42, 43, 44):
    f = feature[feature.seed == seed]
    total = f.mean_abs_shap.sum()
    biu = f[f.block == 'BIU-v2'].mean_abs_shap.sum()
    b = blocks[blocks.seed == seed].set_index('block')
    for block, mass in f.groupby('block').mean_abs_shap.sum().items():
        np.testing.assert_allclose(b.loc[block, 'share_pct'], 100 * mass / total, atol=1e-10)
        tested += 1
    z = np.load(OUT / f'cue_contributions_seed{seed}.npz', allow_pickle=False)
    labels = z['label']
    for j, cue in enumerate(z['cue']):
        s = cues[(cues.seed == seed) & (cues.cue == cue)].set_index('subset')
        for subset, mask in [('ALL', np.ones(len(labels), bool)), ('positive', labels == 1), ('negative', labels == 0)]:
            values = z['contribution'][mask, j] * (2 * labels[mask] - 1)
            np.testing.assert_allclose(s.loc[subset, 'toward_pct'], 100 * (values > 1e-10).mean(), atol=1e-10)
            np.testing.assert_allclose(s.loc[subset, 'net_mean'], values.mean(), atol=1e-10)
            tested += 2
        mass = f[f.cue == cue].mean_abs_shap.sum()
        np.testing.assert_allclose(s.loc['ALL', 'importance_pct'], 100 * mass / biu, atol=1e-10)
        np.testing.assert_allclose(s.loc['ALL', 'all_input_pct'], 100 * mass / total, atol=1e-10)
        weighted = (s.loc['positive', 'toward_pct'] * (labels == 1).sum()
                    + s.loc['negative', 'toward_pct'] * (labels == 0).sum()) / len(labels)
        np.testing.assert_allclose(weighted, s.loc['ALL', 'toward_pct'], atol=1e-10)
        tested += 3
for (subset, cue), group in cues.groupby(['subset', 'cue']):
    row = summary[(summary.subset == subset) & (summary.cue == cue)].iloc[0]
    for metric in ['importance_pct', 'all_input_pct', 'toward_pct', 'net_mean']:
        np.testing.assert_allclose(row[metric + '_mean'], group[metric].mean(), atol=1e-10)
        np.testing.assert_allclose(row[metric + '_std'], group[metric].std(ddof=1), atol=1e-10)
        tested += 2
result = dict(checks=tested, source_files_unchanged=len(hashes), passed=True,
              independent_aggregation=True, new_training=False)
(OUT / 'independent_verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(result)
