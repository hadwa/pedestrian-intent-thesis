"""Independently verify saved validation metrics, models and frozen sources."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
frozen = json.loads((ROOT / 'metrics/FROZEN.json').read_text())
for path, expected in frozen['sources'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
assert hashlib.sha256((ROOT / 'DESIGN.md').read_bytes()).hexdigest() == frozen['design_sha']
assert hashlib.sha256((ROOT / 'scripts/run.py').read_bytes()).hexdigest() == frozen['runner_sha']
table = pd.read_csv(ROOT / 'metrics/per_seed_results.csv')
assert len(table) == 6 and set(table.seed) == {42, 43, 44}
identity = None
for row in table.itertuples():
    p = pd.read_csv(ROOT / f'predictions/{row.arm}_seed{row.seed}_validation.csv')
    assert len(p) == 3619 and set(p.set_id) == {'set04', 'set05'}
    assert not p.sequence_id.duplicated().any()
    current = p[['sequence_id', 'label', 'track_id', 'video_id']]
    if identity is None:
        identity = current
    pd.testing.assert_frame_equal(identity, current)
    assert np.array_equal(p.prediction, p.probability >= .5)
    expected = dict(f1=f1_score(p.label, p.prediction),
                    accuracy=accuracy_score(p.label, p.prediction),
                    pr_auc=average_precision_score(p.label, p.probability),
                    roc_auc=roc_auc_score(p.label, p.probability))
    for key, value in expected.items():
        assert abs(getattr(row, key) - value) < 1e-12
    assert tuple(confusion_matrix(p.label, p.prediction, labels=[0, 1]).ravel()) == (row.tn, row.fp, row.fn, row.tp)
    assert ((p.label == 0) & (p.probability >= .9)).sum() == row.fp_ge_0p9_count
    assert ((p.label == 1) & (p.probability <= .1)).sum() == row.fn_le_0p1_count
    if row.arm == 'ordered':
        model = joblib.load(ROOT / f'models/ordered_seed{row.seed}.joblib')
        assert list(model.feature_names_in_) == frozen['columns']
        assert model['model'].get_booster().num_boosted_rounds() == 250
        assert model['imputer'].statistics_.shape == (471,)
summary = pd.read_csv(ROOT / 'metrics/summary_results.csv').set_index('arm')
for arm, rows in table.groupby('arm'):
    for metric in ('f1', 'pr_auc', 'roc_auc', 'fp', 'fn'):
        assert abs(summary.loc[arm, metric + '_mean'] - rows[metric].mean()) < 1e-12
        assert abs(summary.loc[arm, metric + '_std'] - rows[metric].std()) < 1e-12
delta = pd.read_csv(ROOT / 'metrics/paired_deltas.csv').set_index('seed')
for metric in ('f1', 'pr_auc', 'roc_auc'):
    expected = table[table.arm == 'ordered'].set_index('seed')[metric] - table[table.arm == 'baseline'].set_index('seed')[metric]
    assert np.allclose(delta[metric], expected, rtol=0, atol=1e-12)
output = dict(status='PASS', prediction_files=6, new_models=3,
              train_rows=frozen['train_n'], validation_rows=3619,
              frozen_source_hashes_verified=True, test_accessed=False)
(ROOT / 'metrics/INDEPENDENT_VERIFICATION.json').write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps(output, indent=2))
