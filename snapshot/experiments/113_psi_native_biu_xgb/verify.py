"""Independent saved-model/metric checks; does not train or open test data."""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import run


def main():
    root = run.ROOT
    freeze = json.loads((root / 'freeze.json').read_text())
    for r in freeze['sources']:
        assert run.record(r['path']) == r
    done = json.loads((root / 'EXPERIMENT_COMPLETE.json').read_text())
    assert done['validation_only'] and not done['test_accessed'] and done['fits'] == 9
    for r in done['outputs']:
        assert run.record(r['path']) == r
    table = pd.read_csv(root / 'metrics/per_seed_results.csv')
    train = pd.read_csv(root / 'cache/train/rows.csv')
    val = pd.read_csv(root / 'cache/val/rows.csv')
    assert not set(train.video_id) & set(val.video_id)
    for rows in (train, val):
        assert rows.sequence_id.is_unique
        for r in rows.itertuples(index=False):
            frames = json.loads(r.observed_frames_json)
            assert len(frames) == 15 and frames == list(range(frames[0], frames[0] + 15))
            assert r.target_frame == frames[-1] + 1
            assert r.label == int(r.annotation_vote >= .5)
    for row in table.itertuples(index=False):
        x = np.load(root / f'cache/val/{row.arm}.npy')
        xt = np.load(root / f'cache/train/{row.arm}.npy')
        model = joblib.load(root / f'models/{row.arm}_seed{row.seed}.joblib')
        np.testing.assert_array_equal(model.named_steps['imputer'].statistics_, np.median(xt, axis=0))
        saved = pd.read_csv(root / f'predictions/validation_{row.arm}_seed{row.seed}.csv')
        np.testing.assert_array_equal(saved.sequence_id, val.sequence_id)
        np.testing.assert_array_equal(saved.label, val.label)
        p = model.predict_proba(x)[:, 1]
        np.testing.assert_array_equal(saved.probability.to_numpy(np.float32), p)
        metrics = run.helper.core.compute_metrics(val.label.to_numpy(), p)
        metrics['macro_f1'] = f1_score(val.label, p >= .5, average='macro')
        for name, value in metrics.items():
            np.testing.assert_allclose(getattr(row, name), value, rtol=0, atol=1e-12)
    for split in ('train', 'val'):
        a = np.load(root / f'cache/{split}/A.npy')
        for arm in ('B', 'C'):
            np.testing.assert_array_equal(a, np.load(root / f'cache/{split}/{arm}.npy')[:, :53])
    run.write(root / 'VERIFIED.json', dict(prediction_arrays_reproduced=9,
        metrics_reproduced=True, source_hashes_unchanged=True, common_bbox_verified=True,
        observation_horizon_verified=True, train_only_imputation_verified=True, test_accessed=False))
    print('PASS: nine predictions/metrics, source hashes, cohorts, horizons and train-only imputation.')


if __name__ == '__main__':
    main()
