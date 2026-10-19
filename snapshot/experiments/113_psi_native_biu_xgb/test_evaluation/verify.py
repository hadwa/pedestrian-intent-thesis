"""Independently reproduce frozen test predictions and metrics; no fitting."""
import json
import joblib
import numpy as np
import pandas as pd
import evaluate as ev


def main():
    ev.authenticate()
    done = json.loads((ev.ROOT / 'EXPERIMENT_COMPLETE.json').read_text())
    for r in done['outputs']:
        assert ev.record(r['path']) == r
    for r in json.loads((ev.ROOT / 'freeze.json').read_text())['sources']:
        assert ev.record(r['path']) == r
    rows = pd.read_csv(ev.ROOT / 'cache/rows.csv')
    assert rows.sequence_id.is_unique
    for row in rows.itertuples(index=False):
        frames = json.loads(row.observed_frames_json)
        assert frames == list(range(row.target_frame - 15, row.target_frame))
        assert row.label == int(row.annotation_vote >= .5)
    table = pd.read_csv(ev.ROOT / 'metrics/per_seed_results.csv')
    assert len(table) == 9
    for row in table.itertuples(index=False):
        model = joblib.load(ev.EXP / f'models/{row.arm}_seed{row.seed}.joblib')
        x = np.load(ev.ROOT / f'cache/{row.arm}.npy')
        p = model.predict_proba(x)[:, 1]
        saved = pd.read_csv(ev.ROOT / f'predictions/test_{row.arm}_seed{row.seed}.csv')
        np.testing.assert_array_equal(saved.sequence_id, rows.sequence_id)
        np.testing.assert_array_equal(saved.label, rows.label)
        np.testing.assert_array_equal(saved.probability.to_numpy(np.float32), p)
        for name, value in ev.score(rows.label.to_numpy(), p).items():
            np.testing.assert_allclose(getattr(row, name), value, rtol=0, atol=1e-12)
    ev.write(ev.ROOT / 'VERIFIED.json', dict(prediction_arrays_reproduced=9,
        metrics_reproduced=True, frozen_sources_unchanged=True, no_refit=True,
        observation_frames_verified=True))
    print('PASS: nine test predictions/metrics, source hashes and observation horizons.')


if __name__ == '__main__':
    main()
