"""Read-only result checks; only the verification report is newly written."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    marker = json.loads((ROOT / 'EXPERIMENT_COMPLETE.json').read_text())
    assert marker['models'] == 9 and marker['test_after_all_training']
    for path, digest in marker['artifacts'].items():
        assert sha(ROOT / path) == digest, path
    trained = json.loads((ROOT / 'TRAINED.json').read_text())
    assert len(trained['artifacts']) == 18
    for path, digest in trained['artifacts'].items():
        assert sha(Path(path)) == digest, path
    table = pd.read_csv(ROOT / 'metrics/per_seed_results.csv')
    assert not table.duplicated(['case', 'seed', 'split', 'arm']).any()
    assert table.threshold.eq(.5).all()
    new = table[table.arm == 'core56']
    assert len(new) == 18
    for row in new.itertuples(index=False):
        f = pd.read_csv(ROOT / f'predictions/{row.case}_{row.split}_seed{row.seed}.csv')
        np.testing.assert_array_equal(f.prediction, (f.probability >= .5).astype(int))
        assert len(f) == row.n and not f.sequence_id.duplicated().any()
        values = dict(accuracy=accuracy_score(f.label, f.prediction), f1=f1_score(f.label, f.prediction),
                      pr_auc=average_precision_score(f.label, f.probability), roc_auc=roc_auc_score(f.label, f.probability))
        for k, v in values.items():
            np.testing.assert_allclose(getattr(row, k), v, atol=1e-12, rtol=0)
        assert tuple(confusion_matrix(f.label, f.prediction, labels=[0, 1]).ravel()) == (row.tn, row.fp, row.fn, row.tp)
        if row.split != 'validation':
            continue
        variant = 'I' if row.case == 'mslstm_pie' else 'D'
        checkpoint = torch.load(ROOT / row.case / f'checkpoints/{variant}_seed{row.seed}.best.pt', map_location='cpu', weights_only=False)
        if row.case == 'mslstm_pie':
            history = pd.read_csv(ROOT / row.case / f'logs/training_history_I_seed{row.seed}.csv')
            assert len(history) == 15
            selected = history.sort_values(['val_f1', 'epoch'], ascending=[False, True]).iloc[0]
            assert selected.epoch == checkpoint['best_epoch']
            np.testing.assert_allclose(selected.val_f1, row.f1, atol=1e-12, rtol=0)
        else:
            history = pd.read_csv(ROOT / row.case / f'logs/D_seed{row.seed}.history.csv')
            assert len(history) <= 200
            selected = history.sort_values(['validation_classification_loss', 'epoch']).iloc[0]
            assert selected.epoch == checkpoint['epoch']
            np.testing.assert_allclose(selected.validation_f1, row.f1, atol=1e-12, rtol=0)
    summary = pd.read_csv(ROOT / 'metrics/summary_results.csv')
    for row in summary.itertuples(index=False):
        f = table[(table.case == row.case) & (table.split == row.split) & (table.arm == row.arm)]
        assert set(f.seed) == {42, 43, 44}
        for k in ('f1', 'pr_auc', 'roc_auc'):
            for agg in ('mean', 'std'):
                np.testing.assert_allclose(getattr(row, k + '_' + agg), getattr(f[k], agg)(), atol=1e-12, rtol=0)
    deltas = pd.read_csv(ROOT / 'metrics/paired_deltas.csv')
    for row in deltas.itertuples(index=False):
        f = table[(table.case == row.case) & (table.seed == row.seed) & (table.split == row.split)].set_index('arm')
        for k in ('f1', 'pr_auc', 'roc_auc'):
            np.testing.assert_allclose(getattr(row, k), f.loc['core56', k] - f.loc[row.reference, k], atol=1e-12, rtol=0)
    report = dict(status='PASS', models=9, fresh_prediction_files=18, fixed_threshold=.5,
                  completion_sha256=sha(ROOT / 'EXPERIMENT_COMPLETE.json'),
                  checks=['artifact hashes', 'fixed-0.5 probabilities and metrics', 'confusion matrices',
                          'seed-grid completeness', 'parent-specific best-checkpoint selection',
                          'means/sample SD', 'paired deltas'])
    (ROOT / 'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
