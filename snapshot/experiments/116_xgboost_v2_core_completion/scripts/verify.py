"""Independent saved-artifact verification; run after Exp116 completes."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             roc_auc_score, accuracy_score)

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    completed = json.loads((ROOT / 'EXPERIMENT_COMPLETE.json').read_text())
    assert completed['fits'] == 33 and completed['cases'] == 11
    for p, digest in completed['artifacts'].items():
        assert sha(ROOT / p) == digest, p
    trained = json.loads((ROOT / 'TRAINED.json').read_text())
    for p, digest in trained['artifacts'].items():
        assert sha(Path(p)) == digest, p
    frame = pd.read_csv(ROOT / 'metrics/per_seed_results.csv')
    assert len(frame) == 132 and frame.threshold.eq(.5).all()
    assert not frame.duplicated(['case', 'split', 'arm', 'seed']).any()
    counts = []
    for (case, split), block in frame.groupby(['case', 'split']):
        ids = labels = None
        for row in block[block.arm == 'core56'].itertuples(index=False):
            pred = pd.read_csv(ROOT / f'predictions/{case}_{split}_seed{row.seed}.csv')
            assert not pred.sequence_id.duplicated().any()
            assert pred.threshold.eq(.5).all()
            np.testing.assert_array_equal(pred.prediction, (pred.probability >= .5).astype(int))
            if ids is not None:
                np.testing.assert_array_equal(ids, pred.sequence_id)
                np.testing.assert_array_equal(labels, pred.label)
            ids, labels = pred.sequence_id, pred.label
            checks = dict(f1=f1_score(labels, pred.prediction),
                          accuracy=accuracy_score(labels, pred.prediction),
                          pr_auc=average_precision_score(labels, pred.probability),
                          roc_auc=roc_auc_score(labels, pred.probability))
            for name, val in checks.items():
                np.testing.assert_allclose(getattr(row, name), val, atol=1e-12, rtol=0)
            tn, fp, fn, tp = confusion_matrix(labels, pred.prediction, labels=[0, 1]).ravel()
            assert (tn, fp, fn, tp) == (row.tn, row.fp, row.fn, row.tp)
        counts.append(dict(case=case, split=split, windows=len(ids), positives=int(labels.sum())))
    deltas = pd.read_csv(ROOT / 'metrics/paired_deltas.csv')
    for row in deltas.itertuples(index=False):
        b = frame[(frame.case == row.case) & (frame.split == row.split) & (frame.seed == row.seed)].set_index('arm')
        for m in ('f1', 'pr_auc', 'roc_auc'):
            np.testing.assert_allclose(getattr(row, m), b.loc['core56', m] - b.loc['parent', m], atol=1e-12, rtol=0)
    summary = pd.read_csv(ROOT / 'metrics/summary_results.csv')
    for row in summary.itertuples(index=False):
        b = frame[(frame.case == row.case) & (frame.split == row.split) & (frame.arm == row.arm)]
        assert set(b.seed) == {42, 43, 44}
        for m in ('f1', 'pr_auc', 'roc_auc'):
            for agg in ('mean', 'std'):
                np.testing.assert_allclose(getattr(row, m + '_' + agg), getattr(b[m], agg)(), atol=1e-12, rtol=0)
    pd.DataFrame(counts).to_csv(ROOT / 'metrics/verified_populations.csv', index=False)
    report = dict(status='PASS', models=33, saved_prediction_files=66, matched_parent_replays=66,
                  completion_sha256=sha(ROOT / 'EXPERIMENT_COMPLETE.json'),
                  checks=['artifact hashes', 'prediction/label alignment across seeds',
                          'fixed-0.5 decisions', 'confusion matrices', 'F1/AP/ROC/accuracy',
                          'paired deltas', 'means and sample standard deviations'])
    (ROOT / 'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
