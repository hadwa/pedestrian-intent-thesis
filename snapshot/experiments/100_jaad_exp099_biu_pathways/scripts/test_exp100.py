import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def run():
    path = Path(__file__).with_name('run_experiment.py')
    assert path.exists(), 'Exp100 implementation missing'
    spec = importlib.util.spec_from_file_location('exp100_test', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_pathway_dimensions_and_no_cross_contamination(run):
    names = run.contract()
    assert {k: len(v) for k, v in names.items()} == dict(v1=149, v2=145, combined=197)
    assert not any('v2__' in c for c in names['v1'])
    assert not any('v1__' in c for c in names['v2'])
    assert sum(c.startswith('vehicle_') for c in names['v1']) == 15
    assert sum(c.startswith('vehicle_') for c in names['v2']) == 15
    assert not any('temporal__lead_ankle' in c for c in names['v1'])
    assert any('temporal__lead_ankle' in c for c in names['v2'])
    assert 'v2__direction_sign' not in names['v2']


def test_full_v2_overlapping_values_are_not_v1_values(run):
    frame = pd.DataFrame(np.zeros((2, 197)), columns=run.parent.contract()['B'])
    v1 = run.families()['biu_v1_only']
    v2 = run.families()['biu_v2_core']
    for name in v1:
        frame[name] = 10.
    restored = np.full((2, 56), 20.)
    arms = run.build_arms(frame, restored)
    names = run.contract()
    overlapping = sorted(set(v1) & set(v2))[0]
    assert (arms['v1'][:, names['v1'].index('v1__' + overlapping)] == 10).all()
    assert (arms['v2'][:, names['v2'].index('v2__' + overlapping)] == 20).all()
    np.testing.assert_array_equal(arms['combined'], frame.to_numpy())


def test_bad_recovered_features_fail(run):
    frame = pd.DataFrame(np.zeros((2, 197)), columns=run.parent.contract()['B'])
    for values in (np.zeros((2, 63)), np.full((2, 56), np.nan)):
        with pytest.raises(ValueError):
            run.build_arms(frame, values)


def test_gate_requires_all_nine_rows(run):
    rows = pd.DataFrame([dict(arm=a, seed=s, threshold=.5, f1=.5, pr_auc=.6, roc_auc=.7)
                         for a in run.ARMS for s in run.SEEDS])
    run.check_grid(rows)
    with pytest.raises(ValueError):
        run.check_grid(rows.iloc[:-1])


def test_test_load_locked(run, tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    with pytest.raises(RuntimeError, match='validate'):
        run.load_split('test')


def test_hash_tampering_fails(run, tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    path = tmp_path / 'artifact'
    path.write_text('a')
    run.publish('unit', [path])
    run.authenticate('unit')
    path.write_text('b')
    with pytest.raises(RuntimeError):
        run.authenticate('unit')
