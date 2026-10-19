import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def run():
    path = Path(__file__).with_name('run_experiment.py')
    assert path.exists(), 'Exp099 implementation missing'
    spec = importlib.util.spec_from_file_location('exp099_test', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_categorical_encoding(run):
    codes = np.array([[0] * 8 + [4] * 8, [2] * 16])
    out = run.encode_vehicle(codes / 4.0)
    assert out.shape == (2, 15)
    np.testing.assert_array_equal(out[0], [.5, 0, 0, 0, .5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    np.testing.assert_array_equal(out[1], [0, 0, 1, 0, 0] * 3)


@pytest.mark.parametrize('value', [-.25, .3, 1.25, np.nan])
def test_invalid_category_fails(run, value):
    with pytest.raises(ValueError):
        run.encode_vehicle(np.full((2, 16), value))


def test_incorrect_window_shape_fails(run):
    with pytest.raises(ValueError):
        run.encode_vehicle(np.zeros((2, 17)))


def test_no_nominal_order_assumption(run):
    first = run.encode_vehicle(np.full((1, 16), .75))
    second = run.encode_vehicle(np.full((1, 16), 1.0))
    assert set(np.unique(first)) == {0, 1}
    assert first.sum() == second.sum() == 3


def test_incomplete_validation_gate_fails(run):
    rows = pd.DataFrame([dict(arm=a, seed=s, threshold=.5, f1=.4, pr_auc=.5, roc_auc=.6)
                         for a in 'AB' for s in (42, 43, 44)])
    run.check_grid(rows)
    with pytest.raises(ValueError):
        run.check_grid(rows.iloc[:-1])


def test_test_access_locked(run, tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    with pytest.raises(RuntimeError, match='validate'):
        run.load_split('test')


def test_hash_guard(run, tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    source = tmp_path / 'x.txt'
    source.write_text('one')
    run.publish('unit', [source])
    run.authenticate('unit')
    source.write_text('two')
    with pytest.raises(RuntimeError):
        run.authenticate('unit')


def test_identical_bootstrap_is_zero(run):
    rows = pd.DataFrame(dict(label=[0, 1, 0, 1], track_id=['a', 'a', 'b', 'b']))
    p = np.array([.1, .8, .6, .7])
    out = run.bootstrap(rows, p, p, replicates=20)
    assert (out[['delta', 'ci_low', 'ci_high']] == 0).all().all()
