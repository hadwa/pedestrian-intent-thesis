import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def run():
    path = Path(__file__).with_name('run_experiment.py')
    assert path.exists(), 'Exp101 implementation missing'
    spec = importlib.util.spec_from_file_location('exp101_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_seven_features_appended(run):
    base = np.arange(290).reshape(2, 145)
    extra = np.arange(14).reshape(2, 7)
    full = run.append_metadata(base, extra)
    assert full.shape == (2, 152)
    np.testing.assert_array_equal(full[:, :145], base)
    np.testing.assert_array_equal(full[:, 145:], extra)
    columns = run.contract()
    assert columns['full63'][:145] == columns['core56']
    assert columns['full63'][145:] == ['v2__' + n for n in run.EXTRA]


@pytest.mark.parametrize('shape', [(2, 6), (2, 8), (3, 7)])
def test_invalid_extra_schema_fails(run, shape):
    with pytest.raises(ValueError):
        run.append_metadata(np.zeros((2, 145)), np.zeros(shape))


def test_nonfinite_metadata_fails(run):
    with pytest.raises(ValueError):
        run.append_metadata(np.zeros((2, 145)), np.full((2, 7), np.nan))


def test_extra_schema_exactly_matches_63_minus_56(run):
    assert set(run.EXTRA) == {'direction_sign', 'was_flipped', 'direction_confidence',
                             'shoulder_width_mean', 'shoulder_width_delta',
                             'torso_length_mean', 'torso_length_delta'}
    assert not set(run.EXTRA) & set(run.parent.families()['biu_v2_core'])


def test_gate_requires_all_six_rows(run):
    table = pd.DataFrame([dict(arm=a, seed=s, threshold=.5, f1=.5, pr_auc=.6, roc_auc=.7)
                          for a in run.ARMS for s in run.SEEDS])
    run.check_grid(table)
    with pytest.raises(ValueError):
        run.check_grid(table.iloc[:-1])


def test_test_locked_until_validation(run, tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    with pytest.raises(RuntimeError, match='validate'):
        run.load_split('test')


def test_tampering_detected(run, tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    path = tmp_path / 'data'
    path.write_text('first')
    run.publish('unit', [path])
    run.authenticate('unit')
    path.write_text('second')
    with pytest.raises(RuntimeError):
        run.authenticate('unit')
