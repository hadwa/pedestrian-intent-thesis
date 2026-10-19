import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def run():
    path = Path(__file__).with_name('run_experiment.py')
    assert path.exists(), 'runner not implemented'
    spec = importlib.util.spec_from_file_location('test102', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_feature_arms(run):
    names = run.contract()
    bb, v1, v2 = np.arange(106).reshape(2, 53), np.arange(144).reshape(2, 72), np.arange(126).reshape(2, 63)
    arms = run.assemble(bb, v1, v2)
    for arm, dim in zip(run.ARMS, (53, 125, 109, 116, 188)):
        assert arms[arm].shape == (2, dim)
        assert len(names[arm]) == len(set(names[arm])) == dim
        np.testing.assert_array_equal(arms[arm][:, :53], bb)
    np.testing.assert_array_equal(arms['E'][:, :125], arms['B'])
    np.testing.assert_array_equal(arms['E'][:, 125:], v2)
    for i, name in enumerate(names['C']):
        np.testing.assert_array_equal(arms['C'][:, i], arms['D'][:, names['D'].index(name)])


@pytest.mark.parametrize('shape', [(2, 62), (3, 63)])
def test_invalid_schema(run, shape):
    with pytest.raises(ValueError):
        run.assemble(np.zeros((2, 53)), np.zeros((2, 72)), np.zeros(shape))


def test_nonfinite(run):
    with pytest.raises(ValueError):
        run.assemble(np.zeros((2, 53)), np.zeros((2, 72)), np.full((2, 63), np.nan))


def test_complete_grid(run):
    t = pd.DataFrame([dict(cohort=c, arm=a, seed=s, threshold=.5, f1=.5, pr_auc=.6, roc_auc=.7)
                      for c in run.COHORTS for a in run.ARMS for s in run.SEEDS])
    run.check_grid(t)
    with pytest.raises(ValueError):
        run.check_grid(t.iloc[:-1])
    t.loc[0, 'threshold'] = .6
    with pytest.raises(ValueError):
        run.check_grid(t)


def test_test_gate(run, tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    with pytest.raises(RuntimeError, match='validate'):
        run.load_split('test')


def test_hash_guard(run, tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    p = tmp_path / 'artifact'
    p.write_text('first')
    run.publish('unit', [p])
    run.authenticate('unit')
    p.write_text('changed')
    with pytest.raises(RuntimeError):
        run.authenticate('unit')


def test_identity_does_not_depend_on_cohort_prefix(run):
    row = dict(video_id='video_1', ped_id='p1', observed_frames_json=str(list(range(16))), label=0)
    a = pd.DataFrame([dict(row, sequence_id='all:1')])
    b = pd.DataFrame([dict(row, sequence_id='beh:1')])
    assert run.identities(a) == run.identities(b)
    b.loc[0, 'observed_frames_json'] = str(list(range(1, 17)))
    assert run.identities(a) != run.identities(b)


def test_api_tte_is_track_index_distance_not_raw_frame_distance(run):
    rows = run.manifest('jaad_beh', 'train')
    assert len(rows) == 2134
    assert (rows.api_track_length - 1 - rows.api_window_end_index == rows.tte_frames).all()
    assert int(((rows.event_frame - rows.end_frame) != rows.tte_frames).sum()) == 33
