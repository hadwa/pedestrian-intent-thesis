import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def runner():
    path = Path(__file__).with_name('run_experiment.py')
    assert path.exists(), 'Exp098 runner has not been implemented'
    spec = importlib.util.spec_from_file_location('exp098_test_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_arm_blocks(runner):
    bbox = np.arange(106).reshape(2, 53)
    v1 = np.arange(144).reshape(2, 72)
    v2 = np.arange(126).reshape(2, 63)
    arms = runner.build_arms(bbox, v1, v2)
    assert {k: x.shape[1] for k, x in arms.items()} == dict(A=53, B=125, C=116, D=188)
    np.testing.assert_array_equal(arms['B'][:, :53], bbox)
    np.testing.assert_array_equal(arms['C'][:, 53:], v2)
    np.testing.assert_array_equal(arms['D'][:, 53:125], v1)


def test_bad_feature_dimensions_fail(runner):
    with pytest.raises(ValueError):
        runner.build_arms(np.zeros((2, 53)), np.zeros((2, 72)), np.zeros((2, 56)))
    with pytest.raises(ValueError):
        runner.build_arms(np.zeros((2, 53)), np.zeros((3, 72)), np.zeros((2, 63)))


def test_nonfinite_features_fail(runner):
    with pytest.raises(ValueError):
        runner.build_arms(np.full((2, 53), np.nan), np.zeros((2, 72)), np.zeros((2, 63)))


def test_row_alignment_and_label_guard(runner):
    a = pd.DataFrame(dict(sequence_id=['a', 'b'], label=[0, 1], start_frame=[0, 2],
                          end_frame=[15, 17], video_id=['v', 'v'], ped_id=['p', 'q']))
    b = a.iloc[::-1].copy()
    assert runner.align_rows(a, b).sequence_id.tolist() == ['a', 'b']
    b.loc[b.sequence_id == 'a', 'label'] = 1
    with pytest.raises(ValueError):
        runner.align_rows(a, b)


def test_duplicate_or_missing_rows_fail(runner):
    a = pd.DataFrame(dict(sequence_id=['a', 'b'], label=[0, 1], start_frame=[0, 2],
                          end_frame=[15, 17], video_id=['v', 'v'], ped_id=['p', 'q']))
    for b in (a.iloc[:1], pd.concat([a, a.iloc[:1]])):
        with pytest.raises(ValueError):
            runner.align_rows(a, b)


def test_gate_requires_complete_grid(runner):
    rows = pd.DataFrame([dict(arm=a, seed=s, threshold=.5, f1=.1, pr_auc=.2, roc_auc=.3)
                         for a in 'ABCD' for s in (42, 43, 44)])
    runner.check_grid(rows)
    with pytest.raises(ValueError):
        runner.check_grid(rows.iloc[:-1])
    rows.loc[0, 'threshold'] = .4
    with pytest.raises(ValueError):
        runner.check_grid(rows)


def test_test_source_locked_before_validation(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    with pytest.raises(RuntimeError, match='validate'):
        runner.load_split('test')


def test_manifest_detects_tampering(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    file = tmp_path / 'output.txt'
    file.write_text('one')
    runner.publish('unit', [file])
    runner.authenticate('unit')
    file.write_text('two')
    with pytest.raises(RuntimeError):
        runner.authenticate('unit')


def test_bootstrap_zero_for_identical_predictions(runner):
    rows = pd.DataFrame(dict(label=[0, 1, 0, 1], track_id=['a', 'a', 'b', 'b']))
    p = np.array([.1, .8, .7, .6])
    result = runner.bootstrap(rows, {'A': p, 'B': p, 'C': p, 'D': p}, replicates=20)
    assert (result[['delta', 'ci_low', 'ci_high']] == 0).all().all()


def test_frozen_estimator_recipe_and_determinism(runner):
    x = np.random.default_rng(42).normal(size=(30, 4))
    y = np.array([0, 1] * 15)
    first = runner.core.build_candidate(42, y)
    second = runner.bbox_core.build_xgb(42, y, n_estimators=250)
    p = first.named_steps['model'].get_params()
    q = second.named_steps['model'].get_params()
    for name in ('n_estimators', 'max_depth', 'learning_rate', 'subsample',
                 'colsample_bytree', 'reg_lambda', 'n_jobs', 'scale_pos_weight'):
        assert p[name] == q[name]
    first.fit(x, y)
    second.fit(x, y)
    np.testing.assert_array_equal(first.predict_proba(x), second.predict_proba(x))


def test_video_leakage_fails(runner):
    a = pd.DataFrame(dict(sequence_id=['a'], video_id=['same'], track_id=['p']))
    b = pd.DataFrame(dict(sequence_id=['b'], video_id=['same'], track_id=['q']))
    with pytest.raises(ValueError, match='video_id'):
        runner.split_disjoint(a, b)
