#!/usr/bin/env python3
"""Frozen, resumable, train-only JAAD BBox/BIU XGBoost ablation."""

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score
import xgboost


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
EXP089 = REPO / 'experiments/089_pedcmt_standardized_jaad_biu_transfer'
EXP077 = REPO / 'experiments/077_jaad_two_stage_early_late_xgb_transfer'
CORE_PATH = REPO / 'experiments/072_pie_joint_biu_temporal_ego_confirmation/scripts/model_core.py'
BBOX_PATH = REPO / 'experiments/076_pie_two_stage_early_late_xgb/scripts/two_stage_core.py'
DEFINITIONS = REPO / 'experiments/012_pie_biu_v2_30_60_benchmark/metrics/pie_biu_v2_feature_set_definitions.csv'
SPLITS = Path('/home/abdelrahman/Hadwa/datasets/JAAD/split_ids/default')
ARMS = ('A', 'B', 'C', 'D')
SEEDS = (42, 43, 44)
PAIRS = (('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'C'), ('B', 'D'), ('C', 'D'))
METRICS = ('accuracy', 'precision', 'recall', 'f1', 'pr_auc', 'roc_auc', 'fp', 'fn',
           'fpr', 'fnr', 'brier', 'fp_ge_0p9_count', 'fn_le_0p1_count')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


core = module('exp098_metrics_core', CORE_PATH)
bbox_core = module('exp098_bbox_core', BBOX_PATH)


def record(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    if isinstance(value, pd.DataFrame):
        value.to_csv(temp, index=False)
    elif isinstance(value, str):
        temp.write_text(value)
    else:
        temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temp, path)


def publish(phase, paths):
    write(ROOT / f'{phase.upper()}_COMPLETE.json', [record(path) for path in paths])


def authenticate(phase):
    marker = ROOT / f'{phase.upper()}_COMPLETE.json'
    if not marker.exists():
        raise RuntimeError(f'missing {phase} completion')
    records = json.loads(marker.read_text())
    if not records:
        raise RuntimeError(f'empty {phase} completion')
    for expected in records:
        if record(expected['path']) != expected:
            raise RuntimeError(f'{phase} artifact changed: {expected["path"]}')


def build_arms(bbox, v1, v2):
    for x, size in ((bbox, 53), (v1, 72), (v2, 63)):
        if x.ndim != 2 or x.shape != (len(bbox), size) or not np.isfinite(x).all():
            raise ValueError('nonfinite, unaligned or incorrect feature dimensions')
    return dict(A=bbox, B=np.column_stack((bbox, v1)), C=np.column_stack((bbox, v2)),
                D=np.column_stack((bbox, v1, v2)))


def align_rows(source, other):
    if (source.sequence_id.duplicated().any() or other.sequence_id.duplicated().any()
            or set(source.sequence_id) != set(other.sequence_id)):
        raise ValueError('missing or duplicate sample IDs')
    aligned = other.set_index('sequence_id').loc[source.sequence_id].reset_index()
    for name in ('sequence_id', 'label', 'start_frame', 'end_frame', 'video_id', 'ped_id'):
        if not np.array_equal(source[name].to_numpy(), aligned[name].to_numpy()):
            raise ValueError(f'cohort mismatch: {name}')
    return aligned


def check_grid(rows):
    if (len(rows) != 12 or set(zip(rows.arm, rows.seed)) != set(itertools.product(ARMS, SEEDS))
            or not rows.threshold.eq(.5).all()
            or not np.isfinite(rows[['f1', 'pr_auc', 'roc_auc']].to_numpy()).all()):
        raise ValueError('complete fixed-0.5 twelve-run grid required')


def sources(split):
    cache = EXP089 / 'cache' / ('test_source' if split == 'test' else 'trainval_source')
    return cache, EXP077 / 'metrics' / f'prepared_jaad_{split}.csv'


def load_split(split):
    if split == 'test':
        authenticate('validate')
    cache, parent_path = sources(split)
    metadata = json.loads((cache / 'metadata.json').read_text())
    rows_all = pd.read_csv(cache / 'rows.csv')
    mask = rows_all.split.eq('val' if split == 'validation' else split).to_numpy()
    rows = rows_all.loc[mask].reset_index(drop=True)
    expected = {'train': (1552, 999), 'validation': (274, 170), 'test': (1501, 720)}[split]
    if (len(rows), int(rows.label.sum())) != expected or set(rows.label.unique()) != {0, 1}:
        raise ValueError('population or labels changed')
    official = set((SPLITS / f'{"val" if split == "validation" else split}.txt').read_text().split())
    if not set(rows.video_id).issubset(official):
        raise ValueError('video outside official split')
    frames = np.array([json.loads(value) for value in rows.observed_frames_json])
    if (frames.shape != (len(rows), 16) or not (np.diff(frames, axis=1) == 1).all()
            or not np.array_equal(frames[:, 0], rows.start_frame)
            or not np.array_equal(frames[:, -1], rows.end_frame)
            or not rows.tte_frames.between(30, 60).all()
            or not np.array_equal(rows.event_frame - rows.end_frame, rows.tte_frames)):
        raise ValueError('observation/TTE protocol changed')
    if any(any(json.loads(value)) for value in rows.observed_crosses_json):
        raise ValueError('crossing visible in observation window')
    parent = align_rows(rows, pd.read_csv(parent_path))
    definitions = pd.read_csv(DEFINITIONS).set_index('feature_set')
    names = json.loads(definitions.loc['bbox_only', 'feature_columns_json'])
    v1_names = metadata['biu_v1_feature_names']
    v2_names = metadata['biu_v2_feature_names']
    if len(names) != 53 or len(v1_names) != 72 or len(v2_names) != 63:
        raise ValueError('feature name contract changed')
    bbox = parent[names].to_numpy(float)
    rebuilt_records = [bbox_core._bbox53(np.array(json.loads(value))) for value in rows.bbox_sequence_json]
    rebuilt = np.array([[values[name] for name in names] for values in rebuilt_records])
    np.testing.assert_allclose(bbox, rebuilt, rtol=1e-9, atol=1e-8)
    # Only observation BIUs are loaded. ego_speed, endpoint and future-box arrays are never inputs.
    arrays = [np.load(cache / f'biu_{version}.npy', allow_pickle=False) for version in ('v1', 'v2')]
    if any(len(x) != len(rows_all) for x in arrays):
        raise ValueError('cache row count mismatch')
    v1, v2 = [x[mask].astype(float) for x in arrays]
    arms = build_arms(bbox, v1, v2)
    feature_names = dict(A=names, B=names + ['v1__' + n for n in v1_names],
                         C=names + ['v2__' + n for n in v2_names],
                         D=names + ['v1__' + n for n in v1_names] + ['v2__' + n for n in v2_names])
    source_files = [parent_path, cache / 'metadata.json', cache / 'rows.csv',
                    cache / 'biu_v1.npy', cache / 'biu_v2.npy',
                    SPLITS / f'{"val" if split == "validation" else split}.txt']
    audit = dict(split=split, rows=len(rows), positives=int(rows.label.sum()),
                 tracks=int(rows.track_id.nunique()), videos=int(rows.video_id.nunique()),
                 sources=[record(p) for p in source_files], bbox_reconstruction_passed=True,
                 no_missing_or_dropped_rows=True, no_observed_crossing=True,
                 cached_pose_primary_sha=metadata['primary_pose_sha256'],
                 cached_pose_supplement_sha=metadata['supplement_pose_sha256'])
    return rows, arms, feature_names, audit


def split_disjoint(*populations):
    for a, b in itertools.combinations(populations, 2):
        for field in ('sequence_id', 'video_id', 'track_id'):
            if set(a[field]) & set(b[field]):
                raise ValueError(f'split leakage in {field}')


def summarize(table):
    return table.groupby(['scope', 'arm', 'feature_count'], sort=False).agg(
        **{f'{m}_{s}': (m, s) for m in METRICS for s in ('mean', 'std')}).reset_index()


def predictions(rows, p, arm, seed):
    out = rows[['sequence_id', 'video_id', 'ped_id', 'track_id', 'label',
                'start_frame', 'end_frame', 'tte_frames', 'observed_frames_json']].copy()
    out['arm'], out['seed'], out['threshold'] = arm, seed, .5
    out['probability'], out['prediction'] = p, (p >= .5).astype(int)
    return out


def prepare():
    if (ROOT / 'PREPARE_COMPLETE.json').exists():
        authenticate('prepare')
        return
    train, _, names, a = load_split('train')
    val, _, names_val, b = load_split('validation')
    assert names == names_val
    split_disjoint(train, val)
    source_paths = [ROOT / 'DESIGN.md', Path(__file__), Path(__file__).with_name('test_exp098.py'),
                    CORE_PATH, BBOX_PATH, DEFINITIONS,
                    REPO / 'src/biu_v2_extractor.py', REPO / 'src/features/biu_features.py',
                    EXP089 / 'scripts/exp089_data.py']
    source_paths += [Path(s['path']) for audit in (a, b) for s in audit['sources']]
    recipe = core.build_candidate(42, train.label.to_numpy()).named_steps['model'].get_params()
    recipe = {k: ('NaN (default missing sentinel)' if isinstance(v, float) and np.isnan(v) else v)
              for k, v in recipe.items()}
    freeze = dict(experiment=98, source_records=[record(p) for p in dict.fromkeys(source_paths)],
                  versions=dict(python=sys.version, numpy=np.__version__,
                                sklearn=sklearn.__version__, xgboost=xgboost.__version__),
                  recipe=recipe,
                  test_accessed=False, classification='retrospective matched ablation')
    paths = [ROOT / 'metrics' / name for name in ('source_audit.json', 'feature_contract.json', 'freeze.json')]
    write(paths[0], [a, b])
    write(paths[1], names)
    write(paths[2], freeze)
    publish('prepare', paths + list(dict.fromkeys(source_paths)))


def validate():
    authenticate('prepare')
    if (ROOT / 'VALIDATE_COMPLETE.json').exists():
        authenticate('validate')
        return
    train, xt, _, _ = load_split('train')
    val, xv, _, _ = load_split('validation')
    all_metrics, outputs = [], []
    for arm, seed in itertools.product(ARMS, SEEDS):
        run = f'{arm}_seed{seed}'
        marker = ROOT / f'{run.upper()}_COMPLETE.json'
        model_path = ROOT / 'models' / f'{run}.joblib'
        prediction_path = ROOT / 'predictions' / f'validation_{run}.csv'
        metric_path = ROOT / 'metrics' / f'validation_{run}.json'
        if marker.exists():
            authenticate(run)
            metric = json.loads(metric_path.read_text())
        else:
            start = time.perf_counter()
            model = core.build_candidate(seed, train.label.to_numpy())
            model.fit(xt[arm], train.label.to_numpy())
            p = model.predict_proba(xv[arm])[:, 1]
            metric = dict(scope='validation', arm=arm, seed=seed, feature_count=xt[arm].shape[1],
                          fit_seconds=time.perf_counter() - start,
                          **core.compute_metrics(val.label.to_numpy(), p))
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, str(model_path) + '.tmp')
            os.replace(str(model_path) + '.tmp', model_path)
            write(prediction_path, predictions(val, p, arm, seed))
            write(metric_path, metric)
            publish(run, [model_path, prediction_path, metric_path])
        all_metrics.append(metric)
        outputs += [model_path, prediction_path, metric_path, marker]
        print(f'{run}: validation saved ({metric["fit_seconds"]:.2f}s)', flush=True)
    table = pd.DataFrame(all_metrics)
    check_grid(table)
    authenticate('prepare')
    for name, data in [('validation_per_seed_results.csv', table),
                       ('validation_summary_results.csv', summarize(table))]:
        path = ROOT / 'metrics' / name
        write(path, data)
        outputs.append(path)
    publish('validate', outputs)


def bootstrap(rows, probabilities, replicates=5000):
    y = rows.label.to_numpy()
    codes, groups = pd.factorize(rows.track_id, sort=True)
    rng = np.random.default_rng(42)

    def scores(p, weights):
        pred = p >= .5
        tp = np.sum(weights * ((y == 1) & pred))
        fp = np.sum(weights * ((y == 0) & pred))
        fn = np.sum(weights * ((y == 1) & ~pred))
        return np.array([2 * tp / max(2 * tp + fp + fn, 1),
                         average_precision_score(y, p, sample_weight=weights),
                         roc_auc_score(y, p, sample_weight=weights)])

    point = {a: scores(p, np.ones(len(y))) for a, p in probabilities.items()}
    draws = {pair: [] for pair in PAIRS}
    for index in range(replicates):
        counts = np.bincount(rng.integers(len(groups), size=len(groups)), minlength=len(groups))
        weights = counts[codes]
        if not all(np.sum(weights[y == label]) > 0 for label in (0, 1)):
            continue
        values = {a: scores(p, weights) for a, p in probabilities.items()}
        for left, right in PAIRS:
            draws[(left, right)].append(values[right] - values[left])
        if replicates >= 1000 and (index + 1) % 1000 == 0:
            print(f'track bootstrap {index + 1}/{replicates}', flush=True)
    result = []
    for left, right in PAIRS:
        values = np.array(draws[(left, right)])
        for index, metric in enumerate(('f1', 'pr_auc', 'roc_auc')):
            result.append(dict(comparison=f'{right}_minus_{left}', metric=metric,
                               delta=float((point[right] - point[left])[index]),
                               ci_low=float(np.quantile(values[:, index], .025)),
                               ci_high=float(np.quantile(values[:, index], .975)),
                               replicates=len(values), requested_replicates=replicates, bootstrap_seed=42))
    return pd.DataFrame(result)


def evaluate():
    authenticate('prepare')
    authenticate('validate')
    if (ROOT / 'EVALUATE_COMPLETE.json').exists():
        authenticate('evaluate')
        return
    train, _, _, _ = load_split('train')
    val, _, _, _ = load_split('validation')
    test, xt, names, audit = load_split('test')
    split_disjoint(train, val, test)
    if names != json.loads((ROOT / 'metrics/feature_contract.json').read_text()):
        raise RuntimeError('test schema differs from frozen training schema')
    outputs, metrics, parts = [], [], []
    for arm, seed in itertools.product(ARMS, SEEDS):
        model = joblib.load(ROOT / 'models' / f'{arm}_seed{seed}.joblib')
        p = model.predict_proba(xt[arm])[:, 1]
        metrics.append(dict(scope='test', arm=arm, seed=seed, feature_count=xt[arm].shape[1],
                            **core.compute_metrics(test.label.to_numpy(), p)))
        part = predictions(test, p, arm, seed)
        path = ROOT / 'predictions' / f'test_{arm}_seed{seed}.csv'
        if path.exists():
            saved = pd.read_csv(path)
            np.testing.assert_allclose(saved.probability, p, rtol=0, atol=1e-15)
            align_rows(part, saved)
        else:
            write(path, part)
        outputs.append(path)
        parts.append(part)
    authenticate('validate')
    table = pd.DataFrame(metrics)
    check_grid(table)
    ensemble = {a: np.mean([p.probability.to_numpy() for p in parts if p.arm.iloc[0] == a], axis=0)
                for a in ARMS}
    intervals = bootstrap(test, ensemble)
    deltas = []
    indexed = table.set_index(['arm', 'seed'])
    for left, right in PAIRS:
        for seed in SEEDS:
            deltas.append(dict(comparison=f'{right}_minus_{left}', seed=seed,
                               **{f'delta_{m}': float(indexed.loc[(right, seed), m] - indexed.loc[(left, seed), m])
                                  for m in METRICS}))
    transitions = []
    for seed in SEEDS:
        left, right = [next(p for p in parts if p.arm.iloc[0] == a and p.seed.iloc[0] == seed) for a in 'BC']
        y = left.label.to_numpy()
        b, c = left.prediction.to_numpy(), right.prediction.to_numpy()
        transitions.append(dict(seed=seed, B_FP_to_C_TN=int(((y == 0) & (b == 1) & (c == 0)).sum()),
                                B_FN_to_C_TP=int(((y == 1) & (b == 0) & (c == 1)).sum()),
                                B_TN_to_C_FP=int(((y == 0) & (b == 0) & (c == 1)).sum()),
                                B_TP_to_C_FN=int(((y == 1) & (b == 1) & (c == 0)).sum())))
    data = {'test_source_audit.json': audit, 'per_seed_results.csv': table,
            'fixed_0p5_results.csv': table, 'summary_results.csv': summarize(table),
            'paired_deltas.csv': pd.DataFrame(deltas), 'ensemble_track_bootstrap.csv': intervals,
            'error_transitions_fixed_0p5.csv': pd.DataFrame(transitions)}
    for name, value in data.items():
        path = ROOT / 'metrics' / name
        write(path, value)
        outputs.append(path)
    publish('evaluate', outputs + [Path(s['path']) for s in audit['sources']])


def report():
    authenticate('evaluate')
    summary = pd.read_csv(ROOT / 'metrics/summary_results.csv')
    lines = ['# Exp098 Final Report', '', 'Retrospective matched clean-JAAD ablation; not untouched confirmation.',
             '16 observed frames, TTE 30--60, fixed 0.5; identical cohorts and XGBoost recipe.',
             'No vehicle-action, ego, road or extra temporal inputs. No pose re-extraction.', '',
             '| Arm | Features | F1 mean +/- SD | PR-AUC mean +/- SD | ROC-AUC mean +/- SD |',
             '|---|---:|---:|---:|---:|']
    for row in summary.to_dict('records'):
        cells = [row['arm'], str(row['feature_count'])] + [f'{row[m + "_mean"]:.6f} +/- {row[m + "_std"]:.6f}'
                                                        for m in ('f1', 'pr_auc', 'roc_auc')]
        lines.append('| ' + ' | '.join(cells) + ' |')
    delta = pd.read_csv(ROOT / 'metrics/paired_deltas.csv')
    lines += ['', '## C (v2) minus B (v1)', '']
    for metric in ('f1', 'pr_auc', 'roc_auc'):
        values = delta.loc[delta.comparison == 'C_minus_B', 'delta_' + metric]
        lines.append(f'- {metric}: mean delta {values.mean():+.6f}; favorable seeds {int((values > 0).sum())}/3.')
    lines += ['', 'Track-bootstrap uncertainty is in metrics/ensemble_track_bootstrap.csv; it uses',
              'seed-ensemble probabilities, not mean per-seed metrics. Intervals are descriptive',
              'and are not multiplicity-adjusted. Per-seed evidence and B-to-C error transitions',
              'are reported separately. Do not infer universal version superiority.', '',
              'Source arrays and historical code were reused read-only. Twelve new train-only',
              'fits used cached features. All twelve validation models were frozen before test',
              'access, and saved models were evaluated without refitting. Existing 72-D v1',
              'validity counters and 63-D v2 direction metadata remain intact. See DESIGN.md',
              'and metrics/freeze.json for the exact experimental and provenance boundaries.', '']
    path = ROOT / 'FINAL_REPORT.md'
    write(path, '\n'.join(lines))
    publish('report', [path])


def verify():
    for phase in ('prepare', 'validate', 'evaluate', 'report'):
        authenticate(phase)
    table = pd.read_csv(ROOT / 'metrics/per_seed_results.csv')
    check_grid(table)
    keys = None
    for arm, seed in itertools.product(ARMS, SEEDS):
        pred = pd.read_csv(ROOT / 'predictions' / f'test_{arm}_seed{seed}.csv')
        current = list(zip(pred.sequence_id, pred.label, pred.observed_frames_json))
        if keys is None:
            keys = current
        if current != keys or len(current) != 1501:
            raise RuntimeError('saved prediction cohorts differ')
        metric = core.compute_metrics(pred.label.to_numpy(), pred.probability.to_numpy())
        saved = table.loc[(table.arm == arm) & (table.seed == seed)].iloc[0]
        for name in METRICS:
            np.testing.assert_allclose(metric[name], saved[name], atol=1e-12, rtol=1e-12)
    write(ROOT / 'EXPERIMENT_COMPLETE.json', dict(status='complete', experiment=98, runs=12,
          test_rows=1501, no_refit=True, report=record(ROOT / 'FINAL_REPORT.md'),
          classification='retrospective matched ablation'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['all', 'prepare', 'validate', 'evaluate', 'report', 'verify'])
    args = parser.parse_args()
    log_path = ROOT / 'logs/progress.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'logs/commands.log').open('a') as log:
        log.write(time.strftime('%Y-%m-%dT%H:%M:%S%z') + ' ' + sys.executable + ' ' + ' '.join(sys.argv) + '\n')
    for phase in (['prepare', 'validate', 'evaluate', 'report', 'verify'] if args.phase == 'all' else [args.phase]):
        message = time.strftime('%Y-%m-%dT%H:%M:%S%z') + f' {phase} started'
        print(message, flush=True)
        with log_path.open('a') as log:
            log.write(message + '\n')
        globals()[phase]()
        message = time.strftime('%Y-%m-%dT%H:%M:%S%z') + f' {phase} complete'
        print(message, flush=True)
        with log_path.open('a') as log:
            log.write(message + '\n')


if __name__ == '__main__':
    main()
