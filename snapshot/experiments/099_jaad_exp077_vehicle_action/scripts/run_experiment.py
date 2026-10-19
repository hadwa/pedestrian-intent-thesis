#!/usr/bin/env python3
"""Frozen vehicle-action ablation with authenticated Exp077 baseline reuse."""

import argparse
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
EXP077 = REPO / 'experiments/077_jaad_two_stage_early_late_xgb_transfer'
EXP089 = REPO / 'experiments/089_pedcmt_standardized_jaad_biu_transfer'
HELPER = REPO / 'experiments/098_jaad_matched_biu_xgb/scripts/run_experiment.py'
spec = importlib.util.spec_from_file_location('exp099_existing_helpers', HELPER)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
core, record, write = helper.core, helper.record, helper.write
SEEDS = (42, 43, 44)
CATEGORIES = ('stopped', 'moving_slow', 'moving_fast', 'decelerating', 'accelerating')
VEHICLE = Path('/home/abdelrahman/Hadwa/datasets/JAAD/annotations_vehicle')
METRICS = helper.METRICS


def publish(phase, files):
    write(ROOT / f'{phase.upper()}_COMPLETE.json', [record(p) for p in dict.fromkeys(files)])


def authenticate(phase):
    path = ROOT / f'{phase.upper()}_COMPLETE.json'
    if not path.exists():
        raise RuntimeError(f'missing {phase} completion')
    records = json.loads(path.read_text())
    if not records:
        raise RuntimeError(f'empty {phase} completion')
    for expected in records:
        if record(expected['path']) != expected:
            raise RuntimeError(f'{phase} artifact changed: {expected["path"]}')


def parent_auth(path, phase):
    marker = EXP077 / 'metrics' / f'.{phase}.complete.json'
    records = json.loads(marker.read_text())['outputs']
    matches = [r for r in records if Path(r['path']).resolve() == path.resolve()]
    if len(matches) != 1 or record(path)['sha256'] != matches[0]['sha256']:
        raise RuntimeError(f'Exp077 {phase} provenance mismatch: {path}')
    return marker


def encode_vehicle(normalized):
    values = np.asarray(normalized, dtype=float)
    if (values.ndim != 2 or values.shape[1] != 16 or not np.isfinite(values).all()
            or not np.isin(values, np.arange(5) / 4).all()):
        raise ValueError('vehicle action requires 16 exact categorical observations')
    onehot = np.eye(5)[(values * 4).astype(int)]
    return np.concatenate((onehot.mean(axis=1), onehot[:, 0], onehot[:, -1]), axis=1)


def check_grid(table):
    if (len(table) != 6 or set(zip(table.arm, table.seed)) != set(itertools.product('AB', SEEDS))
            or not table.threshold.eq(.5).all()
            or not np.isfinite(table[['f1', 'pr_auc', 'roc_auc']].to_numpy()).all()):
        raise ValueError('all six fixed-0.5 validation rows required')


def contract():
    base = list(helper.bbox_core.load_joint_contract(helper.DEFINITIONS)['portable'])
    vehicle = [f'vehicle_{stat}__{name}' for stat in ('fraction', 'first', 'last') for name in CATEGORIES]
    if len(base) != 182:
        raise RuntimeError('parent feature contract changed')
    return dict(A=base, B=base + vehicle)


def load_split(split):
    if split == 'test':
        authenticate('validate')
    parent_path = EXP077 / 'metrics' / f'prepared_jaad_{split}.csv'
    parent_marker = parent_auth(parent_path, 'evaluate' if split == 'test' else 'prepare')
    parent = pd.read_csv(parent_path)
    source = EXP089 / 'cache' / ('test_source' if split == 'test' else 'trainval_source')
    original = pd.read_csv(source / 'rows.csv')
    mask = original.split.eq('val' if split == 'validation' else split).to_numpy()
    rows = original.loc[mask].reset_index(drop=True)
    # Preserve the original Exp077 matrix ordering exactly, not an incidental cache ordering.
    aligned = helper.align_rows(parent, rows)
    indices = pd.Series(np.arange(len(rows)), index=rows.sequence_id).loc[parent.sequence_id].to_numpy()
    normalized_all = np.load(source / 'ego_speed.npy', allow_pickle=False)
    if normalized_all.shape != (len(original), 16, 1):
        raise ValueError('vehicle cache shape mismatch')
    normalized = normalized_all[mask][indices, :, 0]
    encoded = encode_vehicle(normalized)
    expected = {'train': (1552, 999), 'validation': (274, 170), 'test': (1501, 720)}[split]
    if (len(parent), int(parent.label.sum())) != expected or set(parent.label.unique()) != {0, 1}:
        raise ValueError('population changed')
    frames = np.array([json.loads(s) for s in aligned.observed_frames_json])
    if (frames.shape != (len(parent), 16) or not (np.diff(frames, axis=1) == 1).all()
            or not np.array_equal(frames[:, 0], parent.start_frame)
            or not np.array_equal(frames[:, -1], parent.end_frame)
            or not parent.tte_frames.between(30, 60).all()
            or not np.array_equal(aligned.tte_frames, parent.tte_frames)
            or not np.array_equal(aligned.event_frame - aligned.end_frame, parent.tte_frames)):
        raise ValueError('frame identity or TTE mismatch')
    split_path = helper.SPLITS / f'{"val" if split == "validation" else split}.txt'
    if not set(parent.video_id).issubset(set(split_path.read_text().split())):
        raise ValueError('unapproved video split')
    source_files = [parent_path, parent_marker, source / 'rows.csv', source / 'ego_speed.npy', split_path]
    codebook = dict(zip(CATEGORIES, range(5)))
    codes_by_video = {}
    for video in sorted(parent.video_id.unique()):
        path = VEHICLE / f'{video}_vehicle.xml'
        entries = ET.parse(path).getroot().findall('./frame')
        mapping = {int(e.get('id')): codebook[e.get('action')] for e in entries}
        if len(mapping) != len(entries):
            raise ValueError('duplicate vehicle frame annotations')
        codes_by_video[video] = mapping
        source_files.append(path)
    actual = np.array([[codes_by_video[v][int(f)] for f in window]
                       for v, window in zip(parent.video_id, frames)])
    np.testing.assert_array_equal(normalized * 4, actual)
    base = parent[contract()['A']].to_numpy(float)
    if base.shape != (len(parent), 182) or not np.isfinite(base).all():
        raise ValueError('parent features incomplete')
    parent['observed_frames_json'] = aligned.observed_frames_json.to_numpy()
    audit = dict(split=split, rows=len(parent), positive_count=int(parent.label.sum()),
                 tracks=int(parent.track_id.nunique()), exact_raw_xml_match=True, no_dropped_rows=True,
                 category_frame_counts={name: int((actual == k).sum()) for k, name in enumerate(CATEGORIES)},
                 source_records=[record(p) for p in source_files])
    return parent, dict(A=base, B=np.column_stack((base, encoded))), audit


def saved_baseline(rows, seed, split):
    filename = 'jaad_native_validation_all_seeds.csv' if split == 'validation' else 'jaad_native_test_all_seeds.csv'
    path = EXP077 / 'predictions' / filename
    parent_auth(path, 'validate' if split == 'validation' else 'evaluate')
    frame = pd.read_csv(path)
    frame = frame.loc[frame.seed == seed]
    if frame.sequence_id.duplicated().any() or set(frame.sequence_id) != set(rows.sequence_id):
        raise ValueError('baseline prediction population mismatch')
    aligned = frame.set_index('sequence_id').loc[rows.sequence_id]
    for name in ('label', 'video_id', 'track_id'):
        np.testing.assert_array_equal(aligned[name], rows[name])
    return aligned.one_stage_probability.to_numpy(dtype=np.float32).astype(float), path


def prepare():
    if (ROOT / 'PREPARE_COMPLETE.json').exists():
        authenticate('prepare')
        return
    train, xt, audit_t = load_split('train')
    val, xv, audit_v = load_split('validation')
    helper.split_disjoint(train, val)
    reuse, source_files = [], []
    for seed in SEEDS:
        path = EXP077 / 'models' / f'jaad_native_one_stage_seed{seed}.joblib'
        marker = parent_auth(path, 'validate')
        model = joblib.load(path)
        expected = helper.bbox_core.build_xgb(seed, train.label.to_numpy(), n_estimators=250)
        actual_params = {k: repr(v) for k, v in model.named_steps['model'].get_params().items()}
        expected_params = {k: repr(v) for k, v in expected.named_steps['model'].get_params().items()}
        if actual_params != expected_params or model.n_features_in_ != 182:
            raise RuntimeError('baseline estimator parity failed; no new fitting authorized yet')
        np.testing.assert_array_equal(model.named_steps['imputer'].statistics_, np.median(xt['A'], axis=0))
        p, probability_file = saved_baseline(val, seed, 'validation')
        np.testing.assert_array_equal(model.predict_proba(xv['A'])[:, 1].astype(float), p)
        reuse.append(dict(seed=seed, model=record(path), validation_predictions=record(probability_file),
                          parameters=actual_params, exact_prediction_parity=True, imputer_parity=True,
                          active_features=182, source_experiment=77))
        source_files += [path, marker, probability_file]
    source_files += [Path(s['path']) for a in (audit_t, audit_v) for s in a['source_records']]
    source_files += [ROOT / 'DESIGN.md', Path(__file__), Path(__file__).with_name('test_exp099.py'),
                     HELPER, helper.CORE_PATH, helper.BBOX_PATH, helper.DEFINITIONS,
                     REPO / 'src/biu_v2_extractor.py', REPO / 'src/features/biu_features.py',
                     EXP077 / 'scripts/run_experiment.py']
    files = []
    for name, value in [('development_audit.json', [audit_t, audit_v]), ('baseline_reuse_audit.json', reuse),
                        ('feature_contract.json', contract()),
                        ('freeze.json', dict(source_records=[record(p) for p in dict.fromkeys(source_files)],
                                             test_accessed=False, seeds=SEEDS, new_fits=3, baseline_reused=3,
                                             versions=dict(python=sys.version, xgboost=helper.xgboost.__version__,
                                                           sklearn=helper.sklearn.__version__, numpy=np.__version__)))]:
        path = ROOT / 'metrics' / name
        write(path, value)
        files.append(path)
    publish('prepare', files + source_files)


def pred_frame(rows, p, arm, seed):
    # Use float64 serialization to preserve original float32 values exactly on reload.
    return helper.predictions(rows, np.asarray(p, dtype=float), arm, seed)


def metric_row(rows, p, arm, seed, scope):
    return dict(scope=scope, arm=arm, seed=seed, feature_count=182 if arm == 'A' else 197,
                **core.compute_metrics(rows.label.to_numpy(), p))


def validate():
    authenticate('prepare')
    if (ROOT / 'VALIDATE_COMPLETE.json').exists():
        authenticate('validate')
        return
    train, xt, _ = load_split('train')
    val, xv, _ = load_split('validation')
    results, files = [], []
    for seed in SEEDS:
        a, _ = saved_baseline(val, seed, 'validation')
        model_path = ROOT / 'models' / f'B_seed{seed}.joblib'
        marker = f'B_seed{seed}'
        if (ROOT / f'{marker.upper()}_COMPLETE.json').exists():
            authenticate(marker)
            model = joblib.load(model_path)
        else:
            model = helper.bbox_core.build_xgb(seed, train.label.to_numpy(), n_estimators=250)
            start = time.perf_counter()
            model.fit(xt['B'], train.label.to_numpy())
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, str(model_path) + '.tmp')
            os.replace(str(model_path) + '.tmp', model_path)
            publish(marker, [model_path])
            print(f'B_seed{seed} fitted in {time.perf_counter() - start:.2f}s', flush=True)
        b = model.predict_proba(xv['B'])[:, 1].astype(float)
        for arm, p in (('A', a), ('B', b)):
            results.append(metric_row(val, p, arm, seed, 'validation'))
            path = ROOT / 'predictions' / f'validation_{arm}_seed{seed}.csv'
            write(path, pred_frame(val, p, arm, seed))
            files.append(path)
        files += [model_path, ROOT / f'{marker.upper()}_COMPLETE.json']
    table = pd.DataFrame(results)
    check_grid(table)
    for name, value in [('validation_per_seed_results.csv', table),
                        ('validation_summary_results.csv', helper.summarize(table))]:
        path = ROOT / 'metrics' / name
        write(path, value)
        files.append(path)
    authenticate('prepare')
    publish('validate', files)


def bootstrap(rows, a, b, replicates=5000):
    y = rows.label.to_numpy()
    codes, groups = pd.factorize(rows.track_id, sort=True)
    rng = np.random.default_rng(42)

    def score(p, weights):
        prediction = p >= .5
        tp = (weights * ((y == 1) & prediction)).sum()
        fp = (weights * ((y == 0) & prediction)).sum()
        fn = (weights * ((y == 1) & ~prediction)).sum()
        return np.array([2 * tp / max(2 * tp + fp + fn, 1),
                         average_precision_score(y, p, sample_weight=weights),
                         roc_auc_score(y, p, sample_weight=weights)])

    delta = score(b, np.ones(len(y))) - score(a, np.ones(len(y)))
    draws = []
    for index in range(replicates):
        weights = np.bincount(rng.integers(len(groups), size=len(groups)), minlength=len(groups))[codes]
        if not all(weights[y == label].sum() > 0 for label in (0, 1)):
            continue
        draws.append(score(b, weights) - score(a, weights))
        if replicates >= 1000 and (index + 1) % 1000 == 0:
            print(f'track bootstrap {index + 1}/{replicates}', flush=True)
    draws = np.array(draws)
    return pd.DataFrame([dict(metric=m, delta=delta[i], ci_low=np.quantile(draws[:, i], .025),
                              ci_high=np.quantile(draws[:, i], .975), replicates=len(draws), bootstrap_seed=42)
                         for i, m in enumerate(('f1', 'pr_auc', 'roc_auc'))])


def evaluate():
    authenticate('prepare')
    authenticate('validate')
    if (ROOT / 'EVALUATE_COMPLETE.json').exists():
        authenticate('evaluate')
        return
    train, _, _ = load_split('train')
    val, _, _ = load_split('validation')
    test, xt, audit = load_split('test')
    helper.split_disjoint(train, val, test)
    results, deltas, transitions, files = [], [], [], []
    probabilities = {'A': [], 'B': []}
    for seed in SEEDS:
        a, source = saved_baseline(test, seed, 'test')
        model = joblib.load(ROOT / 'models' / f'B_seed{seed}.joblib')
        b = model.predict_proba(xt['B'])[:, 1].astype(float)
        scores = {}
        for arm, p in (('A', a), ('B', b)):
            scores[arm] = metric_row(test, p, arm, seed, 'test')
            results.append(scores[arm])
            probabilities[arm].append(p)
            path = ROOT / 'predictions' / f'test_{arm}_seed{seed}.csv'
            write(path, pred_frame(test, p, arm, seed))
            files.append(path)
        files.append(source)
        deltas.append(dict(seed=seed, **{f'delta_{m}': scores['B'][m] - scores['A'][m] for m in METRICS}))
        y, pa, pb = test.label.to_numpy(), a >= .5, b >= .5
        transitions.append(dict(seed=seed,
            FP_corrected=int(((y == 0) & pa & ~pb).sum()), FN_corrected=int(((y == 1) & ~pa & pb).sum()),
            FP_introduced=int(((y == 0) & ~pa & pb).sum()), FN_introduced=int(((y == 1) & pa & ~pb).sum())))
    authenticate('validate')
    table = pd.DataFrame(results)
    check_grid(table)
    intervals = bootstrap(test, np.mean(probabilities['A'], axis=0), np.mean(probabilities['B'], axis=0))
    for name, value in [('test_source_audit.json', audit), ('per_seed_results.csv', table),
                        ('fixed_0p5_results.csv', table), ('summary_results.csv', helper.summarize(table)),
                        ('paired_deltas.csv', pd.DataFrame(deltas)), ('ensemble_track_bootstrap.csv', intervals),
                        ('error_transitions_fixed_0p5.csv', pd.DataFrame(transitions))]:
        path = ROOT / 'metrics' / name
        write(path, value)
        files.append(path)
    publish('evaluate', files + [Path(r['path']) for r in audit['source_records']])


def report():
    authenticate('evaluate')
    table = pd.read_csv(ROOT / 'metrics/summary_results.csv')
    deltas = pd.read_csv(ROOT / 'metrics/paired_deltas.csv')
    lines = ['# Exp099 Final Report', '', 'Clean JAAD: 16 observations, TTE30--60, fixed0.5, seeds42/43/44.',
             'A is the authenticated Exp077 182-D one-stage baseline (three saved models reused).',
             'B adds 15 fixed categorical vehicle-state summaries (three fresh 197-D models).', '',
             '| Arm | F1 mean +/- SD | PR-AUC mean +/- SD | ROC-AUC mean +/- SD |',
             '|---|---:|---:|---:|']
    for row in table.to_dict('records'):
        lines.append('| ' + ' | '.join([row['arm']] + [f'{row[m+"_mean"]:.6f} +/- {row[m+"_std"]:.6f}'
                                                     for m in ('f1', 'pr_auc', 'roc_auc')]) + ' |')
    lines += ['', '## B minus A', '']
    for metric in ('f1', 'pr_auc', 'roc_auc'):
        values = deltas['delta_' + metric]
        lines.append(f'- {metric}: {values.mean():+.6f}; favorable seeds {int((values > 0).sum())}/3.')
    lines += ['', 'All inputs except the categorical vehicle-state block are unchanged. The new',
              'block is exact JAAD vehicle annotation, not measured speed or automatically',
              'inferred ego motion. State occupancies and first/last one-hot codes use only',
              'observation frames. No model or threshold was selected using test results.', '',
              'This is a retrospective matched ablation on previously exposed test data, not',
              'untouched confirmation. Track-bootstrap intervals are descriptive and based on',
              'seed-ensemble probabilities. No subsequent experimental change is authorized.', '']
    path = ROOT / 'FINAL_REPORT.md'
    write(path, '\n'.join(lines))
    publish('report', [path])


def verify():
    for phase in ('prepare', 'validate', 'evaluate', 'report'):
        authenticate(phase)
    checked = []
    for split in ('validation', 'test'):
        table = pd.read_csv(ROOT / 'metrics' / ('validation_per_seed_results.csv' if split == 'validation'
                                              else 'per_seed_results.csv'), float_precision='round_trip')
        check_grid(table)
        identities = None
        for arm, seed in itertools.product('AB', SEEDS):
            path = ROOT / 'predictions' / f'{split}_{arm}_seed{seed}.csv'
            frame = pd.read_csv(path, float_precision='round_trip')
            current = list(zip(frame.sequence_id, frame.label, frame.observed_frames_json))
            if identities is None:
                identities = current
            assert current == identities and len(current) == (274 if split == 'validation' else 1501)
            np.testing.assert_array_equal(frame.prediction, (frame.probability >= .5).astype(int))
            computed = core.compute_metrics(frame.label.to_numpy(), frame.probability.to_numpy())
            saved = table.loc[(table.arm == arm) & (table.seed == seed)].iloc[0]
            for name, value in computed.items():
                np.testing.assert_allclose(value, saved[name], rtol=1e-12, atol=1e-12)
            checked.append(record(path))
    write(ROOT / 'metrics/verification.json', dict(status='passed', checked=checked))
    write(ROOT / 'EXPERIMENT_COMPLETE.json', dict(status='complete', experiment=99, new_fits=3,
          reused_baselines=3, test_rows=1501, classification='retrospective matched ablation',
          report=record(ROOT / 'FINAL_REPORT.md'), verification=record(ROOT / 'metrics/verification.json')))
    print('All 12 prediction files and metrics verified; experiment complete.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['all', 'prepare', 'validate', 'evaluate', 'report', 'verify'])
    args = parser.parse_args()
    logs = ROOT / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / 'commands.log').open('a') as handle:
        handle.write(sys.executable + ' ' + ' '.join(sys.argv) + '\n')
    phases = ['prepare', 'validate', 'evaluate', 'report', 'verify'] if args.phase == 'all' else [args.phase]
    for phase in phases:
        for event in ('started', 'complete'):
            if event == 'complete':
                globals()[phase]()
            message = time.strftime('%Y-%m-%dT%H:%M:%S%z') + f' {phase} {event}'
            print(message, flush=True)
            with (logs / 'progress.log').open('a') as handle:
                handle.write(message + '\n')


if __name__ == '__main__':
    main()
