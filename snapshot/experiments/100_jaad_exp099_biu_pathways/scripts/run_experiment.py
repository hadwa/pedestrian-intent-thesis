#!/usr/bin/env python3
"""Separate historical BIU pathways while retaining Exp099 vehicle context."""

import argparse
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


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
PARENT_SCRIPT = REPO / 'experiments/099_jaad_exp077_vehicle_action/scripts/run_experiment.py'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    out = importlib.util.module_from_spec(spec)
    sys.modules[name] = out
    spec.loader.exec_module(out)
    return out


parent = module('exp100_exp099', PARENT_SCRIPT)
helper, core = parent.helper, parent.core
record, write = parent.record, parent.write
DATA_SCRIPT = helper.BBOX_PATH.with_name('data_core.py')
data = module('exp100_source_data', DATA_SCRIPT)
POSE = REPO / 'data/jaad_keypoints_vitpose_s_extended_short_horizon.jsonl'
SUPPLEMENT = REPO / 'experiments/030_final_confidence_benchmark/jaad30_repair/jaad30_missing_supplemental_keypoints.jsonl'
SPLITS = Path('/home/abdelrahman/Hadwa/BIU-PID-Distillation/data/jaad_splits_event_16_30_60_overlap08_default')
SEEDS = (42, 43, 44)
ARMS = ('v1', 'v2', 'combined')
PAIRS = (('v1', 'v2'), ('v1', 'combined'), ('v2', 'combined'))
STATS = ('early', 'late', 'late_minus_early')


def publish(phase, files):
    write(ROOT / f'{phase.upper()}_COMPLETE.json', [record(p) for p in dict.fromkeys(files)])


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


def families():
    frame = pd.read_csv(helper.DEFINITIONS).set_index('feature_set')
    expected = dict(bbox_only=53, biu_v1_only=72, biu_v2_core=56)
    result = {name: json.loads(frame.loc[name, 'feature_columns_json']) for name in expected}
    if any(len(result[k]) != n or len(set(result[k])) != n for k, n in expected.items()):
        raise ValueError('historical feature family changed')
    return result


def temporal(bases):
    return [f'temporal__{base}_{stat}' for base in bases for stat in STATS]


def contract():
    f = families()
    base = f['bbox_only'] + temporal(('bbox_center_dx_norm', 'bbox_scale_log_delta'))
    base += parent.contract()['B'][182:]
    quality = temporal(('pose_confidence_mean',))
    v2_time = temporal(('lower_body_motion_magnitude_mean', 'lead_ankle_forward_delta',
                        'torso_motion_magnitude_mean', 'torso_center_x_delta'))
    return dict(v1=base + ['v1__' + n for n in f['biu_v1_only']] + quality,
                v2=base + ['v2__' + n for n in f['biu_v2_core']] + quality + v2_time,
                combined=parent.contract()['B'])


def build_arms(parent_frame, recovered_v2):
    if recovered_v2.shape != (len(parent_frame), 56) or not np.isfinite(recovered_v2).all():
        raise ValueError('complete finite 56-D v2 required')
    f, columns = families(), contract()
    combined = parent_frame[columns['combined']].to_numpy(float)
    values = {name: parent_frame[name].to_numpy(float) for name in parent_frame.columns}
    values.update({'v1__' + name: parent_frame[name].to_numpy(float) for name in f['biu_v1_only']})
    values.update({'v2__' + name: recovered_v2[:, i] for i, name in enumerate(f['biu_v2_core'])})
    out = {a: np.column_stack([values[name] for name in columns[a]]) for a in ('v1', 'v2')}
    out['combined'] = combined
    for arm, size in dict(v1=149, v2=145, combined=197).items():
        if out[arm].shape != (len(parent_frame), size) or not np.isfinite(out[arm]).all():
            raise ValueError(f'invalid {arm} matrix')
    return out


def check_grid(table):
    if (len(table) != 9 or set(zip(table.arm, table.seed)) != set(itertools.product(ARMS, SEEDS))
            or not table.threshold.eq(.5).all()
            or not np.isfinite(table[['f1', 'pr_auc', 'roc_auc']].to_numpy()).all()):
        raise ValueError('nine complete fixed-0.5 rows required')


def load_split(split):
    if split == 'test':
        authenticate('validate')
    marker = f'materialize_{split}'
    cache = ROOT / 'cache' / split
    if (ROOT / f'{marker.upper()}_COMPLETE.json').exists():
        authenticate(marker)
        rows = pd.read_csv(cache / 'rows.csv')
        with np.load(cache / 'features.npz', allow_pickle=False) as arrays:
            features = {a: arrays[a] for a in ARMS}
        audit = json.loads((ROOT / 'metrics' / f'{split}_source_audit.json').read_text())
        return rows, features, audit
    parent.authenticate('prepare')
    rows, matrices, parent_audit = parent.load_split(split)
    parent_frame = pd.DataFrame(matrices['B'], columns=parent.contract()['B'])
    source_path = SPLITS / f'{"val" if split == "validation" else split}.csv'
    source = pd.read_csv(source_path)
    aligned = helper.align_rows(rows, source)
    aligned['set_id'] = 'jaad'
    data.validate_window_cohort(aligned, tte_column='tte_frames')
    for original, expected in zip(aligned.observed_frames_json, rows.observed_frames_json):
        if json.loads(original) != json.loads(expected):
            raise ValueError('source observation identities differ')
    old_provenance = parent.EXP077 / 'metrics/source_provenance.json'
    lineage = json.loads(old_provenance.read_text())
    for path, key in ((POSE, 'jaad_keypoints_primary'), (SUPPLEMENT, 'jaad_keypoints_supplement')):
        if record(path)['sha256'] != lineage[key]['sha256']:
            raise RuntimeError('historical pose SHA changed')
    needed = data.needed_keys(aligned)
    poses, pose_audit = data.load_keypoint_index((POSE, SUPPLEMENT), needed)
    boxes = data.load_jaad_bboxes(aligned)
    f = families()
    common_names = sorted(set(f['biu_v2_core']) & set(f['biu_v1_only']))
    unique_names = [n for n in f['biu_v2_core'] if n not in common_names]
    if len(common_names) != 20 or len(unique_names) != 36:
        raise ValueError('historical v1/v2 overlap changed')
    v2_values = np.empty((len(rows), 56))
    max_v1, max_v2, fallback = 0., 0., 0
    for index, row in enumerate(aligned.itertuples(index=False)):
        keys = [('jaad', str(row.video_id), str(row.ped_id), frame) for frame in data.observed_frames(row)]
        xy = np.array([poses[k]['keypoints'] for k in keys], dtype=float)
        confidence = np.array([poses[k]['kp_conf'] for k in keys], dtype=float)
        bbox = np.array([boxes[k] for k in keys], dtype=float)
        v2 = helper.bbox_core.extract_biu_v2(xy, confidence, bbox_sequence=bbox)
        v1 = helper.bbox_core._biu_v1(xy, confidence)
        v2_values[index] = [v2[n] for n in f['biu_v2_core']]
        v1_old = parent_frame.loc[index, f['biu_v1_only']].to_numpy(float)
        v1_new = np.array([v1[n] for n in f['biu_v1_only']])
        v2_old = parent_frame.loc[index, ['v2__' + n for n in unique_names]].to_numpy(float)
        v2_new = np.array([v2[n] for n in unique_names])
        np.testing.assert_allclose(v1_new, v1_old, rtol=1e-9, atol=1e-8)
        np.testing.assert_allclose(v2_new, v2_old, rtol=1e-9, atol=1e-8)
        max_v1 = max(max_v1, float(np.max(np.abs(v1_new - v1_old))))
        max_v2 = max(max_v2, float(np.max(np.abs(v2_new - v2_old))))
        fallback += int(v2['direction_source'] == 'bbox_center_x')
        if (index + 1) % 500 == 0:
            print(f'{split}: v2 reconstruction/parity {index + 1}/{len(rows)}', flush=True)
    arms = build_arms(parent_frame, v2_values)
    np.testing.assert_array_equal(arms['combined'], matrices['B'])
    sources = [Path(s['path']) for s in parent_audit['source_records']]
    sources += [source_path, POSE, SUPPLEMENT, old_provenance]
    audit = dict(split=split, rows=len(rows), source_records=[record(p) for p in dict.fromkeys(sources)],
                 pose=pose_audit, inherited_vehicle_audit=parent_audit,
                 recovered_v2_features=56, overlapping_v2_values_recovered=common_names,
                 max_abs_v1_parity_difference=max_v1, max_abs_unique_v2_parity_difference=max_v2,
                 exact_combined_input_reuse=True, bbox_direction_fallback_count=fallback,
                 bbox_direction_fallback_fraction=fallback / len(rows),
                 no_pose_estimator_rerun=True, normalization_image_size=[1920, 1080])
    cache.mkdir(parents=True, exist_ok=True)
    temporary = cache / 'features.tmp'
    with temporary.open('wb') as file:
        np.savez(file, **arms)
    os.replace(temporary, cache / 'features.npz')
    write(cache / 'rows.csv', rows)
    audit_path = ROOT / 'metrics' / f'{split}_source_audit.json'
    write(audit_path, audit)
    publish(marker, [cache / 'features.npz', cache / 'rows.csv', audit_path, *sources])
    return rows, arms, audit


def combined_probability(rows, seed, split):
    parent.authenticate('validate' if split == 'validation' else 'evaluate')
    path = parent.ROOT / 'predictions' / f'{split}_B_seed{seed}.csv'
    frame = pd.read_csv(path, float_precision='round_trip')
    aligned = helper.align_rows(rows, frame)
    if not np.array_equal(aligned.observed_frames_json, rows.observed_frames_json):
        raise ValueError('combined prediction frames differ')
    return aligned.probability.to_numpy(float), path


def prepare():
    if (ROOT / 'PREPARE_COMPLETE.json').exists():
        authenticate('prepare')
        return
    train, xt, a = load_split('train')
    val, xv, b = load_split('validation')
    helper.split_disjoint(train, val)
    reuse, files = [], []
    for seed in SEEDS:
        path = parent.ROOT / 'models' / f'B_seed{seed}.joblib'
        parent.authenticate(f'B_seed{seed}')
        model = joblib.load(path)
        expected = helper.bbox_core.build_xgb(seed, train.label.to_numpy(), n_estimators=250)
        assert {k: repr(v) for k, v in model.named_steps['model'].get_params().items()} == {
            k: repr(v) for k, v in expected.named_steps['model'].get_params().items()}
        np.testing.assert_array_equal(model.named_steps['imputer'].statistics_, np.median(xt['combined'], axis=0))
        p, source = combined_probability(val, seed, 'validation')
        np.testing.assert_array_equal(model.predict_proba(xv['combined'])[:, 1].astype(float), p)
        reuse.append(dict(seed=seed, source_model=record(path), prediction_source=record(source),
                          identical_parameters=True, identical_imputer=True, identical_predictions=True))
        files += [path, source, parent.ROOT / f'B_SEED{seed}_COMPLETE.json']
    for name, value in [('feature_contract.json', contract()), ('baseline_reuse_audit.json', reuse)]:
        path = ROOT / 'metrics' / name
        write(path, value)
        files.append(path)
    for split, audit in (('train', a), ('validation', b)):
        files += [ROOT / f'MATERIALIZE_{split.upper()}_COMPLETE.json', ROOT / 'cache' / split / 'features.npz',
                  ROOT / 'cache' / split / 'rows.csv', ROOT / 'metrics' / f'{split}_source_audit.json']
        files += [Path(r['path']) for r in audit['source_records']]
    files += [ROOT / 'DESIGN.md', Path(__file__), Path(__file__).with_name('test_exp100.py'),
              PARENT_SCRIPT, parent.HELPER, helper.CORE_PATH, helper.BBOX_PATH, DATA_SCRIPT,
              helper.DEFINITIONS, REPO / 'src/biu_v2_extractor.py', REPO / 'src/features/biu_features.py',
              parent.ROOT / 'DESIGN.md', parent.ROOT / 'metrics/freeze.json']
    freeze = ROOT / 'metrics/freeze.json'
    write(freeze, dict(source_records=[record(p) for p in dict.fromkeys(files)], test_accessed=False,
                       new_fits=6, reused_models=3, seeds=SEEDS, dimensions=dict(v1=149,v2=145,combined=197),
                       versions=dict(python=sys.version, numpy=np.__version__, sklearn=helper.sklearn.__version__,
                                     xgboost=helper.xgboost.__version__)))
    publish('prepare', files + [freeze])


def metric_row(rows, p, arm, seed, scope):
    return dict(scope=scope, arm=arm, seed=seed, feature_count=len(contract()[arm]),
                **core.compute_metrics(rows.label.to_numpy(), p))


def validate():
    authenticate('prepare')
    if (ROOT / 'VALIDATE_COMPLETE.json').exists():
        authenticate('validate')
        return
    train, xt, _ = load_split('train')
    val, xv, _ = load_split('validation')
    results, files = [], []
    for arm, seed in itertools.product(ARMS, SEEDS):
        if arm == 'combined':
            p, source = combined_probability(val, seed, 'validation')
            files.append(source)
        else:
            path = ROOT / 'models' / f'{arm}_seed{seed}.joblib'
            phase = f'{arm}_seed{seed}'
            if (ROOT / f'{phase.upper()}_COMPLETE.json').exists():
                authenticate(phase)
                model = joblib.load(path)
            else:
                model = helper.bbox_core.build_xgb(seed, train.label.to_numpy(), n_estimators=250)
                start = time.perf_counter()
                model.fit(xt[arm], train.label.to_numpy())
                path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, str(path) + '.tmp')
                os.replace(str(path) + '.tmp', path)
                publish(phase, [path])
                print(f'{arm}_seed{seed} fitted in {time.perf_counter() - start:.2f}s', flush=True)
            p = model.predict_proba(xv[arm])[:, 1].astype(float)
            files += [path, ROOT / f'{phase.upper()}_COMPLETE.json']
        results.append(metric_row(val, p, arm, seed, 'validation'))
        out = ROOT / 'predictions' / f'validation_{arm}_seed{seed}.csv'
        write(out, helper.predictions(val, p, arm, seed))
        files.append(out)
    table = pd.DataFrame(results)
    check_grid(table)
    for name, value in [('validation_per_seed_results.csv', table),
                        ('validation_summary_results.csv', helper.summarize(table))]:
        path = ROOT / 'metrics' / name
        write(path, value)
        files.append(path)
    authenticate('prepare')
    publish('validate', files)


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
    results, files, probabilities = [], [], {}
    for arm, seed in itertools.product(ARMS, SEEDS):
        if arm == 'combined':
            p, source = combined_probability(test, seed, 'test')
            files.append(source)
        else:
            model = joblib.load(ROOT / 'models' / f'{arm}_seed{seed}.joblib')
            p = model.predict_proba(xt[arm])[:, 1].astype(float)
        probabilities[(arm, seed)] = p
        results.append(metric_row(test, p, arm, seed, 'test'))
        path = ROOT / 'predictions' / f'test_{arm}_seed{seed}.csv'
        write(path, helper.predictions(test, p, arm, seed))
        files.append(path)
    table = pd.DataFrame(results)
    check_grid(table)
    indexed = table.set_index(['arm', 'seed'])
    deltas, transitions, intervals = [], [], []
    for left, right in PAIRS:
        comparison = f'{right}_minus_{left}'
        for seed in SEEDS:
            deltas.append(dict(comparison=comparison, seed=seed,
                               **{'delta_' + m: indexed.loc[(right, seed), m] - indexed.loc[(left, seed), m]
                                  for m in parent.METRICS}))
            y, a, b = test.label.to_numpy(), probabilities[(left, seed)] >= .5, probabilities[(right, seed)] >= .5
            transitions.append(dict(comparison=comparison, seed=seed,
                FP_corrected=int(((y == 0) & a & ~b).sum()), FN_corrected=int(((y == 1) & ~a & b).sum()),
                FP_introduced=int(((y == 0) & ~a & b).sum()), FN_introduced=int(((y == 1) & a & ~b).sum())))
        a = np.mean([probabilities[(left, s)] for s in SEEDS], axis=0)
        b = np.mean([probabilities[(right, s)] for s in SEEDS], axis=0)
        print(f'Uncertainty: {comparison}', flush=True)
        result = parent.bootstrap(test, a, b, replicates=5000)
        result.insert(0, 'comparison', comparison)
        intervals.append(result)
    authenticate('validate')
    for name, value in [('per_seed_results.csv', table), ('fixed_0p5_results.csv', table),
                        ('summary_results.csv', helper.summarize(table)), ('paired_deltas.csv', pd.DataFrame(deltas)),
                        ('error_transitions_fixed_0p5.csv', pd.DataFrame(transitions)),
                        ('ensemble_track_bootstrap.csv', pd.concat(intervals, ignore_index=True))]:
        path = ROOT / 'metrics' / name
        write(path, value)
        files.append(path)
    files += [ROOT / 'MATERIALIZE_TEST_COMPLETE.json', ROOT / 'cache/test/features.npz',
              ROOT / 'cache/test/rows.csv', ROOT / 'metrics/test_source_audit.json']
    files += [Path(r['path']) for r in audit['source_records']]
    publish('evaluate', files)


def report():
    authenticate('evaluate')
    table = pd.read_csv(ROOT / 'metrics/summary_results.csv')
    deltas = pd.read_csv(ROOT / 'metrics/paired_deltas.csv')
    lines = ['# Exp100 Final Report', '', 'Clean JAAD; 16 observations, TTE30--60, fixed0.5, seeds42/43/44.',
             'All arms retain the same BBox and categorical vehicle-action inputs.',
             'v1/v2 use six new fits; combined reuses Exp099 models and probabilities.', '',
             '| Pathway | Features | F1 mean +/- SD | PR-AUC mean +/- SD | ROC-AUC mean +/- SD |',
             '|---|---:|---:|---:|---:|']
    for row in table.to_dict('records'):
        lines.append('| ' + ' | '.join([row['arm'], str(row['feature_count'])] +
                     [f'{row[m+"_mean"]:.6f} +/- {row[m+"_std"]:.6f}' for m in ('f1', 'pr_auc', 'roc_auc')]) + ' |')
    for comparison in deltas.comparison.unique():
        lines += ['', f'## {comparison}', '']
        for metric in ('f1', 'pr_auc', 'roc_auc'):
            values = deltas.loc[deltas.comparison == comparison, 'delta_' + metric]
            lines.append(f'- {metric}: {values.mean():+.6f}; favorable seeds {int((values > 0).sum())}/3.')
    lines += ['', '## Interpretation limits', '',
              'V2 is the complete historical core-56, not full-63. Its 20 overlapping',
              'feature names were reconstructed from the original cached pose measurements,',
              'not filled with v1 values from the deduplicated parent. All 36 unique v2',
              'and 72 v1 cached columns passed numeric parity checks. No pose estimator ran.', '',
              'This is an Exp096-style pathway comparison: v2 carries twelve additional',
              'historical temporal features; both arms retain three existing pose-quality',
              'summaries. Vehicle input is annotation-derived categorical state, not physical',
              'speed. The existing normalization and direction fallback were unchanged.', '',
              'Previously exposed test cohort: retrospective matched ablation, not pristine',
              'confirmation. Track-bootstrap intervals are descriptive, based on seed-ensemble',
              'probabilities and not multiplicity adjusted. Do not claim universal version',
              'dominance or silently equate this pathway comparison with Exp098.', '']
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
        for arm, seed in itertools.product(ARMS, SEEDS):
            path = ROOT / 'predictions' / f'{split}_{arm}_seed{seed}.csv'
            frame = pd.read_csv(path, float_precision='round_trip')
            current = list(zip(frame.sequence_id, frame.label, frame.observed_frames_json))
            if identities is None:
                identities = current
            assert current == identities and len(current) == (274 if split == 'validation' else 1501)
            np.testing.assert_array_equal(frame.prediction, (frame.probability >= .5).astype(int))
            values = core.compute_metrics(frame.label.to_numpy(), frame.probability.to_numpy())
            saved = table.loc[(table.arm == arm) & (table.seed == seed)].iloc[0]
            for key, value in values.items():
                np.testing.assert_allclose(value, saved[key], rtol=1e-12, atol=1e-12)
            if arm == 'combined':
                old = pd.read_csv(parent.ROOT / 'predictions' / f'{split}_B_seed{seed}.csv', float_precision='round_trip')
                np.testing.assert_array_equal(frame.probability, old.probability)
            checked.append(record(path))
    write(ROOT / 'metrics/verification.json', dict(status='passed', prediction_files=checked))
    write(ROOT / 'EXPERIMENT_COMPLETE.json', dict(status='complete', experiment=100, new_fits=6,
          reused_models=3, test_rows=1501, classification='retrospective matched pathway ablation',
          report=record(ROOT / 'FINAL_REPORT.md'), verification=record(ROOT / 'metrics/verification.json')))
    print('All 18 prediction files and metrics verified; Exp100 complete.', flush=True)


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
