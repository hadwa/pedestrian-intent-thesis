#!/usr/bin/env python3
"""Matched seven-value augmentation of the authenticated Exp100 v2 pathway."""

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
PARENT_SCRIPT = REPO / 'experiments/100_jaad_exp099_biu_pathways/scripts/run_experiment.py'
spec = importlib.util.spec_from_file_location('exp101_parent', PARENT_SCRIPT)
parent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parent
spec.loader.exec_module(parent)
helper, core, data = parent.helper, parent.core, parent.data
write, record = parent.write, parent.record
SEEDS = (42, 43, 44)
ARMS = ('core56', 'full63')
EXTRA = ('direction_sign', 'was_flipped', 'direction_confidence',
         'shoulder_width_mean', 'shoulder_width_delta',
         'torso_length_mean', 'torso_length_delta')


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


def contract():
    base = parent.contract()['v2']
    assert len(base) == 145 and not set(EXTRA) & set(parent.families()['biu_v2_core'])
    return dict(core56=base, full63=base + ['v2__' + n for n in EXTRA])


def append_metadata(base, extra):
    base, extra = np.asarray(base), np.asarray(extra)
    if (base.ndim != 2 or base.shape[1] != 145 or extra.shape != (len(base), 7)
            or not np.isfinite(base).all() or not np.isfinite(extra).all()):
        raise ValueError('finite matched 145-D base and exactly seven additional values required')
    return np.column_stack((base, extra))


def check_grid(table):
    if (len(table) != 6 or set(zip(table.arm, table.seed)) != set(itertools.product(ARMS, SEEDS))
            or not table.threshold.eq(.5).all()
            or not np.isfinite(table[['f1', 'pr_auc', 'roc_auc']].to_numpy()).all()):
        raise ValueError('six complete fixed-0.5 rows required')


def load_split(split):
    if split == 'test':
        authenticate('validate')
    if split not in ('train', 'validation', 'test'):
        raise ValueError('unknown split')
    phase, cache = f'materialize_{split}', ROOT / 'cache' / split
    audit_path = ROOT / 'metrics' / f'{split}_source_audit.json'
    if (ROOT / f'{phase.upper()}_COMPLETE.json').exists():
        authenticate(phase)
        with np.load(cache / 'features.npz', allow_pickle=False) as values:
            arrays = {arm: values[arm] for arm in ARMS}
        return pd.read_csv(cache / 'rows.csv'), arrays, json.loads(audit_path.read_text())
    rows, matrices, old_audit = parent.load_split(split)
    base = matrices['v2']
    if len(rows) != dict(train=1552, validation=274, test=1501)[split]:
        raise ValueError('parent population changed')
    source_path = parent.SPLITS / f'{"val" if split == "validation" else split}.csv'
    aligned = helper.align_rows(rows, pd.read_csv(source_path))
    aligned['set_id'] = 'jaad'
    data.validate_window_cohort(aligned, tte_column='tte_frames')
    for a, b in zip(aligned.observed_frames_json, rows.observed_frames_json):
        if json.loads(a) != json.loads(b):
            raise ValueError('observation frame identity mismatch')
    needed = data.needed_keys(aligned)
    poses, pose_audit = data.load_keypoint_index((parent.POSE, parent.SUPPLEMENT), needed)
    boxes = data.load_jaad_bboxes(aligned)
    names = parent.families()['biu_v2_core']
    indices = [contract()['core56'].index('v2__' + n) for n in names]
    extras, max_diff, fallback = np.empty((len(rows), 7)), 0., 0
    for i, row in enumerate(aligned.itertuples(index=False)):
        keys = [('jaad', str(row.video_id), str(row.ped_id), f) for f in data.observed_frames(row)]
        xy = np.array([poses[k]['keypoints'] for k in keys], dtype=float)
        confidence = np.array([poses[k]['kp_conf'] for k in keys], dtype=float)
        bbox = np.array([boxes[k] for k in keys], dtype=float)
        features = helper.bbox_core.extract_biu_v2(xy, confidence, bbox_sequence=bbox)
        recovered = np.array([features[n] for n in names])
        np.testing.assert_allclose(recovered, base[i, indices], rtol=1e-9, atol=1e-8)
        max_diff = max(max_diff, float(np.max(np.abs(recovered - base[i, indices]))))
        extras[i] = [features[n] for n in EXTRA]
        fallback += int(features['direction_source'] == 'bbox_center_x')
        if (i + 1) % 500 == 0:
            print(f'{split}: feature reconstruction/parity {i + 1}/{len(rows)}', flush=True)
    arrays = dict(core56=base, full63=append_metadata(base, extras))
    np.testing.assert_array_equal(arrays['full63'][:, :145], base)
    sources = [Path(s['path']) for s in old_audit['source_records']]
    sources += [parent.ROOT / 'cache' / split / 'features.npz',
                parent.ROOT / 'cache' / split / 'rows.csv',
                parent.ROOT / f'MATERIALIZE_{split.upper()}_COMPLETE.json',
                source_path, parent.POSE, parent.SUPPLEMENT]
    audit = dict(split=split, rows=len(rows), pose=pose_audit,
                 source_records=[record(p) for p in dict.fromkeys(sources)],
                 exact_parent_145_columns=True, core56_max_abs_difference=max_diff,
                 additional_features=list(EXTRA), bbox_direction_fallback_count=fallback,
                 bbox_direction_fallback_fraction=fallback / len(rows),
                 normalization_image_size=[1920, 1080], no_pose_estimator_rerun=True)
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / 'features.tmp').open('wb') as handle:
        np.savez(handle, **arrays)
    os.replace(cache / 'features.tmp', cache / 'features.npz')
    write(cache / 'rows.csv', rows)
    write(audit_path, audit)
    publish(phase, [cache / 'features.npz', cache / 'rows.csv', audit_path, *sources])
    return rows, arrays, audit


def baseline_probability(rows, seed, split):
    if split == 'test':
        authenticate('validate')
    parent.authenticate('validate' if split == 'validation' else 'evaluate')
    path = parent.ROOT / 'predictions' / f'{split}_v2_seed{seed}.csv'
    frame = helper.align_rows(rows, pd.read_csv(path, float_precision='round_trip'))
    np.testing.assert_array_equal(frame.observed_frames_json, rows.observed_frames_json)
    return frame.probability.to_numpy(float), path


def prepare():
    if (ROOT / 'PREPARE_COMPLETE.json').exists():
        authenticate('prepare')
        return
    parent.authenticate('prepare')
    parent.authenticate('validate')
    train, xt, a = load_split('train')
    val, xv, b = load_split('validation')
    helper.split_disjoint(train, val)
    files, reuse = [], []
    for seed in SEEDS:
        path = parent.ROOT / 'models' / f'v2_seed{seed}.joblib'
        parent.authenticate(f'v2_seed{seed}')
        model = joblib.load(path)
        expected = helper.bbox_core.build_xgb(seed, train.label.to_numpy(), n_estimators=250)
        assert {k: repr(v) for k, v in model.named_steps['model'].get_params().items()} == {
            k: repr(v) for k, v in expected.named_steps['model'].get_params().items()}
        np.testing.assert_array_equal(model.named_steps['imputer'].statistics_, np.median(xt['core56'], axis=0))
        p, source = baseline_probability(val, seed, 'validation')
        np.testing.assert_array_equal(model.predict_proba(xv['core56'])[:, 1].astype(float), p)
        reuse.append(dict(seed=seed, source_model=record(path), prediction_source=record(source),
                          identical_parameters=True, identical_imputer=True, identical_predictions=True))
        files += [path, source, parent.ROOT / f'V2_SEED{seed}_COMPLETE.json']
    for name, value in [('feature_contract.json', contract()), ('baseline_reuse_audit.json', reuse)]:
        path = ROOT / 'metrics' / name
        write(path, value)
        files.append(path)
    for split, audit in (('train', a), ('validation', b)):
        files += [ROOT / f'MATERIALIZE_{split.upper()}_COMPLETE.json', ROOT / 'cache' / split / 'features.npz',
                  ROOT / 'cache' / split / 'rows.csv', ROOT / 'metrics' / f'{split}_source_audit.json']
        files += [Path(r['path']) for r in audit['source_records']]
    files += [ROOT / 'DESIGN.md', Path(__file__), Path(__file__).with_name('test_exp101.py'), PARENT_SCRIPT,
              parent.PARENT_SCRIPT, parent.parent.HELPER, helper.CORE_PATH, helper.BBOX_PATH,
              parent.DATA_SCRIPT, helper.DEFINITIONS, REPO / 'src/biu_v2_extractor.py',
              REPO / 'src/features/biu_features.py', parent.ROOT / 'metrics/freeze.json',
              parent.ROOT / 'PREPARE_COMPLETE.json', parent.ROOT / 'VALIDATE_COMPLETE.json']
    freeze = ROOT / 'metrics/freeze.json'
    write(freeze, dict(source_records=[record(p) for p in dict.fromkeys(files)], test_accessed=False,
                       new_fits=3, reused_models=3, seeds=SEEDS, dimensions=dict(core56=145, full63=152),
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
        if arm == 'core56':
            p, source = baseline_probability(val, seed, 'validation')
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
        if arm == 'core56':
            p, source = baseline_probability(test, seed, 'test')
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
    deltas, transitions = [], []
    for seed in SEEDS:
        deltas.append(dict(comparison='full63_minus_core56', seed=seed,
                           **{'delta_' + m: indexed.loc[('full63', seed), m] - indexed.loc[('core56', seed), m]
                              for m in helper.METRICS}))
        y = test.label.to_numpy()
        a, b = probabilities[('core56', seed)] >= .5, probabilities[('full63', seed)] >= .5
        transitions.append(dict(seed=seed,
            FP_corrected=int(((y == 0) & a & ~b).sum()), FN_corrected=int(((y == 1) & ~a & b).sum()),
            FP_introduced=int(((y == 0) & ~a & b).sum()), FN_introduced=int(((y == 1) & a & ~b).sum())))
    a = np.mean([probabilities[('core56', s)] for s in SEEDS], axis=0)
    b = np.mean([probabilities[('full63', s)] for s in SEEDS], axis=0)
    intervals = parent.parent.bootstrap(test, a, b, replicates=5000)
    authenticate('validate')
    for name, value in [('per_seed_results.csv', table), ('fixed_0p5_results.csv', table),
                        ('summary_results.csv', helper.summarize(table)), ('paired_deltas.csv', pd.DataFrame(deltas)),
                        ('error_transitions_fixed_0p5.csv', pd.DataFrame(transitions)),
                        ('ensemble_track_bootstrap.csv', intervals)]:
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
    intervals = pd.read_csv(ROOT / 'metrics/ensemble_track_bootstrap.csv')
    lines = ['# Exp101 Final Report', '', 'Clean JAAD, 16 observations, TTE30--60, fixed0.5, seeds42/43/44.',
             'Three core56 baselines reused from Exp100; three new full63 fits.', '',
             '| Pathway | Total features | F1 mean +/- SD | PR-AUC mean +/- SD | ROC-AUC mean +/- SD |',
             '|---|---:|---:|---:|---:|']
    for row in table.to_dict('records'):
        lines.append('| ' + ' | '.join([row['arm'], str(row['feature_count'])] +
                     [f'{row[m+"_mean"]:.6f} +/- {row[m+"_std"]:.6f}' for m in ('f1', 'pr_auc', 'roc_auc')]) + ' |')
    lines += ['', '## Paired full63 minus core56', '']
    for metric in ('f1', 'pr_auc', 'roc_auc'):
        values = deltas['delta_' + metric]
        lines.append(f'- {metric}: {values.mean():+.6f}; favorable seeds {int((values > 0).sum())}/3.')
    lines += ['', '## Seed-ensemble paired track bootstrap', '',
              '| Metric | Delta | 95% lower | 95% upper |', '|---|---:|---:|---:|']
    for row in intervals.itertuples(index=False):
        lines.append(f'| {row.metric} | {row.delta:+.6f} | {row.ci_low:+.6f} | {row.ci_high:+.6f} |')
    lines += ['', '## Interpretation and provenance', '',
              'The comparison adds only the seven existing direction/geometry outputs listed in DESIGN.md.',
              'All first145 columns are exactly unchanged, including BBox, temporal summaries, existing',
              'pose-quality and annotated categorical vehicle action (not measured physical ego speed).',
              'All56 core features were rederived from the same observation-only cached pose/BBoxes',
              'and checked for parity; no pose estimator or baseline was retrained.', '',
              'The historical default1920x1080 normalization and direction fallback remain unchanged.',
              'This isolates seven values within this pathway, not universal BIU-version superiority.',
              'The test cohort was previously exposed: retrospective matched ablation, not untouched',
              'confirmation. Bootstrap intervals use seed-ensemble probabilities, not mean-per-seed',
              'metrics, and are descriptive without multiplicity adjustment.', '',
              'Detailed source hashes and parity: metrics/*_source_audit.json, baseline_reuse_audit.json,',
              'feature_contract.json and freeze.json. Exact commands: logs/commands.log.',
              'Per-seed scores, error transitions and high-confidence mistakes are in metrics/.', '']
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
        rows, arrays, _ = load_split(split)
        np.testing.assert_array_equal(arrays['full63'][:, :145], arrays['core56'])
        identities = list(zip(rows.sequence_id, rows.label, rows.observed_frames_json))
        for arm, seed in itertools.product(ARMS, SEEDS):
            path = ROOT / 'predictions' / f'{split}_{arm}_seed{seed}.csv'
            frame = pd.read_csv(path, float_precision='round_trip')
            assert list(zip(frame.sequence_id, frame.label, frame.observed_frames_json)) == identities
            np.testing.assert_array_equal(frame.prediction, (frame.probability >= .5).astype(int))
            values = core.compute_metrics(frame.label.to_numpy(), frame.probability.to_numpy())
            saved = table.loc[(table.arm == arm) & (table.seed == seed)].iloc[0]
            for key, value in values.items():
                np.testing.assert_allclose(value, saved[key], rtol=1e-12, atol=1e-12)
            if arm == 'core56':
                expected, _ = baseline_probability(rows, seed, split)
            else:
                model = joblib.load(ROOT / 'models' / f'{arm}_seed{seed}.joblib')
                expected = model.predict_proba(arrays[arm])[:, 1].astype(float)
            np.testing.assert_array_equal(frame.probability, expected)
            checked.append(record(path))
    write(ROOT / 'metrics/verification.json', dict(status='passed', prediction_files=checked))
    write(ROOT / 'EXPERIMENT_COMPLETE.json', dict(status='complete', experiment=101, new_fits=3,
          reused_models=3, test_rows=1501, classification='retrospective matched seven-value ablation',
          report=record(ROOT / 'FINAL_REPORT.md'), verification=record(ROOT / 'metrics/verification.json')))
    print('All12 prediction files and metrics verified; Exp101 complete.', flush=True)


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
