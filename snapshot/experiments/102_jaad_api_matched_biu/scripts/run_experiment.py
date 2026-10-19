#!/usr/bin/env python3
"""API-style JAAD BBox/BIU ablation, with audited historical prediction reuse."""

import argparse
import ast
import csv
import itertools
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET
import importlib.util

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    out = importlib.util.module_from_spec(spec)
    sys.modules[name] = out
    spec.loader.exec_module(out)
    return out


P100 = REPO / 'experiments/100_jaad_exp099_biu_pathways/scripts/run_experiment.py'
support = module('exp102_support', P100)
helper, core = support.helper, support.core
write, record = support.write, support.record
P089 = REPO / 'experiments/089_pedcmt_standardized_jaad_biu_transfer/scripts/exp089_data.py'
extractor = module('exp102_frozen_biu', P089)
EXP083 = REPO / 'experiments/083_jaad_pcpa_aligned_exp082'
coverage = module('exp102_pose_sources', EXP083 / 'scripts/coverage.py')
EXP031 = REPO / 'experiments/031_jaad_api_label_protocol_sensitivity'
OLD_PREDICTIONS = EXP031 / 'metrics/selected_model_predictions.csv'
OLD_WINDOWS = EXP031 / 'metrics/protocol_a_window_audit.csv'
OLD_BUILDER = REPO / 'scripts/experiments/jaad_biu_v2_benchmark_comparison.py'
OLD_RUNNER = REPO / 'scripts/experiments/jaad_api_label_protocol_sensitivity.py'
OLD_BBOX = Path('/home/abdelrahman/Hadwa/BIU-PID-Distillation/src/features/bbox_features.py')
old_bbox = module('exp102_old_bbox', OLD_BBOX)
COHORTS = ('jaad_beh', 'jaad_all')
ARMS = ('A', 'B', 'C', 'D', 'E')
SEEDS = (42, 43, 44)
COUNTS = dict(jaad_beh=dict(train=2134, val=242, test=1881),
              jaad_all=dict(train=8613, val=1265, test=6732))
BOOTSTRAP_PAIRS = (('A', 'B'), ('B', 'C'), ('C', 'D'), ('B', 'E'))
V1_NAMES = tuple(f'{f}_{a}' for f in extractor.BIU_BASE_FEATURES for a in extractor.AGGREGATIONS) + (
    'biu_valid_frame_count', 'biu_low_confidence_frame_count')
V2_METADATA = REPO / 'experiments/089_pedcmt_standardized_jaad_biu_transfer/cache/trainval_source/metadata.json'
V2_NAMES = tuple(json.loads(V2_METADATA.read_text())['biu_v2_feature_names'])


def publish(phase, files):
    write(ROOT / f'{phase.upper()}_COMPLETE.json', [record(p) for p in dict.fromkeys(files)])


def authenticate(phase):
    path = ROOT / f'{phase.upper()}_COMPLETE.json'
    if not path.exists():
        raise RuntimeError(f'missing {phase} completion')
    records = json.loads(path.read_text())
    if not records or any(record(r['path']) != r for r in records):
        raise RuntimeError(f'{phase} source/output changed')


def contract():
    f = support.families()
    assert len(V1_NAMES) == 72 and len(V2_NAMES) == 63
    bb = f['bbox_only']
    v1, v2 = ['v1__' + n for n in V1_NAMES], ['v2__' + n for n in V2_NAMES]
    return dict(A=bb, B=bb + v1, C=bb + ['v2__' + n for n in f['biu_v2_core']],
                D=bb + v2, E=bb + v1 + v2)


def assemble(bb, v1, v2):
    n = len(bb)
    if any(x.shape != (n, d) or not np.isfinite(x).all() for x, d in ((bb, 53), (v1, 72), (v2, 63))):
        raise ValueError('finite aligned53/72/63 arrays required')
    indices = [V2_NAMES.index(n) for n in support.families()['biu_v2_core']]
    return dict(A=bb, B=np.column_stack((bb, v1)), C=np.column_stack((bb, v2[:, indices])),
                D=np.column_stack((bb, v2)), E=np.column_stack((bb, v1, v2)))


def identities(rows):
    return [(str(r.video_id), str(r.ped_id), tuple(json.loads(r.observed_frames_json)), int(r.label))
            for r in rows.itertuples(index=False)]


def check_grid(table):
    if (len(table) != 30 or set(zip(table.cohort, table.arm, table.seed)) != set(itertools.product(COHORTS, ARMS, SEEDS))
            or not table.threshold.eq(.5).all()
            or not np.isfinite(table[['f1', 'pr_auc', 'roc_auc']].to_numpy()).all()):
        raise ValueError('complete30-row fixed0.5 grid required')


def manifest(cohort, split):
    if split == 'test':
        authenticate('validate')
    path = EXP083 / 'manifests' / f'{cohort}_{split}.csv'
    rows = pd.read_csv(path)
    frames = np.array([json.loads(x) for x in rows.observed_frames_json])
    gaps = (np.diff(frames, axis=1) != 1).any(axis=1)
    if (len(rows) != COUNTS[cohort][split] or frames.shape != (len(rows), 16)
            or rows.sequence_id.duplicated().any() or len(set(identities(rows))) != len(rows)
            or not rows.cohort.eq(cohort).all() or not rows.split.eq(split).all()
            or not rows.tte_frames.between(30, 60).all()
            or not np.array_equal(frames[:, 0], rows.start_frame)
            or not np.array_equal(frames[:, -1], rows.end_frame)
            or not np.array_equal(rows.api_track_length - 1 - rows.api_window_end_index, rows.tte_frames)
            or not (np.diff(frames, axis=1) > 0).all()
            or int(gaps.sum()) != (2 if (cohort, split) == ('jaad_all', 'train') else 0)):
        raise ValueError('frozen Exp083 population/frame protocol changed')
    return rows


def read_pose(needed):
    result, counts = {}, {}
    for source in coverage.POSE_SOURCES:
        path = Path(source['path'])
        if record(path)['sha256'] != source['sha256']:
            raise ValueError('Exp083 pose source changed')
        added = 0
        with path.open() as handle:
            for line in handle:
                raw = json.loads(line)
                key = (str(raw['video']), str(raw['pid']), int(raw['frame_idx']))
                if key not in needed:
                    continue
                xy = np.asarray(raw['keypoints'], dtype=np.float32)
                conf = np.asarray(raw['kp_conf'], dtype=np.float32)
                if xy.shape != (17, 2) or conf.shape != (17,):
                    raise ValueError('invalid cached pose shape')
                if key in result:
                    np.testing.assert_array_equal(xy, result[key][0])
                    np.testing.assert_array_equal(conf, result[key][1])
                else:
                    result[key] = (xy, conf)
                    added += 1
        counts[str(path)] = added
    if set(result) != needed:
        raise ValueError('missing exact pose measurements; no re-extraction authorized')
    return result, counts


def load_split(split):
    if split == 'test':
        authenticate('validate')
    phase, cache = f'materialize_{split}', ROOT / 'cache' / split
    audit_path = ROOT / 'metrics' / f'{split}_source_audit.json'
    if (ROOT / f'{phase.upper()}_COMPLETE.json').exists():
        authenticate(phase)
        result = {}
        for cohort in COHORTS:
            rows = pd.read_csv(cache / f'{cohort}_rows.csv')
            with np.load(cache / f'{cohort}.npz', allow_pickle=False) as values:
                arrays = assemble(values['bbox'], values['v1'], values['v2'])
            result[cohort] = (rows, arrays)
        return result, json.loads(audit_path.read_text())
    coverage.contract.authenticate_phase(EXP083, 'protocol-audit')
    populations = {c: manifest(c, split) for c in COHORTS}
    rows = populations['jaad_all']
    needed = {(video, ped, f) for video, ped, frames, label in identities(rows) for f in frames}
    poses, contributions = read_pose(needed)
    bb, v1, v2 = np.empty((len(rows), 53)), np.empty((len(rows), 72)), np.empty((len(rows), 63))
    sizes, paths, fallback, max_bbox_difference = {}, [], 0, 0.
    for video, group in rows.groupby('video_id'):
        source = Path(group.iloc[0].source_annotation_path)
        root = ET.parse(source).getroot()
        sizes[video] = (int(root.findtext('./meta/task/original_size/width')),
                        int(root.findtext('./meta/task/original_size/height')))
        paths.append(source)
    bbox_names = contract()['A']
    for i, row in enumerate(rows.itertuples(index=False)):
        keys = [(str(row.video_id), str(row.ped_id), f) for f in json.loads(row.observed_frames_json)]
        xy, conf = np.array([poses[k][0] for k in keys]), np.array([poses[k][1] for k in keys])
        boxes = np.array(json.loads(row.bbox_sequence_json), dtype=float)
        if boxes.shape != (16, 4) or not np.isfinite(boxes).all():
            raise ValueError('invalid observation BBoxes')
        features = helper.bbox_core._bbox53(boxes)
        historical = old_bbox.compute_bbox_motion_features(boxes)
        if list(historical) != bbox_names:
            raise ValueError('historical BBox column ordering differs')
        bb[i] = [features[n] for n in bbox_names]
        old = np.array([historical[n] for n in bbox_names])
        np.testing.assert_allclose(bb[i], old, rtol=1e-9, atol=1e-8)
        max_bbox_difference = max(max_bbox_difference, float(np.max(np.abs(bb[i] - old))))
        first, names1 = extractor._extract_biu_v1(xy, conf)
        second, names2, direction = extractor._extract_biu_v2(xy, boxes.astype(np.float32), sizes[row.video_id])
        if tuple(names1) != V1_NAMES or tuple(names2) != V2_NAMES:
            raise ValueError('BIU schema changed')
        v1[i], v2[i] = first, second
        fallback += int(direction == 'bbox_center_x')
        if (i + 1) % 1000 == 0:
            print(f'{split}: cached-pose feature assembly {i + 1}/{len(rows)}', flush=True)
    lookup = {identity: i for i, identity in enumerate(identities(rows))}
    result, outputs = {}, []
    cache.mkdir(parents=True, exist_ok=True)
    for cohort in COHORTS:
        target = populations[cohort]
        selected = np.array([lookup[k] for k in identities(target)])
        for i, j in enumerate(selected):
            if json.loads(target.iloc[i].bbox_sequence_json) != json.loads(rows.iloc[j].bbox_sequence_json):
                raise ValueError('beh/all BBox mismatch')
        arrays = dict(bbox=bb[selected], v1=v1[selected], v2=v2[selected])
        path = cache / f'{cohort}.npz'
        with Path(str(path) + '.tmp').open('wb') as handle:
            np.savez(handle, **arrays)
        os.replace(str(path) + '.tmp', path)
        row_path = cache / f'{cohort}_rows.csv'
        write(row_path, target)
        outputs += [path, row_path]
        result[cohort] = (target, assemble(**dict(bb=arrays['bbox'], v1=arrays['v1'], v2=arrays['v2'])))
    sources = [EXP083 / 'manifests' / f'{c}_{split}.csv' for c in COHORTS]
    sources += [Path(s['path']) for s in coverage.POSE_SOURCES] + paths
    audit = dict(split=split, unique_windows_computed=len(rows), required_pose_frames=len(needed),
                 pose_contributions=contributions, bbox_max_difference=max_bbox_difference,
                 bbox_fallback_all_count=fallback, image_dimensions=sizes,
                 exact_beh_subset_reuse=True, no_pose_estimator_rerun=True,
                 raw_endpoint_distance_differs_from_nominal={c: int(((r.event_frame-r.end_frame) != r.tte_frames).sum())
                                                            for c, r in populations.items()},
                 source_records=[record(p) for p in dict.fromkeys(sources)])
    write(audit_path, audit)
    publish(phase, outputs + [audit_path, *sources])
    return result, audit


def historical_probability(rows, seed, split):
    if split == 'test':
        authenticate('validate')
    # Test probability strings are never converted/retained during validation.
    records = []
    with OLD_PREDICTIONS.open() as handle:
        for r in csv.DictReader(handle):
            if r['protocol'] == 'api_style_reconstruction' and r['prediction_split'] == split and int(r['seed']) == seed:
                if r['model_name'] != 'xgb' or r['representation'] != 'bbox':
                    raise ValueError('historical selected candidate is not BBox XGBoost')
                records.append(r)
    old = pd.DataFrame(records)
    for n in ('start_frame', 'end_frame', 'label'):
        old[n] = old[n].astype(int)
    fields = ['video_id', 'ped_id', 'start_frame', 'end_frame', 'label']
    lookup = old.set_index(fields)
    if not lookup.index.is_unique or set(lookup.index) != set(pd.MultiIndex.from_frame(rows[fields])):
        raise ValueError('historical prediction identity mismatch')
    aligned = lookup.loc[pd.MultiIndex.from_frame(rows[fields])]
    p = aligned.probability.to_numpy(float)
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError('invalid saved probability')
    return p


def reuse_audit(development):
    old = pd.read_csv(OLD_WINDOWS)
    rows = []
    for split, values in development.items():
        target = values['jaad_all'][0]
        previous = old.loc[old.split.eq(split)].reset_index(drop=True)
        if identities(target) != identities(previous):
            raise ValueError('Exp031 training/validation identity/order differs')
        for column in ('event_frame', 'tte_frames'):
            np.testing.assert_array_equal(target[column], previous[column])
        for a, b in zip(target.bbox_sequence_json, previous.bbox_sequence_json):
            np.testing.assert_array_equal(json.loads(a), json.loads(b))
        rows.append(dict(split=split, samples=len(target), exact_order_labels_frames_boxes=True))
    tree = ast.parse(OLD_BUILDER.read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'XGBClassifier']
    assert len(calls) == 1
    recipe = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}
    expected = dict(n_estimators='250', max_depth='4', learning_rate='0.05', subsample='0.8',
                    colsample_bytree='0.8', reg_lambda='1.0', objective="'binary:logistic'",
                    eval_metric="'logloss'", random_state='seed', n_jobs='4', scale_pos_weight='scale_pos_weight')
    if recipe != expected:
        raise ValueError('historical estimator recipe differs')
    for seed in SEEDS:
        historical_probability(development['val']['jaad_all'][0], seed, 'val')
    audit = dict(status='reuse_three_saved_prediction_baselines', source_experiment=31,
                 arm='A', cohort='jaad_all', seeds=SEEDS, population_checks=rows,
                 feature_order_and_values_verified=True, estimator_recipe=recipe,
                 historical_threshold_not_reused=True, checkpoints_available=False,
                 limitation='No original model file/runtime lock; parity is reconstructed from source and saved probabilities, not live checkpoint reproduction.',
                 records=[record(p) for p in (OLD_WINDOWS, OLD_PREDICTIONS, OLD_BUILDER, OLD_RUNNER, OLD_BBOX)])
    write(ROOT / 'metrics/reuse_audit.json', audit)
    return audit


def clean_development_parity(development):
    checks, sources = [], []
    for split in ('train', 'val'):
        clean, old, names, audit = helper.load_split('validation' if split == 'val' else split)
        rows, arrays = development[split]['jaad_all']
        lookup = {k: i for i, k in enumerate(identities(rows))}
        paired = [(i, lookup[k]) for i, k in enumerate(identities(clean)) if k in lookup]
        if not paired:
            raise ValueError('no exact shared clean/API development windows for extraction parity')
        left, right = np.array(paired).T
        errors = {}
        for old_arm, new_arm in (('A', 'A'), ('B', 'B'), ('C', 'D'), ('D', 'E')):
            assert names[old_arm] == contract()[new_arm]
            np.testing.assert_allclose(old[old_arm][left], arrays[new_arm][right], rtol=1e-7, atol=1e-6)
            errors[new_arm] = float(np.max(np.abs(old[old_arm][left] - arrays[new_arm][right])))
        checks.append(dict(split=split, shared_windows=len(paired), maximum_absolute_differences=errors))
        sources += [Path(r['path']) for r in audit['sources']]
    path = ROOT / 'metrics/clean_development_parity.json'
    write(path, dict(checks=checks, source_records=[record(p) for p in dict.fromkeys(sources)]))
    return [path, *sources]


def prepare():
    if (ROOT / 'PREPARE_COMPLETE.json').exists():
        authenticate('prepare')
        return
    development, files = {}, []
    for split in ('train', 'val'):
        development[split], audit = load_split(split)
        files += [ROOT / f'MATERIALIZE_{split.upper()}_COMPLETE.json', ROOT / 'metrics' / f'{split}_source_audit.json']
        files += [p for p in (ROOT / 'cache' / split).iterdir() if p.is_file()]
        files += [Path(r['path']) for r in audit['source_records']]
    for cohort in COHORTS:
        helper.split_disjoint(development['train'][cohort][0], development['val'][cohort][0])
    files += clean_development_parity(development)
    reuse = reuse_audit(development)
    files += [Path(r['path']) for r in reuse['records']]
    files += [ROOT / 'metrics/reuse_audit.json', ROOT / 'REUSE_AUDIT.md', ROOT / 'DESIGN.md',
              Path(__file__), Path(__file__).with_name('test_exp102.py'), P100, P089, V2_METADATA,
              support.PARENT_SCRIPT, support.parent.HELPER, helper.CORE_PATH, helper.BBOX_PATH,
              helper.DEFINITIONS, REPO / 'src/biu_v2_extractor.py', REPO / 'src/features/biu_features.py',
              EXP083 / 'scripts/coverage.py', EXP083 / 'scripts/contract.py', EXP083 / 'pre_registration.json',
              EXP083 / 'metrics/.protocol-audit.complete.json']
    for name, value in [('feature_contract.json', contract()), ('freeze.json', dict(
            source_records=[record(p) for p in dict.fromkeys(files)],
            new_fits=27, reused_prediction_runs=3, test_features_accessed=False,
            python=sys.version, numpy=np.__version__, xgboost=helper.xgboost.__version__))]:
        p = ROOT / 'metrics' / name
        write(p, value)
        files.append(p)
    publish('prepare', files)


def summaries(table):
    return pd.concat([helper.summarize(g).assign(cohort=c) for c, g in table.groupby('cohort')], ignore_index=True)


def is_reused(cohort, arm):
    return cohort == 'jaad_all' and arm == 'A'


def predict(rows, arrays, cohort, arm, seed, split):
    if is_reused(cohort, arm):
        return historical_probability(rows, seed, split)
    model = joblib.load(ROOT / 'models' / f'{cohort}_{arm}_{seed}.joblib')
    return model.predict_proba(arrays[arm])[:, 1].astype(float)


def save_prediction(rows, p, cohort, arm, seed, split):
    path = ROOT / 'predictions' / f'{cohort}_{split}_{arm}_{seed}.csv'
    write(path, helper.predictions(rows, p, arm, seed).assign(cohort=cohort))
    metrics = dict(cohort=cohort, scope=split, arm=arm, seed=seed, feature_count=len(contract()[arm]),
                   source='Exp031_saved_probabilities' if is_reused(cohort, arm) else 'fresh',
                   **core.compute_metrics(rows.label.to_numpy(), p))
    return path, metrics


def validate():
    authenticate('prepare')
    if (ROOT / 'VALIDATE_COMPLETE.json').exists():
        authenticate('validate')
        return
    train, _ = load_split('train')
    val, _ = load_split('val')
    results, files = [], []
    for cohort, arm, seed in itertools.product(COHORTS, ARMS, SEEDS):
        rows, arrays = train[cohort]
        if not is_reused(cohort, arm):
            path = ROOT / 'models' / f'{cohort}_{arm}_{seed}.joblib'
            phase = f'fit_{cohort}_{arm}_{seed}'
            if (ROOT / f'{phase.upper()}_COMPLETE.json').exists():
                authenticate(phase)
            else:
                model = helper.bbox_core.build_xgb(seed, rows.label.to_numpy(), n_estimators=250)
                start = time.perf_counter()
                model.fit(arrays[arm], rows.label.to_numpy())
                path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, str(path) + '.tmp')
                os.replace(str(path) + '.tmp', path)
                publish(phase, [path])
                print(f'{cohort}/{arm}/{seed} fitted in {time.perf_counter()-start:.2f}s', flush=True)
            files += [path, ROOT / f'{phase.upper()}_COMPLETE.json']
        rows, arrays = val[cohort]
        p = predict(rows, arrays, cohort, arm, seed, 'val')
        path, values = save_prediction(rows, p, cohort, arm, seed, 'val')
        files.append(path)
        results.append(values)
    table = pd.DataFrame(results)
    check_grid(table)
    for name, value in [('validation_per_seed_results.csv', table), ('validation_summary_results.csv', summaries(table))]:
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
    train, _ = load_split('train')
    val, _ = load_split('val')
    test, audit = load_split('test')
    old_rows = pd.read_csv(OLD_WINDOWS)
    old_test = old_rows[old_rows.split.eq('test')].reset_index(drop=True)
    assert identities(old_test) == identities(test['jaad_all'][0])
    results, files, probabilities, deltas, transitions, intervals = [], [], {}, [], [], []
    for cohort in COHORTS:
        rows, arrays = test[cohort]
        helper.split_disjoint(train[cohort][0], val[cohort][0], rows)
        for arm, seed in itertools.product(ARMS, SEEDS):
            p = predict(rows, arrays, cohort, arm, seed, 'test')
            probabilities[(cohort, arm, seed)] = p
            path, values = save_prediction(rows, p, cohort, arm, seed, 'test')
            results.append(values)
            files.append(path)
    table = pd.DataFrame(results)
    check_grid(table)
    index = table.set_index(['cohort', 'arm', 'seed'])
    for cohort in COHORTS:
        rows, _ = test[cohort]
        y = rows.label.to_numpy()
        for left, right in itertools.combinations(ARMS, 2):
            for seed in SEEDS:
                key = dict(cohort=cohort, comparison=f'{right}_minus_{left}', seed=seed)
                deltas.append(dict(**key, **{'delta_' + m: index.loc[(cohort, right, seed), m] - index.loc[(cohort, left, seed), m]
                                             for m in helper.METRICS}))
                a = probabilities[(cohort, left, seed)] >= .5
                b = probabilities[(cohort, right, seed)] >= .5
                transitions.append(dict(**key, FP_corrected=int(((y == 0) & a & ~b).sum()),
                    FN_corrected=int(((y == 1) & ~a & b).sum()), FP_introduced=int(((y == 0) & ~a & b).sum()),
                    FN_introduced=int(((y == 1) & a & ~b).sum())))
        for left, right in BOOTSTRAP_PAIRS:
            print(f'Bootstrap {cohort}: {right}-{left}', flush=True)
            a = np.mean([probabilities[(cohort, left, s)] for s in SEEDS], axis=0)
            b = np.mean([probabilities[(cohort, right, s)] for s in SEEDS], axis=0)
            ci = support.parent.bootstrap(rows, a, b, replicates=5000)
            intervals.append(ci.assign(cohort=cohort, comparison=f'{right}_minus_{left}'))
    for name, value in [('per_seed_results.csv', table), ('fixed_0p5_results.csv', table),
                        ('summary_results.csv', summaries(table)), ('paired_deltas.csv', pd.DataFrame(deltas)),
                        ('error_transitions_fixed_0p5.csv', pd.DataFrame(transitions)),
                        ('ensemble_track_bootstrap.csv', pd.concat(intervals, ignore_index=True))]:
        path = ROOT / 'metrics' / name
        write(path, value)
        files.append(path)
    files += list((ROOT / 'cache/test').iterdir()) + [ROOT / 'MATERIALIZE_TEST_COMPLETE.json', ROOT / 'metrics/test_source_audit.json']
    files += [Path(r['path']) for r in audit['source_records']]
    authenticate('validate')
    publish('evaluate', files)


def report():
    authenticate('evaluate')
    t = pd.read_csv(ROOT / 'metrics/summary_results.csv')
    d = pd.read_csv(ROOT / 'metrics/paired_deltas.csv')
    lines = ['# Exp102 Final Report', '', 'API-style Exp083 JAAD cohorts, fixed0.5, seeds42/43/44.',
             'A=BBox; B=+v1-72; C=+v2-core56; D=+v2-full63; E=+complete v1 and full63.',
             '27 new250-tree XGBoost fits; three Exp031 JAAD_all/A prediction sets reused.', '',
             '| Cohort | Arm | Features | Accuracy | F1 mean +/- SD | PR-AUC mean +/- SD | ROC-AUC mean +/- SD |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in t.to_dict('records'):
        line = [r['cohort'], r['arm'], str(r['feature_count']), f'{r["accuracy_mean"]:.6f}']
        line += [f'{r[m+"_mean"]:.6f} +/- {r[m+"_std"]:.6f}' for m in ('f1', 'pr_auc', 'roc_auc')]
        lines.append('| ' + ' | '.join(line) + ' |')
    lines += ['', '## Paired directions', '']
    for (cohort, comparison), group in d.groupby(['cohort', 'comparison']):
        values = [f'{m} {group["delta_"+m].mean():+.6f} ({int((group["delta_"+m]>0).sum())}/3 favorable)' for m in ('f1', 'pr_auc', 'roc_auc')]
        lines.append(f'- {cohort} {comparison}: ' + '; '.join(values) + '.')
    clean_path = helper.ROOT / 'metrics/summary_results.csv'
    clean = pd.read_csv(clean_path)
    clean['exp102_arm'] = clean.arm.map(dict(A='A', B='B', C='D', D='E'))
    historical = ROOT / 'metrics/clean_exp098_historical_reference.csv'
    write(historical, clean.assign(source_experiment=98, population='clean_event_aligned'))
    lines += ['', '## Historical clean-cohort reference', '',
              'metrics/clean_exp098_historical_reference.csv records the completed Exp098 scores.',
              'Exp098 A/B/C/D map to Exp102 A/B/D/E respectively; Exp098 has no core56 arm.',
              'Exact shared development rows passed extraction/schema parity. These are separate',
              'training/test populations, not a matched-sample causal estimate of sampling effects.', '',
              '## Boundaries', '',
              'This tests frozen representation bundles, including existing v1 validity counters',
              'and v2 metadata. Core56 is an exact subset of full63 from the same extraction.',
              'No new ego, road, temporal or confidence stream was added. No raw pose estimator ran.',
              'The original train/validation/test populations remain unchanged, including observed',
              'crossing, unverified endpoints, non-behavior negatives and two nonconsecutive raw-frame',
              'training windows. API TTE counts track observations to an endpoint, not necessarily',
              'raw frames or time to a verified crossing onset; see per-split source audits.',
              'These cohorts have historical exposure. Published-protocol alignment is not proof',
              'of row or threshold identity to another paper. Do not select a universal BIU winner',
              'or compare clean/API absolute metrics as if they shared the same population.', '',
              'JAAD_all/A probabilities are reused from Exp031 after reconstructed source, cohort,',
              'ordering and recipe checks. Its original model files and runtime lock are unavailable;',
              'this is not live checkpoint reproduction. See REUSE_AUDIT.md and metrics/reuse_audit.json.',
              'Bootstrap intervals are descriptive, not multiplicity-adjusted, and operate on',
              'seed-ensemble probabilities rather than mean-per-seed metrics.', '',
              'Exact sources/hashes: metrics/freeze.json and *_source_audit.json.',
              'Execution commands: logs/commands.log; persistent status: logs/progress.log.', '']
    path = ROOT / 'FINAL_REPORT.md'
    write(path, '\n'.join(lines))
    publish('report', [path, historical, clean_path])


def verify():
    for phase in ('prepare', 'validate', 'evaluate', 'report'):
        authenticate(phase)
    files = []
    for split in ('val', 'test'):
        data, _ = load_split(split)
        table = pd.read_csv(ROOT / 'metrics' / ('validation_per_seed_results.csv' if split == 'val' else 'per_seed_results.csv'), float_precision='round_trip')
        check_grid(table)
        for cohort, arm, seed in itertools.product(COHORTS, ARMS, SEEDS):
            rows, arrays = data[cohort]
            path = ROOT / 'predictions' / f'{cohort}_{split}_{arm}_{seed}.csv'
            frame = pd.read_csv(path, float_precision='round_trip')
            assert identities(frame) == identities(rows)
            np.testing.assert_array_equal(frame.probability, predict(rows, arrays, cohort, arm, seed, split))
            np.testing.assert_array_equal(frame.prediction, (frame.probability >= .5).astype(int))
            metrics = core.compute_metrics(frame.label.to_numpy(), frame.probability.to_numpy())
            saved = table.loc[table.cohort.eq(cohort) & table.arm.eq(arm) & table.seed.eq(seed)].iloc[0]
            for name, value in metrics.items():
                np.testing.assert_allclose(value, saved[name], rtol=1e-12, atol=1e-12)
            files.append(record(path))
    models = list((ROOT / 'models').glob('*.joblib'))
    if len(models) != 27 or len(files) != 60:
        raise ValueError('unexpected new-fit or prediction count')
    write(ROOT / 'metrics/verification.json', dict(status='passed', new_models=27, reused_runs=3, predictions=files))
    write(ROOT / 'EXPERIMENT_COMPLETE.json', dict(experiment=102, status='complete', new_models=27, reused_runs=3,
          verification=record(ROOT / 'metrics/verification.json'), report=record(ROOT / 'FINAL_REPORT.md')))
    print('All60 prediction files verified; Exp102 complete.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['all', 'prepare', 'validate', 'evaluate', 'report', 'verify'])
    args = parser.parse_args()
    logs = ROOT / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / 'commands.log').open('a') as f:
        f.write(sys.executable + ' ' + ' '.join(sys.argv) + '\n')
    phases = ['prepare', 'validate', 'evaluate', 'report', 'verify'] if args.phase == 'all' else [args.phase]
    for phase in phases:
        for event in ('started', 'complete'):
            if event == 'complete':
                globals()[phase]()
            message = time.strftime('%Y-%m-%dT%H:%M:%S%z') + f' {phase} {event}'
            print(message, flush=True)
            with (logs / 'progress.log').open('a') as f:
                f.write(message + '\n')


if __name__ == '__main__':
    main()
