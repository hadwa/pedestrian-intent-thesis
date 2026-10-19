"""Frozen, resumable completion of missing standalone BIU-v2-core tree arms."""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score,
    balanced_accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))
SEEDS = (42, 43, 44)
CASES = {
    'jaad_clean_simple': 109, 'psi_simple': 109, 'pie_road': 195,
    'jaad_clean_road': 190, 'jaad_beh_road': 190, 'jaad_all_road': 190,
    'jaad_clean_road_vehicle': 205, 'jaad_beh_road_vehicle': 205,
    'jaad_all_road_vehicle': 205, 'psi_ego': 135, 'psi_road_ego': 195,
}
SOURCES = {}
TEST = False
AUDITS = []


def exp(n):
    paths = list((REPO / 'experiments').glob(f'{n:03d}_*'))
    assert len(paths) == 1, (n, paths)
    return paths[0]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def source(p):
    p = Path(p).resolve(strict=True)
    parts = p.relative_to(REPO).parts if p.is_relative_to(REPO) else p.parts
    if not TEST and any('test' in part.lower() for part in parts):
        raise RuntimeError(f'test access before frozen final gate: {p}')
    key = str(p)
    if key not in SOURCES:
        SOURCES[key] = dict(sha256=sha(p), bytes=p.stat().st_size)
    return p


def readj(p):
    return json.loads(source(p).read_text())


def readcsv(p):
    return pd.read_csv(source(p))


def readnp(p):
    return np.load(source(p), allow_pickle=False)


def writej(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(tmp, p)


def log(msg):
    line = time.strftime('%Y-%m-%d %H:%M:%S') + ' ' + str(msg)
    print(line, flush=True)
    with (ROOT / 'logs/run.log').open('a') as f:
        f.write(line + '\n')


def module(p, name):
    spec = importlib.util.spec_from_file_location(name, source(p))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


@lru_cache(None)
def definitions():
    d = readcsv(exp(12) / 'metrics/pie_biu_v2_feature_set_definitions.csv').set_index('feature_set')
    box, v1, v2 = [json.loads(d.loc[k, 'feature_columns_json'])
                   for k in ('bbox_only', 'biu_v1_only', 'biu_v2_core')]
    assert (len(box), len(v1), len(v2)) == (53, 72, 56)
    return box, v1, ['v2__' + n for n in v2]


def aligned(rows, other, subset=False):
    assert not rows.sequence_id.duplicated().any()
    assert not other.sequence_id.duplicated().any()
    if subset:
        assert set(rows.sequence_id) <= set(other.sequence_id)
    else:
        assert set(rows.sequence_id) == set(other.sequence_id)
    out = other.set_index('sequence_id').loc[rows.sequence_id].reset_index()
    for c in ('sequence_id', 'label', 'video_id', 'track_id', 'start_frame', 'end_frame', 'target_frame'):
        if c in rows and c in out:
            np.testing.assert_array_equal(rows[c], out[c], err_msg=c)
    if 'observed_frames_json' in rows and 'observed_frames_json' in out:
        assert rows.observed_frames_json.map(json.loads).tolist() == out.observed_frames_json.map(json.loads).tolist()
    return out


def subset_array(rows, other, x):
    aligned(rows, other, subset=True)
    return x[other.set_index('sequence_id').index.get_indexer(rows.sequence_id)]


def select(x, names, wanted):
    assert len(names) == len(set(names)) == x.shape[1]
    return x[:, [names.index(c) for c in wanted]]


@lru_cache(None)
def simple_psi(split):
    p = exp(113)
    folder = p / ('test_evaluation/cache' if split == 'test' else f'cache/{"val" if split == "validation" else split}')
    rows = readcsv(folder / 'rows.csv')
    names = readj(p / 'feature_contract.json')['C']
    x = readnp(folder / 'C.npy')
    return rows, x, names


@lru_cache(None)
def simple_jaad(split):
    p = exp(89) / 'cache' / ('test_source' if split == 'test' else 'trainval_source')
    allrows = readcsv(p / 'rows.csv')
    mask = allrows.split.eq('val' if split == 'validation' else split).to_numpy()
    rows = allrows.loc[mask].reset_index(drop=True)
    parent = aligned(rows, readcsv(exp(77) / f'metrics/prepared_jaad_{split}.csv'))
    names = definitions()[0] + ['v2__' + n for n in readj(p / 'metadata.json')['biu_v2_feature_names']]
    x = np.column_stack((parent[definitions()[0]].to_numpy(float), readnp(p / 'biu_v2.npy')[mask].astype(float)))
    return rows, x, names


@lru_cache(None)
def api_v2(cohort, split):
    """Recover only missing static values with the exact historical pose ABI."""
    f = module(exp(83) / 'scripts/features.py', 'exp116_features083')
    rows = readcsv(exp(83) / f'manifests/{cohort}_{"val" if split == "validation" else split}.csv')
    needed = f._needed_pose_keys(rows)
    for item in f.COVERAGE.POSE_SOURCES:
        source(item['path'])
    log(f'recover {cohort} {split}: {len(rows)} windows, {len(needed)} cached pose frames')
    poses, audit = f.load_pose_index(needed)
    from src.biu_v2_extractor import extract_biu_v2
    source(REPO / 'src/biu_v2_extractor.py')
    vectors = []
    for i, row in enumerate(rows.itertuples(index=False)):
        keys = [('jaad', str(row.video_id), str(row.ped_id), int(n)) for n in json.loads(row.observed_frames_json)]
        xy = np.asarray([poses[k]['keypoints'] for k in keys], dtype=float)
        confidence = np.asarray([poses[k]['kp_conf'] for k in keys], dtype=float)
        boxes = np.asarray(json.loads(row.bbox_sequence_json), dtype=float)
        v = extract_biu_v2(xy, confidence, bbox_sequence=boxes)
        values = [v[n.removeprefix('v2__')] for n in definitions()[2]]
        if i == 0:
            twice = extract_biu_v2(xy, confidence, bbox_sequence=boxes)
            assert values == [twice[n.removeprefix('v2__')] for n in definitions()[2]]
        vectors.append(values)
        if i and i % 1000 == 0:
            log(f'recover {cohort} {split}: {i}/{len(rows)}')
    AUDITS.append(dict(case=cohort, split=split, pose_recovery=audit))
    return rows, np.asarray(vectors)


def replace_combined(x, oldnames, core, ordered=None):
    box, v1, v2 = definitions()
    shared = [n for n in v2 if n in oldnames]
    assert len(shared) == 36, len(shared)
    np.testing.assert_allclose(select(x, oldnames, shared), select(core, v2, shared), rtol=1e-9, atol=1e-8)
    remaining = [n for n in oldnames if n not in v1 and n not in v2 and n not in box]
    assert all(n.startswith(('temporal__', 'obd_', 'road_', 'vehicle_')) for n in remaining), remaining
    names = ordered or box + v2 + remaining
    new = np.column_stack([core[:, v2.index(n)] if n in v2 else x[:, oldnames.index(n)] for n in names]).astype(x.dtype)
    for n in box + remaining:
        np.testing.assert_array_equal(new[:, names.index(n)], x[:, oldnames.index(n)])
    return new, names


def parent_paths(case, seed, split):
    if case == 'jaad_clean_simple':
        return exp(98) / f'models/C_seed{seed}.joblib', exp(98) / f'predictions/{split}_C_seed{seed}.csv'
    if case == 'psi_simple':
        p = exp(113)
        return p / f'models/C_seed{seed}.joblib', p / ('test_evaluation' if split == 'test' else '') / f'predictions/{split}_C_seed{seed}.csv'
    if case == 'pie_road':
        return exp(75) / f'models/candidate_seed{seed}.joblib', exp(75) / f'predictions/{split}_candidate_seed{seed}.csv'
    if case.startswith('psi_'):
        arm = 'S187' if case == 'psi_ego' else 'R247'
        p = exp(114)
        return p / f'models/{arm}_seed{seed}.joblib', p / ('test_evaluation' if split == 'test' else '') / f'predictions/{split}_{arm}_seed{seed}.csv'
    cohort = case.split('_road')[0]
    if case.endswith('_vehicle'):
        model = exp(115) / f'models/{cohort}_seed{seed}.joblib'
        arm = 'RV257'
    else:
        model = exp(81) / f'models/variant_D_seed{seed}.joblib' if cohort == 'jaad_clean' else exp(83) / f'models/{cohort}_seed{seed}.joblib'
        arm = 'R242'
    return model, exp(115) / f'predictions/{cohort}_{split}_{arm}_seed{seed}.csv'


def build(case, split):
    if split == 'test':
        assert TEST and (ROOT / 'TRAINED.json').exists()
    box, _, v2 = definitions()
    if case in ('psi_simple', 'jaad_clean_simple'):
        rows, old, oldnames = (simple_psi if case == 'psi_simple' else simple_jaad)(split)
        names = box + v2
        new = select(old, oldnames, names)
    elif case == 'pie_road':
        frame = readcsv(exp(75) / f'metrics/prepared_{split}.csv').sort_values('sequence_id', kind='stable').reset_index(drop=True)
        oldnames = list(joblib.load(source(parent_paths(case, 42, 'validation')[0])).feature_names_in_)
        old = frame[oldnames].to_numpy(float)
        rows = frame.drop(columns=[c for c in oldnames if c not in ('sequence_id', 'label')])
        fc = module(exp(96) / 'scripts/feature_contract.py', 'exp116_fc096')
        ordered = list(fc.build_matched_arm_columns(source(exp(12) / 'metrics/pie_biu_v2_feature_set_definitions.csv'), source(exp(37) / 'metrics/feature_contract.json'))['v2'])
        ordered += [n for n in oldnames if n.startswith('road__')]
        new, names = replace_combined(old, oldnames, frame[v2].to_numpy(float), ordered)
    elif case.startswith('psi_'):
        p = exp(114)
        folder = p / ('test_evaluation/cache/features' if split == 'test' else f'cache/features/{"val" if split == "validation" else split}')
        arm = 'S187' if case == 'psi_ego' else 'R247'
        rows, old = readcsv(folder / 'rows.csv'), readnp(folder / f'{arm}.npy')
        oldnames = readj(p / 'feature_contract.json')[arm]
        sr, sx, sn = simple_psi(split)
        core = subset_array(rows, sr, select(sx, sn, v2)).astype(old.dtype)
        np.testing.assert_array_equal(select(old, oldnames, box), subset_array(rows, sr, select(sx, sn, box)).astype(old.dtype))
        new, names = replace_combined(old, oldnames, core)
    else:
        cohort = case.split('_road')[0]
        p = exp(115)
        rows = readcsv(p / f'cache/{cohort}_{split}_rows.csv')
        z = readnp(p / f'cache/{cohort}_{split}.npz')
        np.testing.assert_array_equal(rows.sequence_id.astype(str), z['sequence_id'].astype(str))
        np.testing.assert_array_equal(rows.label, z['label'])
        arm = 'RV257' if case.endswith('_vehicle') else 'R242'
        oldnames = readj(p / 'feature_contract.json')[arm]
        old = z['X'][:, :len(oldnames)]
        if cohort == 'jaad_clean':
            q = exp(100) / f'cache/{split}'
            sr = readcsv(q / 'rows.csv')
            sx = readnp(q / 'features.npz')['v2']
            sn = readj(exp(100) / 'metrics/feature_contract.json')['v2']
            core = subset_array(rows, sr, select(sx, sn, v2))
        else:
            sr, sx = api_v2(cohort, split)
            # Exp083 write_predictor_cache quantized every feature to float32.
            # Exp115 later promoted that matrix while appending vehicle state.
            core = subset_array(rows, sr, sx).astype(np.float32).astype(old.dtype)
        new, names = replace_combined(old, oldnames, core)
    assert new.shape == (len(rows), CASES[case])
    assert len(names) == len(set(names)) == new.shape[1]
    assert np.isfinite(new).all() and np.isfinite(old).all()
    assert set(rows.label.unique()) == {0, 1} and not rows.sequence_id.duplicated().any()
    frames = None
    if 'observed_frames_json' in rows:
        frames = np.array(rows.observed_frames_json.map(json.loads).tolist())
        assert frames.shape == (len(rows), 15 if case.startswith('psi_') else 16)
        if case.startswith('psi_'):
            assert (np.diff(frames, axis=1) == 1).all()
            np.testing.assert_array_equal(frames[:, -1] + 1, rows.target_frame)
            np.testing.assert_array_equal(rows.label, (rows.annotation_vote >= .5).astype(int))
        elif case.startswith('jaad_clean'):
            assert (np.diff(frames, axis=1) == 1).all()
    if case == 'pie_road':
        assert rows.observation_length.eq(16).all() and (rows.end_frame - rows.start_frame).eq(15).all()
        assert rows.time_to_event_frames.between(30, 60).all()
        assert set(rows.set_id) == {'train': {'set01', 'set02', 'set06'}, 'validation': {'set04', 'set05'}, 'test': {'set03'}}[split]
    if case.startswith('jaad_clean'):
        assert rows.tte_frames.between(30, 60).all()
    folder = ROOT / 'cache' / case / split
    folder.mkdir(parents=True, exist_ok=True)
    # Cache the exact input matrices; no future boxes, labels, or identities are predictors.
    np.save(folder / 'features.npy', new, allow_pickle=False)
    metadata = [c for c in rows if not (c.startswith(('v2__', 'early__', 'late__', 'temporal__', 'road__')) or c in definitions()[1])]
    rows = rows[metadata].copy()
    rows.to_csv(folder / 'rows.csv', index=False)
    writej(folder / 'columns.json', names)
    AUDITS.append(dict(case=case, split=split, rows=len(rows), positives=int(rows.label.sum()),
                       dimensions=len(names), parent_dimensions=len(oldnames), finite=True,
                       non_biu_unchanged=True, frame_gap_windows=int((np.diff(frames, axis=1) != 1).any(axis=1).sum()) if frames is not None else 0))
    return rows, new, names, old, oldnames


def infer(m, x, names):
    if hasattr(m, 'feature_names_in_'):
        assert list(m.feature_names_in_) == names
        x = pd.DataFrame(x, columns=names)
    return m.predict_proba(x)[:, 1]


def replay(case, seed, split, rows, old, names):
    mp, pp = parent_paths(case, seed, split)
    m = joblib.load(source(mp))
    saved = aligned(rows, readcsv(pp))
    p = infer(m, old, names)
    np.testing.assert_allclose(p, saved.probability, rtol=0, atol=1e-7)
    AUDITS.append(dict(case=case, seed=seed, split=split, parent_replay_max_abs=float(np.max(np.abs(p - saved.probability)))))
    return m, saved.probability.to_numpy()


def metrics(y, p):
    yp = np.asarray(p) >= .5
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    return dict(accuracy=accuracy_score(y, yp), precision=precision_score(y, yp, zero_division=0),
                recall=recall_score(y, yp, zero_division=0), f1=f1_score(y, yp, zero_division=0),
                pr_auc=average_precision_score(y, p), roc_auc=roc_auc_score(y, p),
                macro_f1=f1_score(y, yp, average='macro'), balanced_accuracy=balanced_accuracy_score(y, yp),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                high_conf_fp=int(((np.asarray(y) == 0) & (p >= .9)).sum()),
                high_conf_fn=int(((np.asarray(y) == 1) & (p <= .1)).sum()))


def savepred(case, seed, split, rows, p):
    out = rows[[c for c in ('sequence_id', 'video_id', 'track_id', 'label', 'observed_frames_json', 'start_frame', 'end_frame', 'target_frame') if c in rows]].copy()
    out['probability'], out['prediction'] = p, (p >= .5).astype(int)
    out['threshold'], out['seed'], out['case'] = .5, seed, case
    path = ROOT / f'predictions/{case}_{split}_seed{seed}.csv'
    out.to_csv(path, index=False)
    return path


def disjoint(case, populations):
    for a, b in itertools.combinations(populations, 2):
        for c in ('sequence_id', 'video_id', 'track_id'):
            if case == 'pie_road' and c != 'sequence_id':
                left, right = set(zip(a.set_id, a[c])), set(zip(b.set_id, b[c]))
            else:
                left, right = set(a[c]), set(b[c])
            assert not left & right, (case, 'split overlap', c)


def checkpoint_inputs():
    return {str(p.relative_to(ROOT)): sha(p) for p in [ROOT / 'DESIGN.md', Path(__file__)]}


def main():
    global TEST
    for d in ('cache', 'models', 'metrics', 'predictions', 'logs'):
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    frozen = dict(cases=CASES, seeds=list(SEEDS), threshold=.5, code_design=checkpoint_inputs())
    fp = ROOT / 'FROZEN.json'
    if fp.exists():
        previous = json.loads(fp.read_text())
        if previous != frozen:
            assert not list((ROOT / 'models').glob('*.joblib')) and not (ROOT / 'TRAINED.json').exists(), 'cannot amend after training'
            writej(ROOT / f'logs/pretraining_freeze_{sha(fp)[:12]}.json', previous)
            writej(fp, frozen)
            log('Pretraining implementation repair recorded; no fitted models or new validation results exist')
    else:
        writej(fp, frozen)
    log('Frozen Exp116: prepare all train/validation inputs; test gate CLOSED')
    prepared = {}
    for case in CASES:
        prepared[case] = {}
        for split in ('train', 'validation'):
            prepared[case][split] = build(case, split)
        tr, va = prepared[case]['train'], prepared[case]['validation']
        disjoint(case, [tr[0], va[0]])
        assert tr[2] == va[2] and tr[4] == va[4]
        for seed in SEEDS:
            m, _ = replay(case, seed, 'validation', va[0], va[3], va[4])
            params = m.steps[-1][1].get_params()
            assert params['random_state'] == seed and params['n_jobs'] == 4
            assert params['n_estimators'] == 250 and params['max_depth'] == 4
            np.testing.assert_allclose(params['scale_pos_weight'], (tr[0].label == 0).sum() / tr[0].label.sum(), rtol=1e-12)
        log(f'PASS {case}: train={len(tr[0])}, validation={len(va[0])}, features={len(tr[2])}; three parent replays match')
    writej(ROOT / 'metrics/pretraining_invariants.json', AUDITS)
    writej(ROOT / 'metrics/trainval_sources.json', SOURCES)
    writej(ROOT / 'PREPARED.json', dict(code_design=checkpoint_inputs(), cases=list(CASES)))
    results, runtimes = [], []
    trained_artifacts = {}
    for case in CASES:
        tr, va = prepared[case]['train'], prepared[case]['validation']
        for seed in SEEDS:
            assert checkpoint_inputs() == frozen['code_design']
            path = ROOT / f'models/{case}_seed{seed}.joblib'
            done = ROOT / f'models/{case}_seed{seed}.json'
            parent, pp = replay(case, seed, 'validation', va[0], va[3], va[4])
            if done.exists():
                info = json.loads(done.read_text())
                assert info['sha256'] == sha(path)
                m = joblib.load(path)
            else:
                assert not path.exists(), f'uncommitted fit artifact: {path}'
                m = clone(parent)
                start = time.perf_counter()
                m.fit(pd.DataFrame(tr[1], columns=tr[2]), tr[0].label.to_numpy())
                info = dict(case=case, seed=seed, fit_seconds=time.perf_counter() - start)
                tmp = path.with_suffix('.tmp')
                joblib.dump(m, tmp)
                os.replace(tmp, path)
                info['sha256'] = sha(path)
                writej(done, info)
            p = infer(m, va[1], va[2])
            pred = savepred(case, seed, 'validation', va[0], p)
            runtimes.append(info)
            for arm, prob, dims in [('core56', p, len(va[2])), ('parent', pp, len(va[4]))]:
                results.append(dict(case=case, split='validation', seed=seed, arm=arm, dimensions=dims, threshold=.5, **metrics(va[0].label, prob)))
            trained_artifacts[str(path)] = sha(path)
            trained_artifacts[str(pred)] = sha(pred)
            log(f'FIT {case} seed={seed}: {info["fit_seconds"]:.2f}s; validation F1={results[-2]["f1"]:.6f}, AP={results[-2]["pr_auc"]:.6f}')
            pd.DataFrame(results).to_csv(ROOT / 'metrics/validation_per_seed.csv', index=False)
            pd.DataFrame(runtimes).to_csv(ROOT / 'metrics/fit_times.csv', index=False)
    assert len(runtimes) == 33
    writej(ROOT / 'TRAINED.json', dict(code_design=checkpoint_inputs(), artifacts=trained_artifacts))
    for p, digest in trained_artifacts.items():
        assert sha(p) == digest
    TEST = True
    log('All 33 models and validation predictions frozen; opening final test gate for every predeclared case')
    transitions = []
    for case in CASES:
        te = build(case, 'test')
        tr, va = prepared[case]['train'], prepared[case]['validation']
        assert te[2] == tr[2] and te[4] == tr[4]
        disjoint(case, [tr[0], va[0], te[0]])
        for seed in SEEDS:
            _, pp = replay(case, seed, 'test', te[0], te[3], te[4])
            m = joblib.load(ROOT / f'models/{case}_seed{seed}.joblib')
            p = infer(m, te[1], te[2])
            savepred(case, seed, 'test', te[0], p)
            for arm, prob, dims in [('core56', p, len(te[2])), ('parent', pp, len(te[4]))]:
                results.append(dict(case=case, split='test', seed=seed, arm=arm, dimensions=dims, threshold=.5, **metrics(te[0].label, prob)))
            oldok, newok = (pp >= .5) == te[0].label.to_numpy(), (p >= .5) == te[0].label.to_numpy()
            transitions.append(dict(case=case, seed=seed, corrected=int((~oldok & newok).sum()), harmed=int((oldok & ~newok).sum()),
                corrected_fp=int((~oldok & newok & (te[0].label.to_numpy() == 0)).sum()),
                corrected_fn=int((~oldok & newok & (te[0].label.to_numpy() == 1)).sum())))
        log(f'TEST {case}: all three seeds saved, parent replay and cohort invariants PASS')
    frame = pd.DataFrame(results)
    assert len(frame) == 132
    for case in CASES:
        for split, seed in itertools.product(('validation', 'test'), SEEDS):
            pred = pd.read_csv(ROOT / f'predictions/{case}_{split}_seed{seed}.csv')
            row = frame[(frame.case == case) & (frame.split == split) & (frame.seed == seed) & (frame.arm == 'core56')].iloc[0]
            for k, v in metrics(pred.label, pred.probability.to_numpy()).items():
                np.testing.assert_allclose(row[k], v, rtol=0, atol=1e-12)
    frame.to_csv(ROOT / 'metrics/per_seed_results.csv', index=False)
    frame.to_csv(ROOT / 'metrics/fixed_0p5_results.csv', index=False)
    keys = list(metrics(np.array([0, 1]), np.array([.2, .8])))
    summary = frame.groupby(['case', 'split', 'arm', 'dimensions'])[keys].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(ROOT / 'metrics/summary_results.csv', index=False)
    joined = frame[frame.arm == 'core56'].merge(frame[frame.arm == 'parent'], on=['case', 'split', 'seed'], suffixes=('_new', '_parent'))
    deltas = joined[['case', 'split', 'seed']].copy()
    for k in keys:
        deltas[k] = joined[k + '_new'] - joined[k + '_parent']
    deltas.to_csv(ROOT / 'metrics/paired_deltas.csv', index=False)
    pd.DataFrame(transitions).to_csv(ROOT / 'metrics/error_transitions.csv', index=False)
    writej(ROOT / 'metrics/all_invariants.json', AUDITS)
    writej(ROOT / 'metrics/all_sources.json', SOURCES)
    log('Verifying source immutability and all saved metrics')
    for p, info in SOURCES.items():
        assert sha(p) == info['sha256'], f'source changed: {p}'
    for p, digest in trained_artifacts.items():
        assert sha(p) == digest
    assert checkpoint_inputs() == frozen['code_design']
    lines = ['# Exp116 final results', '', 'Retrospective controlled completion; fixed threshold 0.5. AP = average precision.',
        'All 33 new fits, all parent probability replays, cohort and feature invariants, and saved-metric checks passed.', '',
        '| Case | Core56 F1 | Parent F1 | Core56 AP | Parent AP | Core56 ROC | Parent ROC |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for case in CASES:
        s = summary[(summary.case == case) & (summary.split == 'test')].set_index('arm')
        a, b = s.loc['core56'], s.loc['parent']
        lines.append(f'| {case} | {a.f1_mean:.6f} | {b.f1_mean:.6f} | {a.pr_auc_mean:.6f} | {b.pr_auc_mean:.6f} | {a.roc_auc_mean:.6f} | {b.roc_auc_mean:.6f} |')
    lines += ['', 'Full per-seed and mean/sample-SD tables: metrics/per_seed_results.csv and metrics/summary_results.csv.',
        'Paired deltas: metrics/paired_deltas.csv; transitions: metrics/error_transitions.csv.',
        'Simple parents use full63. Extended parents use historical combined static BIUs, not standalone full63.',
        'Existing temporal/pose-quality summaries are retained in extended arms. PSI is a different intent task.',
        'No neural reruns, test-driven selection, thesis edits, perception reruns or benchmark overwrites were performed.',
        f'Total measured tree fitting time: {sum(r["fit_seconds"] for r in runtimes):.2f} seconds.', '']
    (ROOT / 'FINAL_REPORT.md').write_text('\n'.join(lines))
    writej(ROOT / 'EXPERIMENT_COMPLETE.json', dict(fits=33, cases=11, test_gate_after_all_fits=True,
        artifacts={str(p.relative_to(ROOT)): sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.suffix in ('.csv', '.json', '.joblib', '.md', '.py') and p.name != 'EXPERIMENT_COMPLETE.json'}))
    log('EXPERIMENT_COMPLETE: all 11 cases x 3 seeds verified. STOP; no neural experiment.')


if __name__ == '__main__':
    main()
