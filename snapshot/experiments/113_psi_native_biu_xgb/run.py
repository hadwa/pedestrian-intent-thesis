"""PSI-native train/validation-only matched feature ablation."""
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def guard(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        path = str(args[0]).replace('\\', '/')
        if '/PSI2.0_Test/annotations/' in path or '/cache/test/' in path:
            raise RuntimeError('Test annotations/features are closed')


sys.addaudithook(guard)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


helper = module('psi113_helpers', REPO / 'experiments/098_jaad_matched_biu_xgb/scripts/run_experiment.py')
extractor = module('psi113_biu', REPO / 'experiments/089_pedcmt_standardized_jaad_biu_transfer/scripts/exp089_data.py')
write, record = helper.write, helper.record
POSE = REPO / 'data/psi_vitpose_s_all_tracks.jsonl'
EXPECTED_POSE = '31d8f45a7f8b61803b2dde8f981ce012086ac9905455d7aaac743bc1ec2a3fc7'
SEEDS = (42, 43, 44)


def extract(points, conf, boxes):
    assert points.shape == (15, 17, 2) and conf.shape == (15, 17) and boxes.shape == (15, 4)
    bbox_dict = helper.bbox_core._bbox53(boxes)
    bbox_names = tuple(bbox_dict)
    bbox = np.array([bbox_dict[n] for n in bbox_names], float)
    v1, n1 = extractor._extract_biu_v1(points, conf)
    v2, n2, fallback = extractor._extract_biu_v2(points, boxes, (1280, 720))
    vectors = dict(A=bbox, B=np.r_[bbox, v1], C=np.r_[bbox, v2])
    names = dict(A=bbox_names, B=bbox_names + tuple('v1__' + n for n in n1),
                 C=bbox_names + tuple('v2__' + n for n in n2))
    return vectors, names, fallback


def prepare():
    recovery = json.loads((ROOT / 'recovery.json').read_text())
    for r in recovery['source_code'] + recovery['annotations']:
        assert record(r['path'])['sha256'] == r['sha256']
    assert record(POSE)['sha256'] == EXPECTED_POSE
    splitpath = Path(recovery['split_path'])
    assert record(splitpath)['sha256'] == recovery['split_sha256']
    splits = json.loads(splitpath.read_text())
    assert not set(splits['train']) & set(splits['val'])
    # The unmodified extension operates only on recovered train/val directories.
    script = ROOT / 'official_code/dataset/extend_intent_annotation.py'
    with (ROOT / 'extension.log').open('w') as log:
        subprocess.run([sys.executable, str(script), str(ROOT / 'dataset')], check=True, stdout=log, stderr=subprocess.STDOUT)
    dbmod = module('psi113_database', ROOT / 'official_code/baseline/database/create_database.py')
    seqmod = module('psi113_sequence', ROOT / 'official_code/baseline/data/process_sequence.py')
    args = SimpleNamespace(dataset='PSI2.0', dataset_root_path=str(ROOT / 'dataset'), max_track_size=16,
                           intent_type='mean', intent_num=2)
    dev = set(splits['train'] + splits['val'])
    pose = {}
    for line in POSE.open():
        r = json.loads(line)
        if r['video'] not in dev:
            continue
        key = (r['video'], r['pid'], int(r['frame_idx']))
        assert key not in pose
        xy, conf = np.asarray(r['keypoints'], float), np.asarray(r['kp_conf'], float)
        assert xy.shape == (17, 2) and conf.shape == (17,) and np.isfinite(xy).all() and np.isfinite(conf).all()
        pose[key] = (xy, conf)
    audits, sources, contracts = [], [], None
    for split in ('train', 'val'):
        db_log = ROOT / f'{split}_database.log'
        db = dbmod.init_db(splits[split], str(db_log), args)
        dbmod.update_db_annotations(db, str(db_log), args)
        rows, arrays, fractions = [], {a: [] for a in ('A', 'B', 'C')}, []
        missing, fallback_count = 0, 0
        for video in sorted(db):
            for pid, track in sorted(db[video].items()):
                frames = track['frames']
                assert all(b == a + 1 for a, b in zip(frames, frames[1:]))
                labels, votes, _, _ = seqmod.get_intent(db, video, pid, args)
                original_pid = track['cv_annotations']['track_id'].split('-')[0]
                boxes = np.asarray(track['cv_annotations']['bbox'], float)
                assert np.isfinite(boxes).all() and (boxes[:, 2:] > boxes[:, :2]).all()
                for start in range(len(frames) - 15):
                    obs = frames[start:start + 15]
                    target = frames[start + 15]
                    assert target == obs[-1] + 1 and target not in obs
                    points, conf, present = [], [], []
                    for frame in obs:
                        value = pose.get((video, original_pid, frame))
                        present.append(value is not None)
                        xy, confidence = value if value is not None else (np.zeros((17, 2)), np.zeros(17))
                        points.append(xy); conf.append(confidence)
                    points, conf = np.array(points), np.array(conf)
                    valid = np.isfinite(points).all(2) & np.any(points != 0, axis=2)
                    bbox = boxes[start:start + 15]
                    vectors, names, fallback = extract(points, conf, bbox)
                    if contracts is None:
                        contracts = names
                    assert contracts == names
                    if len(rows) == 0:
                        again, _, _ = extract(points, conf, bbox)
                        for arm in vectors:
                            np.testing.assert_array_equal(vectors[arm], again[arm])
                    for arm in arrays:
                        arrays[arm].append(vectors[arm])
                    np.testing.assert_array_equal(vectors['A'], vectors['B'][:53])
                    np.testing.assert_array_equal(vectors['A'], vectors['C'][:53])
                    rows.append(dict(sequence_id=f'{video}:{pid}:{obs[0]}:{target}', video_id=video,
                        track_id=f'{video}:{original_pid}', ped_id=original_pid, observed_frames_json=json.dumps(obs),
                        target_frame=target, label=labels[start + 15], annotation_vote=votes[start + 15],
                        pose_complete=bool(all(present) and valid.all()), valid_joint_fraction=float(valid.mean()),
                        missing_pose_frames=15 - sum(present), pose_all_zero=bool(not valid.any())))
                    fractions.append(float(valid.mean()))
                    missing += 15 - sum(present)
                    fallback_count += int(fallback == 'bbox_center_x')
                print(f'{split} {video}/{pid}: {len(rows)} windows', flush=True)
        frame = pd.DataFrame(rows)
        assert len(frame) and frame.sequence_id.is_unique and set(frame.label) == {0, 1}
        directory = ROOT / 'cache' / split
        directory.mkdir(parents=True, exist_ok=True)
        write(directory / 'rows.csv', frame)
        for arm, dim in dict(A=53, B=125, C=116).items():
            x = np.array(arrays[arm], float)
            assert x.shape == (len(frame), dim) and np.isfinite(x).all()
            np.save(directory / f'{arm}.npy', x)
            sources.append(directory / f'{arm}.npy')
        sources.append(directory / 'rows.csv')
        audits.append(dict(split=split, windows=len(frame), positives=int(frame.label.sum()),
            videos=int(frame.video_id.nunique()), tracks=int(frame.track_id.nunique()),
            completely_missing_or_zero_windows=int(frame.pose_all_zero.sum()),
            complete_pose_windows=int(frame.pose_complete.sum()), missing_pose_frame_occurrences=missing,
            mean_valid_joint_fraction=float(np.mean(fractions)), median_valid_joint_fraction=float(np.median(fractions)),
            bbox_direction_fallback_count=fallback_count, bbox_direction_fallback_percent=100 * fallback_count / len(frame)))
    write(ROOT / 'metrics/pose_coverage.csv', pd.DataFrame(audits))
    write(ROOT / 'feature_contract.json', contracts)
    sources += [Path(__file__), ROOT / 'DESIGN.md', ROOT / 'recovery.json', ROOT / 'feature_contract.json', POSE,
                helper.BBOX_PATH, helper.CORE_PATH, Path(extractor.__file__),
                REPO / 'src/biu_v2_extractor.py', REPO / 'src/features/biu_features.py']
    write(ROOT / 'freeze.json', dict(sources=[record(p) for p in sources], test_accessed=False,
                                    seeds=SEEDS, threshold=.5, coverage=audits))
    print('PREPARATION GATES PASSED', flush=True)


def train():
    freeze = json.loads((ROOT / 'freeze.json').read_text())
    for r in freeze['sources']:
        assert record(r['path']) == r
    assert not (ROOT / 'EXPERIMENT_COMPLETE.json').exists()
    train_rows = pd.read_csv(ROOT / 'cache/train/rows.csv')
    val = pd.read_csv(ROOT / 'cache/val/rows.csv')
    assert not set(train_rows.video_id) & set(val.video_id)
    y = val.label.to_numpy()
    results, common = [], []
    for arm in ('A', 'B', 'C'):
        xt = np.load(ROOT / f'cache/train/{arm}.npy')
        xv = np.load(ROOT / f'cache/val/{arm}.npy')
        for seed in SEEDS:
            model = helper.bbox_core.build_xgb(seed, train_rows.label.to_numpy(), n_estimators=250)
            start = time.perf_counter()
            model.fit(xt, train_rows.label.to_numpy())
            (ROOT / 'models').mkdir(exist_ok=True)
            joblib.dump(model, ROOT / f'models/{arm}_seed{seed}.joblib')
            p = model.predict_proba(xv)[:, 1]
            out = val.copy()
            out['probability'], out['prediction'] = p.astype(float), (p >= .5).astype(int)
            write(ROOT / f'predictions/validation_{arm}_seed{seed}.csv', out)
            scores = helper.core.compute_metrics(y, p)
            scores['macro_f1'] = f1_score(y, p >= .5, average='macro')
            results.append(dict(arm=arm, seed=seed, fit_seconds=time.perf_counter() - start, **scores))
            mask = val.pose_complete.to_numpy(bool)
            if mask.any() and len(np.unique(y[mask])) == 2:
                metrics = helper.core.compute_metrics(y[mask], p[mask])
                metrics['macro_f1'] = f1_score(y[mask], p[mask] >= .5, average='macro')
                common.append(dict(arm=arm, seed=seed, **metrics))
            saved = pd.read_csv(ROOT / f'predictions/validation_{arm}_seed{seed}.csv')
            np.testing.assert_array_equal(saved.probability.to_numpy(np.float32), p)
            print(f'{arm}{seed}: F1={scores["f1"]:.6f} macro={scores["macro_f1"]:.6f} AP={scores["pr_auc"]:.6f}', flush=True)
    table = pd.DataFrame(results)
    columns = ['accuracy', 'precision', 'recall', 'f1', 'macro_f1', 'pr_auc', 'roc_auc', 'fp', 'fn']
    summary = table.groupby('arm')[columns].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    write(ROOT / 'metrics/per_seed_results.csv', table)
    write(ROOT / 'metrics/summary_results.csv', summary.reset_index())
    write(ROOT / 'metrics/complete_pose_validation.csv', pd.DataFrame(common))
    deltas = []
    for a, b in (('A', 'B'), ('A', 'C'), ('B', 'C')):
        d = table[table.arm == b].set_index('seed')[columns] - table[table.arm == a].set_index('seed')[columns]
        d['comparison'] = f'{a}->{b}'
        deltas.append(d.reset_index())
    write(ROOT / 'metrics/paired_deltas.csv', pd.concat(deltas))
    for r in freeze['sources']:
        assert record(r['path']) == r
    write(ROOT / 'EXPERIMENT_COMPLETE.json', dict(validation_only=True, test_accessed=False, fits=9,
        outputs=[record(p) for folder in ('metrics', 'predictions', 'models') for p in sorted((ROOT / folder).glob('*'))]))
    print(summary.to_string(), flush=True)


if __name__ == '__main__':
    assert sys.argv[1] in ('prepare', 'train')
    prepare() if sys.argv[1] == 'prepare' else train()
