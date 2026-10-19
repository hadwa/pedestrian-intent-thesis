"""Frozen PSI test recovery and evaluation, with no fitting."""
import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
import zipfile
import zlib
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent
REPO = EXP.parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


helper = module('psi113_test_helper', REPO / 'experiments/098_jaad_matched_biu_xgb/scripts/run_experiment.py')
extractor = module('psi113_test_biu', REPO / 'experiments/089_pedcmt_standardized_jaad_biu_transfer/scripts/exp089_data.py')
write, record = helper.write, helper.record
POSE = REPO / 'data/psi_vitpose_s_all_tracks.jsonl'
node = next(n for n in ast.parse((EXP / 'run.py').read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'extract')
ns = dict(np=np, helper=helper, extractor=extractor)
exec(compile(ast.Module(body=[node], type_ignores=[]), str(EXP / 'run.py'), 'exec'), ns)
extract = ns['extract']
SEEDS = (42, 43, 44)


def authenticate():
    assert json.loads((EXP / 'VERIFIED.json').read_text())['metrics_reproduced']
    for r in json.loads((EXP / 'freeze.json').read_text())['sources']:
        assert record(r['path']) == r
    for r in json.loads((EXP / 'EXPERIMENT_COMPLETE.json').read_text())['outputs']:
        assert record(r['path']) == r


def recover():
    authenticate()
    assert not (ROOT / 'EXPERIMENT_COMPLETE.json').exists()
    sources = [EXP / 'run.py', EXP / 'freeze.json', EXP / 'feature_contract.json', Path(__file__), ROOT / 'DESIGN.md', POSE]
    sources += [EXP / f'models/{a}_seed{s}.joblib' for a in ('A', 'B', 'C') for s in SEEDS]
    write(ROOT / 'freeze.json', dict(sources=[record(p) for p in sources], no_refit=True, threshold=.5))
    remote_mod = module('psi113_test_remote', REPO / 'experiments/psi_reference_recovery_2026_09_11/recover.py')
    remote = remote_mod.RemoteZip()
    records = []
    with zipfile.ZipFile(remote) as archive:
        member = 'PSI/PSI2.0_Test/splits/PSI2_split.json'
        payload = archive.read(member)
        assert hashlib.sha256(payload).hexdigest() == 'c9e1bce7ed3c2f0ca04be35cae7f954942c2235c69ec0e2a97eb23f213699a10'
        split = json.loads(payload)
        write(ROOT / 'official_split.json', split)
        assert len(split['test']) == 49
        old = json.loads((REPO / 'experiments/110_psi_pose_offset_correction/official_split.json').read_text())
        assert split['train'] == old['train'] and split['val'] == old['val']
        assert set(old['test']) - set(split['test']) == {'video_0192'}
        for video in sorted(split['test']):
            member = f'PSI/PSI2.0_Test/annotations/cognitive_annotation_key_frame/{video}/pedestrian_intent.json'
            info = archive.getinfo(member)
            dest = ROOT / 'dataset/PSI2.0_TrainVal/annotations/cognitive_annotation_key_frame' / video / 'pedestrian_intent.json'
            dest.parent.mkdir(parents=True, exist_ok=True)
            payload = dest.read_bytes() if dest.exists() else archive.read(member)
            assert len(payload) == info.file_size and zlib.crc32(payload) == info.CRC
            if not dest.exists():
                dest.write_bytes(payload)
            records.append(dict(member=member, **record(dest)))
            print(f'Recovered test annotation {len(records)}/49: {video}', flush=True)
    write(ROOT / 'recovery.json', dict(url=remote_mod.URL, etag=remote.etag,
          transferred_bytes=remote.transferred, sources=records, split=record(ROOT / 'official_split.json')))


def score(y, p):
    return dict(**helper.core.compute_metrics(y, p), macro_f1=f1_score(y, p >= .5, average='macro'))


def evaluate():
    authenticate()
    assert not (ROOT / 'EXPERIMENT_COMPLETE.json').exists()
    frozen = json.loads((ROOT / 'freeze.json').read_text())
    for r in frozen['sources']:
        assert record(r['path']) == r
    recovery = json.loads((ROOT / 'recovery.json').read_text())
    for r in recovery['sources']:
        actual = record(r['path'])
        assert all(r[k] == v for k, v in actual.items())
    models = {(a, s): joblib.load(EXP / f'models/{a}_seed{s}.joblib') for a in ('A', 'B', 'C') for s in SEEDS}
    val = pd.read_csv(EXP / 'cache/val/rows.csv')
    for (arm, seed), model in models.items():
        p = model.predict_proba(np.load(EXP / f'cache/val/{arm}.npy'))[:, 1]
        saved = pd.read_csv(EXP / f'predictions/validation_{arm}_seed{seed}.csv')
        np.testing.assert_array_equal(saved.sequence_id, val.sequence_id)
        np.testing.assert_array_equal(p, saved.probability.to_numpy(np.float32))
    split = json.loads((ROOT / 'official_split.json').read_text())
    assert not set(split['test']) & set(split['train'] + split['val'])
    pose = {}
    for line in POSE.open():
        r = json.loads(line)
        if r['video'] not in set(split['test'] + split['val']):
            continue
        key = (r['video'], r['pid'], int(r['frame_idx']))
        assert key not in pose
        pose[key] = (np.asarray(r['keypoints'], float), np.asarray(r['kp_conf'], float))
    # Deterministic raw validation reconstruction, checking the AST-loaded extractor.
    for i in (0, len(val) // 2, len(val) - 1):
        row = val.iloc[i]
        source = EXP / f'dataset/PSI2.0_TrainVal/annotations/cognitive_annotation_key_frame/{row.video_id}/pedestrian_intent.json'
        track = json.loads(source.read_text())['pedestrians'][row.ped_id]
        boxes = dict(zip(track['observed_frames'], track['cv_annotations']['bboxes']))
        frames = json.loads(row.observed_frames_json)
        xy = np.array([pose[(row.video_id, row.ped_id, f)][0] for f in frames])
        conf = np.array([pose[(row.video_id, row.ped_id, f)][1] for f in frames])
        vectors, _, _ = extract(xy, conf, np.array([boxes[f] for f in frames], float))
        for arm in vectors:
            np.testing.assert_array_equal(vectors[arm], np.load(EXP / f'cache/val/{arm}.npy')[i])
    print('All nine validation predictions and sampled raw feature reconstructions match.', flush=True)
    with (ROOT / 'extension.log').open('w') as log:
        subprocess.run([sys.executable, str(EXP / 'official_code/dataset/extend_intent_annotation.py'), str(ROOT / 'dataset')], check=True, stdout=log, stderr=subprocess.STDOUT)
    dbmod = module('psi113_test_db', EXP / 'official_code/baseline/database/create_database.py')
    seqmod = module('psi113_test_seq', EXP / 'official_code/baseline/data/process_sequence.py')
    args = SimpleNamespace(dataset='PSI2.0', dataset_root_path=str(ROOT / 'dataset'), max_track_size=16, intent_type='mean', intent_num=2)
    db = dbmod.init_db(split['test'], str(ROOT / 'database.log'), args)
    dbmod.update_db_annotations(db, str(ROOT / 'database.log'), args)
    rows, arrays, fallbacks = [], {a: [] for a in ('A', 'B', 'C')}, 0
    contract = json.loads((EXP / 'feature_contract.json').read_text())
    for video in sorted(db):
        for pid, track in sorted(db[video].items()):
            frames = track['frames']
            assert all(b == a + 1 for a, b in zip(frames, frames[1:]))
            original_pid = track['cv_annotations']['track_id'].split('-')[0]
            labels, votes, _, _ = seqmod.get_intent(db, video, pid, args)
            boxes = np.asarray(track['cv_annotations']['bbox'], float)
            assert np.isfinite(boxes).all() and (boxes[:, 2:] > boxes[:, :2]).all()
            for start in range(len(frames) - 15):
                obs, target = frames[start:start + 15], frames[start + 15]
                assert target == obs[-1] + 1 and target not in obs
                points, conf, present = [], [], []
                for frame in obs:
                    value = pose.get((video, original_pid, frame))
                    present.append(value is not None)
                    xy, confidence = value if value is not None else (np.zeros((17, 2)), np.zeros(17))
                    points.append(xy); conf.append(confidence)
                points, conf = np.array(points), np.array(conf)
                assert np.isfinite(points).all() and np.isfinite(conf).all()
                valid = np.any(points != 0, axis=2)
                vectors, names, fallback = extract(points, conf, boxes[start:start + 15])
                assert {a: list(n) for a, n in names.items()} == contract
                for arm in arrays:
                    arrays[arm].append(vectors[arm])
                fallbacks += int(fallback == 'bbox_center_x')
                rows.append(dict(sequence_id=f'{video}:{pid}:{obs[0]}:{target}', video_id=video,
                    ped_id=original_pid, track_id=f'{video}:{original_pid}', observed_frames_json=json.dumps(obs),
                    target_frame=target, label=labels[start + 15], annotation_vote=votes[start + 15],
                    pose_complete=bool(all(present) and valid.all()), pose_all_zero=bool(not valid.any()),
                    missing_pose_frames=15 - sum(present), valid_joint_fraction=float(valid.mean())))
            print(f'Test {video}/{pid}: {len(rows)} windows', flush=True)
    rows = pd.DataFrame(rows)
    assert rows.sequence_id.is_unique and set(rows.label) == {0, 1}
    write(ROOT / 'cache/rows.csv', rows)
    for arm in arrays:
        arrays[arm] = np.asarray(arrays[arm], float)
        assert np.isfinite(arrays[arm]).all()
        np.testing.assert_array_equal(arrays[arm][:, :53], arrays['A'])
        np.save(ROOT / f'cache/{arm}.npy', arrays[arm])
    write(ROOT / 'metrics/cohort.json', dict(windows=len(rows), positives=int(rows.label.sum()),
        videos=int(rows.video_id.nunique()), tracks=int(rows.track_id.nunique()),
        complete_pose=int(rows.pose_complete.sum()), all_zero_pose=int(rows.pose_all_zero.sum()),
        mean_valid_joint_fraction=float(rows.valid_joint_fraction.mean()), median_valid_joint_fraction=float(rows.valid_joint_fraction.median()),
        bbox_fallback_count=fallbacks, bbox_fallback_percent=100 * fallbacks / len(rows),
        annotation_vote_ties=int(rows.annotation_vote.eq(.5).sum()), validation_parity=True))
    y, table, common, probs = rows.label.to_numpy(), [], [], {}
    for (arm, seed), model in models.items():
        p = model.predict_proba(arrays[arm])[:, 1]
        probs[(arm, seed)] = p
        out = rows.copy()
        out['probability'], out['prediction'] = p.astype(float), (p >= .5).astype(int)
        path = ROOT / f'predictions/test_{arm}_seed{seed}.csv'
        write(path, out)
        np.testing.assert_array_equal(pd.read_csv(path).probability.to_numpy(np.float32), p)
        table.append(dict(arm=arm, seed=seed, **score(y, p)))
        mask = rows.pose_complete.to_numpy(bool)
        if mask.any() and len(np.unique(y[mask])) == 2:
            common.append(dict(arm=arm, seed=seed, **score(y[mask], p[mask])))
    table = pd.DataFrame(table)
    columns = ['accuracy', 'precision', 'recall', 'f1', 'macro_f1', 'pr_auc', 'roc_auc', 'fp', 'fn', 'fp_ge_0p9_count', 'fn_le_0p1_count']
    summary = table.groupby('arm')[columns].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    deltas, transitions = [], []
    for a, b in (('A', 'B'), ('A', 'C'), ('B', 'C')):
        d = table[table.arm == b].set_index('seed')[columns] - table[table.arm == a].set_index('seed')[columns]
        d['comparison'] = f'{a}->{b}'
        deltas.append(d.reset_index())
        for seed in SEEDS:
            ap, bp = probs[a, seed] >= .5, probs[b, seed] >= .5
            transitions.append(dict(comparison=f'{a}->{b}', seed=seed,
                wrong_to_correct=int(((ap != y) & (bp == y)).sum()), correct_to_wrong=int(((ap == y) & (bp != y)).sum()),
                fp_fixed=int(((y == 0) & ap & ~bp).sum()), fn_fixed=int(((y == 1) & ~ap & bp).sum()),
                fp_added=int(((y == 0) & ~ap & bp).sum()), fn_added=int(((y == 1) & ap & ~bp).sum())))
    write(ROOT / 'metrics/per_seed_results.csv', table)
    write(ROOT / 'metrics/summary_results.csv', summary.reset_index())
    write(ROOT / 'metrics/complete_pose_results.csv', pd.DataFrame(common))
    write(ROOT / 'metrics/paired_deltas.csv', pd.concat(deltas))
    write(ROOT / 'metrics/error_transitions.csv', pd.DataFrame(transitions))
    write(ROOT / 'metrics/always_cross.json', score(y, np.ones(len(y)) * .5))
    for r in frozen['sources']:
        assert record(r['path']) == r
    authenticate()
    write(ROOT / 'EXPERIMENT_COMPLETE.json', dict(no_refit=True, frozen_sources_unchanged=True,
        predictions_verified=9, outputs=[record(p) for folder in ('metrics', 'predictions') for p in sorted((ROOT / folder).glob('*'))]))
    print(summary.to_string(), flush=True)


if __name__ == '__main__':
    assert sys.argv[1] in ('recover', 'evaluate')
    recover() if sys.argv[1] == 'recover' else evaluate()
