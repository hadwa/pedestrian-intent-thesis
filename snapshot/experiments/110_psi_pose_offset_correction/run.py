"""Frozen video-disjoint, keyframe-fitted PSI offset correction diagnostic."""
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / 'src'))
from biu_extractor import extract_biu_features

NAMES = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
         'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
         'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
         'right_knee', 'left_ankle', 'right_ankle']
REQ = {'torso_lean_angle': [5, 6, 11, 12], 'stride_length': [15, 16, 5, 6],
       'head_yaw_angle': [0, 5, 6], 'lead_foot_elevation': [15, 16, 11, 12, 5, 6],
       'lead_foot_lifted': [15, 16, 11, 12, 5, 6],
       'arm_swing_asymmetry': [9, 10, 5, 6], 'gaze_to_car_angle': [0, 3, 4],
       'body_yaw_angle': [5, 6], 'face_yaw_angle': [0, 11, 12],
       'norm_shoulder_width': [5, 6, 0, 15]}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def identity(row):
    return (row['set'], row['video'], row['pid'], row['frame_idx'])

def fit_offsets(gt, est, heights, keyframes, videos):
    offsets = np.zeros((17, 2))
    support = []
    for j in range(17):
        mask = keyframes[:, j] & np.isfinite(gt[:, j]).all(1) & np.isfinite(est[:, j]).all(1)
        residual = (gt[:, j] - est[:, j]) / heights[:, None]
        medians = [np.median(residual[mask & (videos == v)], axis=0)
                   for v in np.unique(videos[mask])]
        support.append(len(medians))
        if len(medians) >= 3:
            offsets[j] = np.median(medians, axis=0)
    return offsets, support

def cues(points):
    result = []
    for row in points:
        # Missing reference joints are masked per cue after extraction.
        values = extract_biu_features(dict(zip(NAMES, np.nan_to_num(row))))
        result.append([float(values[k]) for k in REQ])
    return np.array(result)

def main():
    if (ROOT / 'EXPERIMENT_COMPLETE.json').exists():
        raise RuntimeError('Refusing to overwrite a completed experiment')
    refs = REPO / 'experiments/psi_reference_recovery_2026_09_11'
    export = REPO / 'data/psi_filtered_tracks_export_29_tracks.jsonl'
    pose = REPO / 'data/psi_vitpose_s_all_tracks.jsonl'
    verified = json.loads((refs / 'coordinate_verification.json').read_text())
    assert verified['status'] == 'PASS' and verified['export_sha256'] == sha(export)
    rows = [json.loads(s) for s in export.read_text().splitlines()]
    keys = [identity(r) for r in rows]
    assert len(set(keys)) == len(keys)
    wanted = set(keys)
    estimates = {}
    with pose.open() as stream:
        for line in stream:
            row = json.loads(line)
            key = identity(row)
            if key in wanted:
                assert key not in estimates
                estimates[key] = row['keypoints']
    assert set(estimates) == wanted
    videos = np.array([r['video'] for r in rows])
    split = json.loads((ROOT / 'official_split.json').read_text())
    assert set(videos) <= set(split['train'])
    assert not set(videos) & (set(split['val']) | set(split['test']))
    ordered = sorted(set(videos), key=lambda v: hashlib.sha256(('42:' + v).encode()).hexdigest())
    fold_map = {v: i % 3 for i, v in enumerate(ordered)}
    folds = np.array([fold_map[v] for v in videos])
    gt = np.array([r['keypoints'] for r in rows], dtype=float)
    est = np.array([estimates[k] for k in keys], dtype=float)
    heights = np.zeros(len(rows))
    flags = np.zeros((len(rows), 17), dtype=bool)
    source_hashes = {f['path']: f['sha256'] for f in verified['source_files']}
    for v in ordered:
        path = refs / 'source_annotations' / (v + '.json')
        assert sha(path) == source_hashes[str(path.relative_to(refs))]
        original = json.loads(path.read_text())
        for i in np.flatnonzero(videos == v):
            obj = original['frames'][f"frame_{rows[i]['frame_idx']}"]['cv_annotation'][f"pedestrian_{rows[i]['pid']}"]
            heights[i] = obj['bbox'][3] - obj['bbox'][1]
            flags[i] = [str(obj['skeleton'][j]['keyframe']) == '1' for j in NAMES]
    assert np.all(heights > 0)
    metrics = ROOT / 'metrics'
    metrics.mkdir(exist_ok=True)
    pd.DataFrame([{'video': v, 'official_split': 'train', 'fold': fold_map[v]} for v in ordered]).to_csv(metrics / 'folds.csv', index=False)
    freeze = {'design_sha256': sha(ROOT / 'DESIGN.md'), 'runner_sha256': sha(Path(__file__)),
              'pose_sha256': sha(pose), 'export_sha256': sha(export),
              'split_sha256': sha(ROOT / 'official_split.json'),
              'extractor_sha256': sha(REPO / 'src/biu_extractor.py'),
              'pose_utils_sha256': sha(REPO / 'src/pose_utils.py')}
    (ROOT / 'freeze.json').write_text(json.dumps(freeze, indent=2) + '\n')
    gt_cues = cues(gt)
    joint_rows, cue_rows, fitted = [], [], []
    for fold in range(3):
        train, held = folds != fold, folds == fold
        assert not set(videos[train]) & set(videos[held])
        offset, support = fit_offsets(gt[train], est[train], heights[train], flags[train], videos[train])
        fitted.append({'fold': fold, 'offsets': offset.tolist(), 'training_video_support': support})
        corrected = est + heights[:, None, None] * offset[None]
        pred_cues = {'original': cues(est[held]), 'corrected': cues(corrected[held])}
        for scope in ('keyframes', 'all_annotated'):
            valid = np.isfinite(gt).all(2) & np.isfinite(est).all(2)
            if scope == 'keyframes':
                valid &= flags
            for variant, points in [('original', est), ('corrected', corrected)]:
                error = np.linalg.norm(points - gt, axis=2) / heights[:, None]
                for v in np.unique(videos[held]):
                    for family, js in [('head', list(range(5))), ('arms', list(range(5, 11))), ('lower', list(range(11, 17)))]:
                        mask = valid[videos == v][:, js]
                        values = error[videos == v][:, js][mask]
                        if len(values):
                            joint_rows.append(dict(fold=fold, video=v, scope=scope, variant=variant,
                                                   family=family, n=len(values), error=values.mean(), pck=float((values <= .1).mean())))
            for c, (name, js) in enumerate(REQ.items()):
                mask = valid[:, js].all(1) & np.isfinite(gt_cues[:, c])
                # Cue error units use only fitting-video references, never held-out values.
                training_values = gt_cues[train & mask, c]
                if not len(training_values):
                    continue
                scale = max(float(np.diff(np.percentile(training_values, [25, 75]))[0]), 1e-6)
                if name == 'lead_foot_lifted':
                    scale = 1.0
                for variant, predictions in pred_cues.items():
                    delta = predictions[:, c] - gt_cues[held, c]
                    if 'angle' in name:
                        delta = np.arctan2(np.sin(delta), np.cos(delta))
                    for v in np.unique(videos[held]):
                        take = mask[held] & (videos[held] == v) & np.isfinite(delta)
                        if take.any():
                            cue_rows.append(dict(fold=fold, video=v, scope=scope, variant=variant, cue=name,
                                                 n=int(take.sum()), mae=float(np.abs(delta[take]).mean()),
                                                 error=float(np.abs(delta[take]).mean()/scale), training_iqr=scale))
        print(f'Fold {fold} completed; training videos={len(set(videos[train]))}, held-out={len(set(videos[held]))}', flush=True)
    (metrics / 'fitted_offsets.json').write_text(json.dumps(fitted, indent=2) + '\n')
    joint = pd.DataFrame(joint_rows)
    cue = pd.DataFrame(cue_rows)
    joint.to_csv(metrics / 'joint_errors.csv', index=False)
    cue.to_csv(metrics / 'cue_errors.csv', index=False)
    summaries = []
    for name, table in [('joint', joint), ('cue', cue)]:
        macro = table.groupby(['scope', 'variant', 'fold', 'video']).error.mean().reset_index()
        for (scope, variant), group in macro.groupby(['scope', 'variant']):
            summaries.append(dict(metric=name, scope=scope, variant=variant, error=group.error.mean(), videos=group.video.nunique()))
        macro.to_csv(metrics / f'{name}_video_macro.csv', index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(metrics / 'summary.csv', index=False)
    gate = {}
    for name, table in [('joint', joint), ('cue', cue)]:
        primary = table[table.scope == 'keyframes'].groupby(['variant', 'fold', 'video']).error.mean().reset_index()
        paired = primary.pivot(index=['fold', 'video'], columns='variant', values='error').dropna()
        fold_gain = (paired.original - paired.corrected).groupby(level=0).mean()
        relative = float((paired.original.mean() - paired.corrected.mean()) / paired.original.mean())
        counts = paired.groupby(level=0).size().to_dict()
        gate[name] = {'relative_improvement': relative, 'fold_error_reductions': fold_gain.to_dict(),
                      'fold_video_support': counts,
                      'pass': bool(relative >= .02 and len(fold_gain) == 3 and (fold_gain > 0).all() and min(counts.values()) >= 3)}
    result = {'status': 'COMPLETE', 'advance_to_jaad': all(g['pass'] for g in gate.values()),
              'gate': gate, 'scope': 'Training-only PSI video-disjoint OOF measurement diagnostic'}
    (ROOT / 'EXPERIMENT_COMPLETE.json').write_text(json.dumps(result, indent=2) + '\n')
    print(summary.to_string(index=False))
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
