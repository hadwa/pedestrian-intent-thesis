"""Complete fixed capped controls using the original residual fitting routine."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
SEEDS = (42, 43, 44)
ARMS = {'pie': ('raw', 'core56', 'full63'), 'jaad': ('core56', 'full63')}
SOURCES = {}
TEST = False


def exp(n):
    paths = list((REPO / 'experiments').glob(f'{n:03d}_*'))
    assert len(paths) == 1
    return paths[0]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def source(p):
    p = Path(p).resolve(strict=True)
    if not TEST and any('test' in s.lower() or 'set03' in s.lower() for s in p.parts):
        raise PermissionError(f'Test gate closed: {p}')
    if str(p) not in SOURCES:
        SOURCES[str(p)] = sha(p)
    return p


def loadmod(p, name):
    source(p)
    sys.path.insert(0, str(p.parent))
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def dump(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    os.replace(tmp, p)


def arr(p):
    return np.load(source(p), allow_pickle=False)


def frame(p):
    return pd.read_csv(source(p))


P93 = loadmod(exp(93) / 'scripts/run_experiment.py', 'completion_p93')
BASE = P93.BASE90
ORIGINAL_MODEL = BASE.DirectFusionModel
DEVICE = torch.device('cuda')


def model(arm):
    if arm == 'raw':
        return ORIGINAL_MODEL('B')
    d = {'core56': 56, 'full63': 63}[arm]
    # Preserve original initialization order while substituting only BIU input width.
    module = sys.modules[ORIGINAL_MODEL.__module__]
    old = module.BiuUnionEncoder
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Dropout(.3),
                                         nn.Linear(128, 128), nn.LayerNorm(128))
        def forward(self, x):
            assert x.ndim == 2 and x.shape[1] == d
            return self.network(x)
    module.BiuUnionEncoder = Encoder
    try:
        return ORIGINAL_MODEL('C')
    finally:
        module.BiuUnionEncoder = old


def load_data(case, phase):
    parent = exp(84 if case == 'pie' else 89)
    folder = (exp(50) / 'cache' / ('trainval_base' if phase == 'trainval' else 'set03_final')) if case == 'pie' else parent / 'cache' / (phase + '_source')
    rows = frame(folder / 'rows.csv')
    expected = ({'train': 2366, 'val': 3619} if case == 'pie' else {'train': 1552, 'val': 274}) if phase == 'trainval' else {'test': 3760 if case == 'pie' else 1501}
    assert rows.split.value_counts().to_dict() == expected
    assert not rows.sequence_id.duplicated().any()
    assert (rows.end_frame - rows.start_frame).eq(15).all()
    tte = 'time_to_event_frames' if 'time_to_event_frames' in rows else 'tte_frames'
    assert rows[tte].between(30, 60).all()
    if phase == 'trainval':
        tr = rows.split.eq('train')
        key = 'video_id'
        if case == 'pie':
            ids = rows.set_id.astype(str) + '/' + rows[key].astype(str)
        else:
            ids = rows[key].astype(str)
        assert not set(ids[tr]) & set(ids[~tr])
    meta = json.loads(source(folder / 'metadata.json').read_text())
    names = meta['biu_v2_feature_names']
    defs = frame(exp(12) / 'metrics/pie_biu_v2_feature_set_definitions.csv').set_index('feature_set')
    core = json.loads(defs.loc['biu_v2_core', 'feature_columns_json'])
    ix = [names.index(n) for n in core]
    assert len(ix) == len(set(ix)) == 56 and len(names) == 63
    stats = arr(parent / 'cache/trainval/training_stats.npz')
    values = {}
    for rawname, statname in [('pose', 'pose'), ('biu_v2', 'v2')]:
        raw = arr(folder / f'{rawname}.npy').astype(np.float32)
        mean, std = stats[f'{statname}_mean'], stats[f'{statname}_std']
        if phase == 'trainval':
            axes = (0, 1) if raw.ndim == 3 else 0
            calculated_mean = raw[tr].mean(axis=axes, dtype=np.float64).astype(np.float32)
            calculated_std = raw[tr].std(axis=axes, dtype=np.float64).astype(np.float32)
            calculated_std = np.where(calculated_std < 1e-6, 1., calculated_std)
            np.testing.assert_allclose(mean, calculated_mean, atol=1e-6, rtol=0)
            np.testing.assert_allclose(std, calculated_std, atol=1e-6, rtol=0)
        standardized = ((raw - mean) / std).astype(np.float32)
        if phase == 'trainval':
            np.testing.assert_array_equal(standardized, arr(parent / f'cache/trainval/{rawname}_std.npy'))
        values['raw' if rawname == 'pose' else 'full63'] = standardized
    values['core56'] = values['full63'][:, ix]
    bbox = arr(folder / 'bbox.npy').astype(np.float32)
    if case == 'pie':
        bbox = P93.EXP084_DATA._center_bbox_to_xyxy(bbox)
    values['bbox'] = bbox
    values['ego'] = arr(folder / 'ego_speed.npy').astype(np.float32)
    assert values['raw'].shape == (len(rows), 16, 34)
    assert all(len(v) == len(rows) and np.isfinite(v).all() for v in values.values())
    dump(ROOT / case / 'metrics/feature_contract.json', {'core56': core, 'full63': names})
    return rows, values


def parent_outputs(case, seed, phase, rows, values):
    parent = exp(84 if case == 'pie' else 89)
    checkpoint = source(parent / f'checkpoints/A_seed{seed}.best.pt')
    m = P93.load_parent(seed, 1, DEVICE) if case == 'pie' else BASE.load_parent_model(seed, DEVICE)
    assert not any(p.requires_grad for p in m.parameters())
    logits, anchor = P93.extract_parent_outputs(m, values['bbox'], values['ego'], DEVICE)
    split = 'validation' if phase == 'trainval' else 'test'
    saved = frame(parent / f'predictions/{split}_A_seed{seed}.csv')
    idx = np.flatnonzero(rows.split.eq('val')) if phase == 'trainval' else np.arange(len(rows))
    assert rows.iloc[idx].sequence_id.astype(str).tolist() == saved.sequence_id.astype(str).tolist()
    np.testing.assert_array_equal(rows.iloc[idx].label, saved.label)
    np.testing.assert_allclose(P93.sigmoid(logits[idx].ravel()), saved.probability, atol=2e-6, rtol=0)
    count = sum(p.numel() for p in m.parameters())
    del m
    if phase == 'trainval':
        cache = exp(93 if case == 'pie' else 90) / 'cache/trainval'
        cached_logits = arr(cache / f'parent_seed{seed}_logits.npy')
        cached_anchor = arr(cache / f'parent_seed{seed}_anchor.npy')
        np.testing.assert_allclose(P93.sigmoid(logits), P93.sigmoid(cached_logits), atol=2e-6, rtol=0)
        np.testing.assert_allclose(anchor, cached_anchor, atol=1e-4, rtol=0)
        dump(ROOT / case / f'metrics/parent_replay_seed{seed}.json', {
            'max_abs_logit': float(np.max(np.abs(logits-cached_logits))),
            'max_abs_anchor': float(np.max(np.abs(anchor-cached_anchor))),
            'max_abs_probability': float(np.max(np.abs(P93.sigmoid(logits)-P93.sigmoid(cached_logits)))),
            'training_uses_original_cached_outputs': True})
        # Train on byte-identical historical parent outputs; replay is a numerical check.
        logits, anchor = cached_logits, cached_anchor
    return logits, anchor, sha(checkpoint), count


def pack(rows, values, logits, anchor, arm, idx):
    return {'parent_logits': logits[idx], 'parent_anchor': anchor[idx],
            'extra': values[arm][idx], 'labels': rows.iloc[idx].label.to_numpy(np.float32)}


def predict(case, arm, seed, phase, rows, parent_logits, correction):
    out = rows.copy()
    out['parent_probability'] = P93.sigmoid(parent_logits.ravel())
    out['full_correction_logit'] = correction.ravel()
    out['probability'] = P93.sigmoid(parent_logits.ravel() + .10 * correction.ravel())
    out['uncapped_probability'] = P93.sigmoid(parent_logits.ravel() + correction.ravel())
    out['prediction'] = (out.probability >= .5).astype(int)
    assert np.max(np.abs(correction)) <= 2.000001
    p = ROOT / case / f'predictions/{phase}_{arm}_seed{seed}.csv'
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)


def main():
    global TEST
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if (ROOT / 'EXPERIMENT_COMPLETE.json').exists():
        raise RuntimeError('Completed experiment is immutable; refusing rerun')
    lock = (ROOT / '.execution.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert torch.cuda.is_available(), 'GPU required; no silent CPU fallback'
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for p in [ROOT / 'DESIGN.md', Path(__file__)]:
        source(p)
    for n in (84, 89, 90, 93, 94, 117):
        for p in (exp(n) / 'scripts').glob('*.py'):
            source(p)
    frozen = ROOT / 'FROZEN.json'
    specs = {str(p): sha(p) for p in [ROOT / 'DESIGN.md', Path(__file__)]}
    if frozen.exists():
        assert json.loads(frozen.read_text()) == specs, 'Frozen code/design drift'
    contexts = {}
    for case in ARMS:
        rows, values = load_data(case, 'trainval')
        contexts[case] = (rows, values, {})
        for seed in SEEDS:
            contexts[case][2][seed] = parent_outputs(case, seed, 'trainval', rows, values)
        for arm in ARMS[case]:
            BASE.seed_all(42)
            m = model(arm).to(DEVICE)
            logits, anchor, _, _ = contexts[case][2][42]
            idx = np.arange(4)
            t = pack(rows, values, logits, anchor, arm, idx)
            output, correction = m(*[torch.tensor(t[k], device=DEVICE) for k in ('parent_logits', 'parent_anchor', 'extra')])
            assert torch.count_nonzero(correction) == 0
            loss = torch.nn.functional.binary_cross_entropy_with_logits(output, torch.tensor(t['labels'][:, None], device=DEVICE))
            loss.backward()
            assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
            del m
    dump(ROOT / 'metrics/pretraining_sources.json', SOURCES)
    dump(ROOT / 'PRETRAINING_AUDIT_PASSED.json', {'gpu': torch.cuda.get_device_name(0), 'test_accessed': False, 'cases': list(ARMS), 'smoke_backward': True})
    if args.audit_only:
        print('AUDIT PASSED; no training or test access', flush=True)
        return
    if not frozen.exists():
        dump(frozen, specs)
    for case, (rows, values, parents) in contexts.items():
        tr = np.flatnonzero(rows.split.eq('train'))
        va = np.flatnonzero(rows.split.eq('val'))
        for arm in ARMS[case]:
            for seed in SEEDS:
                stem = f'{arm}_seed{seed}'
                ck = ROOT / case / f'checkpoints/{stem}.best.pt'
                marker = ROOT / case / f'logs/{stem}.complete.json'
                if marker.exists():
                    prior = json.loads(marker.read_text())
                    assert sha(ck) == prior['checkpoint_sha256']
                    assert sha(ROOT / case / f'predictions/validation_{stem}.csv') == prior['validation_sha256']
                    continue
                print(f'START {case} {stem}', flush=True)
                logits, anchor, parent_sha, parent_count = parents[seed]
                train_idx = tr[P93.BASE84.balanced_indices(rows.iloc[tr].label.to_numpy(), seed)] if case == 'pie' else tr
                BASE.DirectFusionModel = lambda variant, a=arm: model(a)
                result = BASE.fit_branch('B' if arm == 'raw' else 'C',
                    pack(rows, values, logits, anchor, arm, train_idx),
                    pack(rows, values, logits, anchor, arm, va), seed=seed, device=DEVICE,
                    checkpoint_path=ck, history_path=ROOT / case / f'logs/{stem}.history.csv',
                    parent_sha256=parent_sha)
                correction = result.pop('correction')
                result.pop('probability')
                predict(case, arm, seed, 'validation', rows.iloc[va], logits[va], correction)
                result.update(checkpoint_sha256=sha(ck), validation_sha256=sha(ROOT / case / f'predictions/validation_{stem}.csv'),
                              parent_parameters=parent_count, total_parameters=parent_count + result['trainable_parameters'])
                dump(marker, result)
                print(f'COMPLETE {case} {stem} best_epoch={result["best_epoch"]}', flush=True)
    for path, digest in SOURCES.items():
        assert sha(path) == digest, f'Source mutated: {path}'
    ckhash = {str(p): sha(p) for p in ROOT.glob('*/checkpoints/*.best.pt')}
    assert len(ckhash) == 15
    dump(ROOT / 'TRAINING_COMPLETE.json', {'checkpoints': ckhash, 'test_accessed': False})
    # Only now are any held-out arrays or saved test probabilities opened.
    TEST = True
    allmetrics, transitions = [], []
    for case in ARMS:
        rows, values = load_data(case, 'test')
        devrows = contexts[case][0]
        assert not set(rows.sequence_id) & set(devrows.sequence_id)
        ids = lambda r: set((r.set_id.astype(str) + '/' if case == 'pie' else '') + r.video_id.astype(str))
        assert not ids(rows) & ids(devrows)
        for seed in SEEDS:
            logits, anchor, parent_sha, _ = parent_outputs(case, seed, 'test', rows, values)
            probs = {'baseline': P93.sigmoid(logits.ravel())}
            if case == 'jaad':
                raw = frame(exp(94) / f'predictions/test_P10_seed{seed}.csv')
                assert raw.sequence_id.astype(str).tolist() == rows.sequence_id.astype(str).tolist()
                np.testing.assert_array_equal(raw.label, rows.label)
                probs['raw'] = raw.probability.to_numpy()
            for arm in ARMS[case]:
                ck = ROOT / case / f'checkpoints/{arm}_seed{seed}.best.pt'
                assert sha(ck) == ckhash[str(ck)]
                state = torch.load(ck, map_location=DEVICE, weights_only=False)
                assert state['parent_sha256'] == parent_sha
                m = model(arm).to(DEVICE)
                m.load_state_dict(state['model_state'], strict=True)
                _, _, correction = BASE.evaluate_branch(m, pack(rows, values, logits, anchor, arm, np.arange(len(rows))), DEVICE)
                predict(case, arm, seed, 'test', rows, logits, correction)
                probs[arm] = P93.sigmoid(logits.ravel() + .1 * correction.ravel())
                allmetrics.append(dict(case=case, arm=arm, seed=seed, split='test_uncapped_diagnostic', **BASE.fixed_metrics(rows.label, P93.sigmoid(logits.ravel()+correction.ravel()))))
            for arm, p in probs.items():
                allmetrics.append(dict(case=case, arm=arm, seed=seed, split='test', **BASE.fixed_metrics(rows.label, p)))
            y = rows.label.to_numpy()
            for ref in ('baseline', 'raw'):
                for arm in ('core56', 'full63'):
                    a, b = probs[ref] >= .5, probs[arm] >= .5
                    transitions.append(dict(case=case, seed=seed, reference=ref, arm=arm,
                        wrong_to_correct=int(((a!=y)&(b==y)).sum()), correct_to_wrong=int(((a==y)&(b!=y)).sum()),
                        fp_corrected=int(((y==0)&a&~b).sum()), fn_corrected=int(((y==1)&~a&b).sum())))
    for case, arms in ARMS.items():
        for arm in arms:
            for seed in SEEDS:
                p = pd.read_csv(ROOT / case / f'predictions/validation_{arm}_seed{seed}.csv')
                allmetrics.append(dict(case=case, arm=arm, seed=seed, split='validation', **BASE.fixed_metrics(p.label, p.probability)))
    out = pd.DataFrame(allmetrics)
    out.to_csv(ROOT / 'metrics/per_seed_results.csv', index=False)
    summary = out.groupby(['case','split','arm']).agg({c:['mean','std'] for c in out.select_dtypes('number').columns if c != 'seed'})
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(ROOT / 'metrics/summary_results.csv', index=False)
    pd.DataFrame(transitions).to_csv(ROOT / 'metrics/error_transitions.csv', index=False)
    deltas = []
    for case in ARMS:
        for seed in SEEDS:
            sub = out.query('case == @case and seed == @seed and split == "test"').set_index('arm')
            for ref in ('baseline', 'raw', 'core56'):
                for arm in ('core56', 'full63'):
                    if ref == arm: continue
                    deltas.append(dict(case=case, seed=seed, reference=ref, arm=arm,
                        **{metric:float(sub.loc[arm,metric]-sub.loc[ref,metric]) for metric in ('f1','pr_auc','roc_auc','accuracy')}))
    pd.DataFrame(deltas).to_csv(ROOT / 'metrics/paired_deltas.csv', index=False)
    for path, digest in SOURCES.items(): assert sha(path) == digest
    dump(ROOT / 'metrics/all_sources.json', SOURCES)
    dump(ROOT / 'EXPERIMENT_COMPLETE.json', {'new_trainings':15, 'seeds':SEEDS, 'cases':list(ARMS), 'time':time.time()})
    print('EXPERIMENT COMPLETE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        dump(ROOT / 'FAILURE.json', {'error':repr(e), 'time':time.time(), 'test_accessed':TEST})
        raise
