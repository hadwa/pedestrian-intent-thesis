"""Reuse exact parent training routines; change only full63 to core56."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
SEEDS = (42, 43, 44)
CASES = ('mslstm_pie', 'pedcmt_pie', 'pedcmt_jaad')
TEST = False
SOURCES = {}
AUDIT = []
DEVICE = torch.device('cuda')


def exp(n):
    paths = list((REPO / 'experiments').glob(f'{n:03d}_*'))
    assert len(paths) == 1
    return paths[0]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def src(path):
    p = Path(path).resolve(strict=True)
    if not TEST and any('test' in s.lower() or 'set03' in s.lower() for s in p.parts):
        raise PermissionError(f'final test gate closed: {p}')
    if str(p) not in SOURCES:
        SOURCES[str(p)] = dict(sha256=sha(p), bytes=p.stat().st_size)
    return p


def js(path):
    return json.loads(src(path).read_text())


def csv(path):
    return pd.read_csv(src(path))


def arr(path):
    return np.load(src(path), allow_pickle=False)


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def loadmod(path, name):
    path = src(path)
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def log(msg):
    line = time.strftime('%Y-%m-%d %H:%M:%S') + ' ' + msg
    print(line, flush=True)
    with (ROOT / 'logs/progress.log').open('a') as f:
        f.write(line + '\n')


def index56(names):
    d = csv(exp(12) / 'metrics/pie_biu_v2_feature_set_definitions.csv').set_index('feature_set')
    names56 = json.loads(d.loc['biu_v2_core', 'feature_columns_json'])
    assert len(names56) == len(set(names56)) == 56 and len(names) == 63
    return [list(names).index(n) for n in names56], names56


def checkrows(rows, case, phase):
    assert not rows.sequence_id.duplicated().any() and set(rows.label.unique()) == {0, 1}
    expected = {'train': 1552, 'val': 274} if case == 'pedcmt_jaad' else {'train': 2366, 'val': 3619}
    if phase == 'test':
        expected = {'test': 1501 if case == 'pedcmt_jaad' else 3760}
    assert rows.split.value_counts().to_dict() == expected
    tte = 'tte_frames' if 'tte_frames' in rows else 'time_to_event_frames'
    assert rows[tte].between(30, 60).all()
    assert (rows.end_frame - rows.start_frame).eq(15).all()
    if 'observed_frames_json' in rows:
        fr = np.array(rows.observed_frames_json.map(json.loads).tolist())
        assert fr.shape == (len(rows), 16) and (np.diff(fr, axis=1) == 1).all()
        np.testing.assert_array_equal(fr[:, 0], rows.start_frame)
        np.testing.assert_array_equal(fr[:, -1], rows.end_frame)
    if case != 'pedcmt_jaad':
        sets = {'train': {'set01', 'set02', 'set06'}, 'val': {'set04', 'set05'}, 'test': {'set03'}}
        for s, g in rows.groupby('split'):
            assert set(g.set_id) == sets[s]


def align(rows, pred):
    assert not pred.sequence_id.duplicated().any() and set(pred.sequence_id) == set(rows.sequence_id)
    p = pred.set_index('sequence_id').loc[rows.sequence_id].reset_index()
    np.testing.assert_array_equal(rows.label, p.label)
    return p


class Ped:
    def __init__(self, case):
        self.case, self.root = case, ROOT / case
        self.parent = exp(84 if case == 'pedcmt_pie' else 89)
        w = loadmod(self.parent / 'scripts/run_experiment.py', f'exp117_{case}_runner')
        self.runner = w if case == 'pedcmt_pie' else w.core
        self.modelmod = sys.modules[self.runner.PedCMTTransfer.__module__]
        self.original_projector = self.modelmod.VectorProjector
        self.parent_data = self.runner.load_population
        self.derived = self.parent / 'cache/trainval'
        self.newderived = self.root / 'cache/trainval'
        self.indices = None
        self.population = None

    def population_for(self, phase):
        source = (exp(50) / 'cache' / ('set03_final' if phase == 'test' else 'trainval_base')) if self.case == 'pedcmt_pie' else self.parent / 'cache' / ('test_source' if phase == 'test' else 'trainval_source')
        # Prove caches exist before calling a legacy adapter that can otherwise rebuild.
        for name in ('metadata.json', 'rows.csv', 'bbox.npy', 'ego_speed.npy', 'pose.npy', 'biu_v1.npy', 'biu_v2.npy'):
            src(source / name)
        pop = self.parent_data(phase, allow_test=TEST)
        checkrows(pop.rows, self.case, phase)
        return pop

    def model(self, core):
        original = self.original_projector
        self.modelmod.VectorProjector = (lambda input_dim, model_dim: original(56 if input_dim == 63 else input_dim, model_dim)) if core else original
        return self.runner.PedCMTTransfer(variant='D', ego_dim=1)

    def preparation(self):
        self.population = self.population_for('trainval')
        p = self.population
        self.indices, names = index56(p.biu_v2_names)
        train = p.rows.split.eq('train').to_numpy()
        old = arr(self.derived / 'biu_v2_std.npy')
        stats = arr(self.derived / 'training_stats.npz')
        mean = p.biu_v2[train].mean(axis=0, dtype=np.float64).astype(np.float32)
        std = p.biu_v2[train].std(axis=0, dtype=np.float64).astype(np.float32)
        std = np.where(std < 1e-6, 1., std).astype(np.float32)
        np.testing.assert_array_equal(mean, stats['v2_mean'])
        np.testing.assert_array_equal(std, stats['v2_std'])
        np.testing.assert_array_equal((p.biu_v2 - mean) / std, old)
        self.mean, self.std = mean, std
        self.newderived.mkdir(parents=True, exist_ok=True)
        np.save(self.newderived / 'biu_v2_std.npy', old[:, self.indices])
        endpoint = arr(self.derived / 'endpoint.npy')
        assert endpoint.shape == (len(p.rows), 4) and np.isfinite(endpoint).all()
        assert (p.rows.event_frame > p.rows.end_frame).all()
        np.save(self.newderived / 'endpoint.npy', endpoint)
        np.savez(self.newderived / 'training_stats.npz', mean=mean[self.indices], std=std[self.indices])
        dump(self.root / 'metrics/feature_contract.json', dict(names=names, original_indices=self.indices, standardization='training only'))
        r = self.runner
        for key, val in dict(EXP_DIR=self.root, CACHE_DIR=self.root / 'cache', TRAINVAL_DERIVED=self.newderived,
                             CHECKPOINT_DIR=self.root / 'checkpoints', PREDICTION_DIR=self.root / 'predictions',
                             METRICS_DIR=self.root / 'metrics', LOG_DIR=self.root / 'logs', VARIANTS=('D',)).items():
            setattr(r, key, val)
        r.load_population = lambda phase, allow_test: self.population_for(phase)
        dump(self.root / 'PREPARATION_COMPLETE.json', dict(parent=str(self.parent), core_dimension=56))
        return p.rows

    def predict(self, model, phase, core):
        pop = self.population if phase == 'validation' else self.population_for('test')
        positions = np.flatnonzero(pop.rows.split.eq('val').to_numpy()) if phase == 'validation' else np.arange(len(pop.rows))
        extra = ((pop.biu_v2 - self.mean) / self.std).astype(np.float32)
        if core:
            extra = extra[:, self.indices]
        data = self.runner.ArrayDataset(pop.bbox[positions], pop.ego[positions], pop.rows.iloc[positions].label.to_numpy(),
                                        np.zeros((len(positions), 4), np.float32), extra[positions])
        loader = DataLoader(data, batch_size=256, shuffle=False, num_workers=0)
        model.to(DEVICE)
        _, y, p = self.runner._evaluate_loader(model, loader, 'D', DEVICE)
        rows = pop.rows.iloc[positions].reset_index(drop=True)
        np.testing.assert_array_equal(y.astype(int), rows.label)
        return rows, p

    def parent_checkpoint(self, seed):
        return self.parent / f'checkpoints/D_seed{seed}.best.pt'

    def new_checkpoint(self, seed):
        return self.root / f'checkpoints/D_seed{seed}.best.pt'

    def parent_predictions(self, seed, phase, variant='D'):
        return self.parent / f'predictions/{phase}_{variant}_seed{seed}.csv'

    def train(self, seed):
        self.modelmod.VectorProjector = lambda input_dim, model_dim: self.original_projector(56 if input_dim == 63 else input_dim, model_dim)
        marker = self.root / f'logs/D_seed{seed}.complete.json'
        if marker.exists():
            assert self.new_checkpoint(seed).exists()
            return
        self.runner.train_one('D', seed, device_name='cuda')

    def smoke(self):
        p = self.population
        tr = np.flatnonzero(p.rows.split.eq('train'))[:4]
        model = self.model(True).to(DEVICE)
        args = [torch.tensor(v, device=DEVICE) for v in (p.bbox[tr], p.ego[tr], ((p.biu_v2[tr] - self.mean) / self.std)[:, self.indices])]
        logits, endpoints, lv1, lv2 = model(args[0], args[1], biu=args[2])
        targets = torch.tensor(arr(self.newderived / 'endpoint.npy')[tr], device=DEVICE)
        loss = self.runner.multitask_loss(logits, endpoints, torch.tensor(p.rows.iloc[tr].label.to_numpy(np.float32)[:, None], device=DEVICE), targets, lv1, lv2)[0]
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(q.grad).all() for q in model.parameters() if q.grad is not None)


class Mslstm:
    case = 'mslstm_pie'

    def __init__(self):
        self.root = ROOT / self.case
        self.parent = exp(53)
        self.wrapper = loadmod(self.parent / 'scripts/run_experiment.py', 'exp117_mslstm_parent')
        self.runner, self.core = self.wrapper.base, self.wrapper.core
        self.original_load = self.wrapper._original_load_cache
        self.basecache = exp(50) / 'cache/trainval_base'
        self.testcache = exp(50) / 'cache/set03_final'
        self.streams = self.core.VARIANT_STREAMS['I']

    def read_cache(self, path, core=True):
        path = Path(path)
        if path == self.basecache:
            origin, kind = path, 'trainval'
        elif path == self.testcache:
            assert TEST
            origin, kind = path, 'test'
        else:
            kind = path.name
            assert kind.startswith('augmented_seed')
            origin = exp(50) / 'cache' / kind
        # Read-only parent loader consumes these exact arrays; no preparation is invoked.
        for name in ('rows.csv', 'metadata.json', *[f'{k}.npy' for k in self.runner.ARRAY_NAMES]):
            src(origin / name)
        rows, arrays, meta = self.original_load(origin)
        arrays = dict(arrays)
        arrays['bbox53'] = arr(self.parent / f'cache/bbox53_{kind}.npy')
        if kind.startswith('augmented_seed'):
            assert meta['source_base_metadata_sha256'] == sha(self.basecache / 'metadata.json')
            feature_names = js(self.basecache / 'metadata.json')['biu_v2_feature_names']
        else:
            feature_names = meta['biu_v2_feature_names']
        idx, names = index56(feature_names)
        assert idx == self.indices and arrays['bbox53'].shape == (len(rows), 53)
        if core:
            arrays['biu_v2'] = np.asarray(arrays['biu_v2'][:, idx], np.float32)
        return rows, arrays, meta

    def model(self, core):
        self.core.STREAM_DIMS['biu_v2'] = 56 if core else 63
        m = self.core.MslstmPracticalModel('I')
        self.core.STREAM_DIMS['biu_v2'] = 56
        return m

    def preparation(self):
        metadata = js(self.basecache / 'metadata.json')
        self.indices, names = index56(metadata['biu_v2_feature_names'])
        self.rows, self.arrays, _ = self.read_cache(self.basecache)
        checkrows(self.rows, self.case, 'trainval')
        dump(self.root / 'metrics/feature_contract.json', dict(names=names, original_indices=self.indices, standardization='unchanged parent raw BIU'))
        r = self.runner
        r.ROOT = self.root
        r.PREPARE_MARKER = self.root / '.prepare.complete.json'
        r.BASE_CACHE, r.TEST_CACHE = self.basecache, self.testcache
        r.load_cache = self.read_cache
        self.core.VARIANTS = ('I',)
        self.core.STREAM_DIMS['biu_v2'] = 56
        for seed in SEEDS:
            ar, arrays, _ = self.read_cache(self.root / f'cache/augmented_seed{seed}')
            assert set(ar.source_cache_index) <= set(np.flatnonzero(self.rows.split.eq('train')))
            np.testing.assert_array_equal(ar.label, self.rows.iloc[ar.source_cache_index].label)
            source_original = arr(exp(50) / f'cache/augmented_seed{seed}/biu_v2.npy')
            np.testing.assert_array_equal(arrays['biu_v2'], source_original[:, self.indices])
            assert np.isfinite(arrays['biu_v2']).all()
        feature = self.root / 'metrics/feature_contract.json'
        dump(r.PREPARE_MARKER, dict(test_accessed=False, artifacts={'metrics/feature_contract.json': sha(feature)}))
        return self.rows

    def predict(self, model, phase, core):
        rows, arrays, _ = self.read_cache(self.basecache if phase == 'validation' else self.testcache, core=core)
        pos = np.flatnonzero(rows.split.eq('val')) if phase == 'validation' else np.arange(len(rows))
        if phase == 'test':
            checkrows(rows, self.case, 'test')
        tensors = {k: self.runner.tensor_for_stream(arrays[k][pos], k, DEVICE) for k in self.streams}
        model.to(DEVICE)
        p = self.runner.probabilities(model, tensors, len(pos))
        return rows.iloc[pos].reset_index(drop=True), p

    def parent_checkpoint(self, seed):
        return self.parent / f'checkpoints/I_seed{seed}.best.pt'

    def new_checkpoint(self, seed):
        return self.root / f'checkpoints/I_seed{seed}.best.pt'

    def parent_predictions(self, seed, phase, variant='I'):
        parent = exp(95) if variant in ('J', 'K', 'L') else self.parent
        return parent / f'predictions/{"set03" if phase == "test" else phase}_{variant}_seed{seed}.csv'

    def train(self, seed):
        self.runner.run_training('I', seed)

    def smoke(self):
        pos = np.flatnonzero(self.rows.split.eq('train'))[:4]
        m = self.model(True).to(DEVICE)
        data = {k: self.runner.tensor_for_stream(self.arrays[k][pos], k, DEVICE) for k in self.streams}
        loss = torch.nn.functional.cross_entropy(m(data), torch.tensor(self.rows.iloc[pos].label.to_numpy(), device=DEVICE))
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(q.grad).all() for q in m.parameters() if q.grad is not None)


def immutable():
    for p, info in SOURCES.items():
        assert sha(p) == info['sha256'], p


def disjoint(rows, case):
    groups = list(rows.groupby('split'))
    for i, (a, x) in enumerate(groups):
        for b, y in groups[i + 1:]:
            assert not set(x.sequence_id) & set(y.sequence_id)
            for col in ('video_id', 'ped_id'):
                if col not in x:
                    continue
                keys = (['set_id'] if case != 'pedcmt_jaad' else []) + ['video_id']
                if col != 'video_id':
                    keys += [col]
                assert not set(x[keys].itertuples(index=False, name=None)) & set(y[keys].itertuples(index=False, name=None))


def evaluate_parent(adapter, phase, seed):
    m = adapter.model(False)
    ck = torch.load(src(adapter.parent_checkpoint(seed)), map_location='cpu', weights_only=False)
    m.load_state_dict(ck['model_state'])
    rows, p = adapter.predict(m, phase, False)
    old = align(rows, csv(adapter.parent_predictions(seed, phase)))
    err = float(np.abs(p - old.probability.to_numpy()).max())
    np.testing.assert_allclose(p, old.probability, rtol=0, atol=2e-6)
    np.testing.assert_array_equal(p >= .5, old.probability.to_numpy() >= .5)
    AUDIT.append(dict(case=adapter.case, phase=phase, seed=seed, parent_replay_error=err))
    del m
    torch.cuda.empty_cache()
    return rows, old.probability.to_numpy()


def verify_shape(adapter):
    full, core = adapter.model(False), adapter.model(True)
    a, b = full.state_dict(), core.state_dict()
    assert a.keys() == b.keys()
    diff = [k for k in a if a[k].shape != b[k].shape]
    assert len(diff) == 1 and tuple(a[diff[0]].shape) == (128, 63) and tuple(b[diff[0]].shape) == (128, 56)
    oldn, newn = sum(p.numel() for p in full.parameters()), sum(p.numel() for p in core.parameters())
    assert oldn - newn == 896
    AUDIT.append(dict(case=adapter.case, parent_parameters=oldn, core56_parameters=newn, only_changed_weight=diff[0]))


def freeze():
    spec = {str(p.relative_to(ROOT)): sha(p) for p in (ROOT / 'DESIGN.md', Path(__file__))}
    path = ROOT / 'FROZEN.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != spec:
            assert not list(ROOT.glob('*/checkpoints/*.pt')), 'cannot change code/design after training'
            dump(ROOT / f'logs/pretraining_freeze_{sha(path)[:12]}.json', previous)
    dump(path, spec)
    return spec


def save_predictions(adapter, seed, phase, rows, p):
    out = rows[[c for c in ('sequence_id', 'label', 'set_id', 'video_id', 'ped_id', 'track_id', 'start_frame', 'end_frame') if c in rows]].copy()
    out['probability'], out['prediction'] = p, (p >= .5).astype(int)
    out['threshold'], out['seed'], out['case'], out['arm'] = .5, seed, adapter.case, 'core56'
    path = ROOT / f'predictions/{adapter.case}_{phase}_seed{seed}.csv'
    out.to_csv(path, index=False)
    return path


def main():
    global TEST
    for p in [ROOT, *[ROOT / c for c in CASES]]:
        for d in ('logs', 'metrics', 'checkpoints', 'predictions'):
            (p / d).mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available(), 'GPU required; run outside sandbox with WSL library path'
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    frozen = freeze()
    log(f'Exp117 audit started; GPU={torch.cuda.get_device_name(0)}; test gate CLOSED')
    adapters = [Mslstm(), Ped('pedcmt_pie'), Ped('pedcmt_jaad')]
    for a in adapters:
        rows = a.preparation()
        disjoint(rows, a.case)
        verify_shape(a)
        for seed in SEEDS:
            evaluate_parent(a, 'validation', seed)
        a.smoke()
        torch.cuda.empty_cache()
        log(f'PASS {a.case}: cache, feature, shape, three validation replays, GPU backward smoke')
    # Pin all loaded local implementation dependencies without touching archive bytecode.
    for mod in list(sys.modules.values()):
        file = getattr(mod, '__file__', None)
        if file and str(file).endswith('.py') and '/Hadwa/' in str(file):
            src(file)
    dump(ROOT / 'metrics/pretraining_audit.json', AUDIT)
    dump(ROOT / 'metrics/trainval_sources.json', SOURCES)
    dump(ROOT / 'PREPARED.json', dict(cases=list(CASES), seeds=list(SEEDS), test_accessed=False, gpu=torch.cuda.get_device_name(0)))
    trained = {}
    for a in adapters:
        for seed in SEEDS:
            for p, digest in frozen.items():
                assert sha(ROOT / p) == digest
            log(f'TRAIN {a.case} seed={seed}')
            started = time.monotonic()
            a.train(seed)
            model = a.model(True)
            ck = torch.load(a.new_checkpoint(seed), map_location='cpu', weights_only=False)
            model.load_state_dict(ck['model_state'])
            rows, prob = a.predict(model, 'validation', True)
            pred = save_predictions(a, seed, 'validation', rows, prob)
            for path in (a.new_checkpoint(seed), pred):
                trained[str(path)] = sha(path)
            dump(ROOT / f'metrics/{a.case}_seed{seed}_trained.json', dict(seconds=time.monotonic() - started, checkpoint=str(a.new_checkpoint(seed)), sha256=sha(a.new_checkpoint(seed))))
            log(f'TRAIN COMPLETE {a.case} seed={seed}; elapsed={time.monotonic()-started:.1f}s')
            del model
            torch.cuda.empty_cache()
    assert len(trained) == 18
    immutable()
    dump(ROOT / 'TRAINED.json', dict(artifacts=trained, models=9))
    for path, digest in trained.items():
        assert sha(path) == digest
    TEST = True
    log('All nine models and validation predictions frozen; final test gate OPEN')
    results, transitions = [], []
    metric_fn = adapters[1].runner.fixed_metrics
    for a in adapters:
        for seed in SEEDS:
            model = a.model(True)
            model.load_state_dict(torch.load(a.new_checkpoint(seed), map_location='cpu', weights_only=False)['model_state'])
            for phase in ('validation', 'test'):
                rows, p = a.predict(model, phase, True)
                if phase == 'test':
                    trainrows = a.rows if isinstance(a, Mslstm) else a.population.rows
                    disjoint(pd.concat([trainrows, rows], ignore_index=True), a.case)
                path = save_predictions(a, seed, phase, rows, p)
                saved = pd.read_csv(path)
                for k, v in metric_fn(rows.label, p).items():
                    np.testing.assert_allclose(metric_fn(saved.label, saved.probability)[k], v, atol=1e-12, rtol=0)
                results.append(dict(case=a.case, seed=seed, split=phase, arm='core56', threshold=.5, **metric_fn(rows.label, p)))
                _, old = evaluate_parent(a, phase, seed)
                oldok, newok = (old >= .5) == rows.label.to_numpy(), (p >= .5) == rows.label.to_numpy()
                transitions.append(dict(case=a.case, split=phase, seed=seed, corrected=int((~oldok & newok).sum()), harmed=int((oldok & ~newok).sum())))
                variants = ('G', 'H', 'I', 'J', 'K', 'L') if isinstance(a, Mslstm) else tuple('ABCDE')
                for variant in variants:
                    oldpred = align(rows, csv(a.parent_predictions(seed, phase, variant)))
                    results.append(dict(case=a.case, seed=seed, split=phase, arm=variant, threshold=.5,
                                        **metric_fn(rows.label, oldpred.probability.to_numpy())))
            del model
            torch.cuda.empty_cache()
        log(f'EVALUATED {a.case}: all seeds and saved reference controls verified')
    df = pd.DataFrame(results)
    df.to_csv(ROOT / 'metrics/per_seed_results.csv', index=False)
    df.to_csv(ROOT / 'metrics/fixed_0p5_results.csv', index=False)
    ms = list(metric_fn(np.array([0, 1]), np.array([.2, .8])))
    summary = df.groupby(['case', 'split', 'arm'])[ms].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(ROOT / 'metrics/summary_results.csv', index=False)
    deltas = []
    for (case, split, seed), g in df.groupby(['case', 'split', 'seed']):
        g = g.set_index('arm')
        for ref in g.index.drop('core56'):
            deltas.append(dict(case=case, split=split, seed=int(seed), reference=ref,
                               **{m: float(g.loc['core56', m] - g.loc[ref, m]) for m in ms}))
    pd.DataFrame(deltas).to_csv(ROOT / 'metrics/paired_deltas.csv', index=False)
    pd.DataFrame(transitions).to_csv(ROOT / 'metrics/error_transitions_vs_full63.csv', index=False)
    dump(ROOT / 'metrics/all_audits.json', AUDIT)
    dump(ROOT / 'metrics/all_sources.json', SOURCES)
    immutable()
    for path, digest in trained.items():
        assert sha(path) == digest
    for p, digest in frozen.items():
        assert sha(ROOT / p) == digest
    lines = ['# Exp117 final report', '', 'Retrospective matched neural core56 completion; fixed threshold 0.5.',
             'PR-AUC is average precision. Parent training and model-selection rules are unchanged.', '',
             '| Case | Arm | F1 mean | PR-AUC mean | ROC-AUC mean |', '|---|---|---:|---:|---:|']
    for (case, split, arm), row in summary.iterrows():
        if split == 'test':
            lines.append(f'| {case} | {arm} | {row.f1_mean:.6f} | {row.pr_auc_mean:.6f} | {row.roc_auc_mean:.6f} |')
    lines += ['', 'MSLSTM references G=core, H=v1, I=full63, J=raw pose, K=GT Action/Look, L=full union.',
              'PedCMT references A=core, B=raw pose, C=v1, D=full63, E=full union.',
              'All reference predictions are reused, not retrained. Full per-seed metrics and sample SD are in metrics/.',
              'No architecture or threshold was changed after test. No thesis files were edited. Stop; no further experiments.', '']
    (ROOT / 'FINAL_REPORT.md').write_text('\n'.join(lines))
    dump(ROOT / 'EXPERIMENT_COMPLETE.json', dict(models=9, test_after_all_training=True,
         artifacts={str(p.relative_to(ROOT)): sha(p) for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.csv', '.json', '.md', '.py', '.pt') and p.name != 'EXPERIMENT_COMPLETE.json'}))
    log('EXPERIMENT COMPLETE: nine models, all final evaluations and verification passed')


if __name__ == '__main__':
    main()
