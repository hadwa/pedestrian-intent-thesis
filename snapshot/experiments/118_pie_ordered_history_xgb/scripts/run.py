"""Frozen train/validation-only ordered-history comparison; never load test."""
import hashlib
import importlib.util
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
PARENT = REPO / 'experiments/096_pie_exp072_matched_biu_ablation'
CACHE = REPO / 'experiments/040_pie_temporal_biu_sequence/metrics'


def load_module(path):
    spec = importlib.util.spec_from_file_location('exp118_parent', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def main():
    if (ROOT / 'metrics/FROZEN.json').exists():
        raise RuntimeError('Existing run: refuse overwrite')
    for name in ('metrics', 'models', 'predictions', 'logs'):
        (ROOT / name).mkdir(exist_ok=True)
    parent = load_module(PARENT / 'scripts/run_experiment.py')
    sources = {}
    marker = json.loads((CACHE / '.prepare.complete.json').read_text())
    for name, record in marker['outputs'].items():
        path = CACHE / name
        assert sha(path) == record['sha256'], name
        sources[str(path)] = sha(path)
    meta = pd.read_csv(CACHE / 'prepared_metadata.csv')
    seq = np.load(CACHE / 'prepared_sequences.npz', allow_pickle=False)['sequences']
    contract = json.loads((CACHE / 'sequence_contract.json').read_text())
    assert seq.shape == (5985, 16, 26) and np.isfinite(seq).all()
    assert not meta.sequence_id.duplicated().any() and 'set03' not in set(meta.set_id)
    channel_indices = list(range(16)) + list(range(19, 24))
    channels = [contract['channel_names'][i] for i in channel_indices]
    ordered_names = [f'ordered_f{t+1:02d}__{c}' for t in range(16) for c in channels]
    columns = list(parent.arm_columns()['v2'])
    assert len(columns) == 135 and len(ordered_names) == 336
    meta = meta.assign(cache_row=np.arange(len(meta))).set_index('sequence_id')
    frames, arrays = {}, {}
    for split in ('train', 'validation'):
        frame = parent._load(split)
        matched = meta.loc[frame.sequence_id]
        for key in ('set_id', 'track_id', 'video_id', 'label'):
            assert np.array_equal(frame[key].to_numpy(), matched[key].to_numpy()), key
        selected = seq[matched.cache_row.to_numpy()][:, :, channel_indices]
        flat = selected.reshape(len(frame), -1)
        assert np.array_equal(flat.reshape(len(frame), 16, 21), selected)
        assert np.array_equal(flat, seq[matched.cache_row.to_numpy()][:, :, channel_indices].reshape(len(frame), -1))
        extra = pd.DataFrame(flat.astype(float), columns=ordered_names)
        frames[split] = frame
        arrays[split] = pd.concat([frame[columns].astype(float), extra], axis=1)
        assert arrays[split].shape == (len(frame), 471)
        sources[str(parent.source_path(split))] = sha(parent.source_path(split))
    assert set(frames['train'].track_id).isdisjoint(frames['validation'].track_id)
    assert set(frames['train'].sequence_id).isdisjoint(frames['validation'].sequence_id)
    assert set(frames['train'].sequence_id) | set(frames['validation'].sequence_id) == set(meta.index)
    saved_path = PARENT / 'predictions/validation_predictions.csv'
    saved = pd.read_csv(saved_path)
    sources[str(saved_path)] = sha(saved_path)
    labels = frames['train'].label.to_numpy(int)
    validation = frames['validation']
    results, replay, models = [], [], {}
    for seed in (42, 43, 44):
        path = PARENT / f'models/v2_seed{seed}.joblib'
        model = joblib.load(path)
        sources[str(path)] = sha(path)
        fresh = parent.model_core.build_candidate(seed, labels)
        # NaN is the identical missing-value policy but does not equal itself.
        assert json.dumps(model['model'].get_params(), sort_keys=True) == json.dumps(fresh['model'].get_params(), sort_keys=True)
        assert json.dumps(model['imputer'].get_params(), sort_keys=True) == json.dumps(fresh['imputer'].get_params(), sort_keys=True)
        assert list(model.feature_names_in_) == columns
        p = model.predict_proba(validation[columns].astype(float))[:, 1]
        old = saved[(saved.arm == 'v2') & (saved.seed == seed)].set_index('sequence_id').loc[validation.sequence_id]
        assert np.array_equal(old.label.to_numpy(), validation.label.to_numpy())
        error = float(np.max(np.abs(p - old.probability.to_numpy())))
        assert error < 1e-7 and np.array_equal(p >= .5, old.prediction.to_numpy())
        replay.append(dict(seed=seed, max_abs_error=error))
        models[seed] = fresh
    frozen = dict(sources=sources, design_sha=sha(ROOT / 'DESIGN.md'),
                  runner_sha=sha(Path(__file__)), columns=columns + ordered_names,
                  channels=channels, baseline_replay=replay, invariants='PASS',
                  train_n=len(labels), validation_n=len(validation), test_accessed=False,
                  python=platform.python_version(), sklearn=sklearn.__version__, xgboost=xgboost.__version__)
    write_json(ROOT / 'metrics/FROZEN.json', frozen)
    print('All cache, alignment, leakage and baseline replay checks PASS', flush=True)
    for seed, model in models.items():
        started = time.perf_counter()
        model.fit(arrays['train'], labels)
        fit_seconds = time.perf_counter() - started
        joblib.dump(model, ROOT / f'models/ordered_seed{seed}.joblib')
        probs = model.predict_proba(arrays['validation'])[:, 1]
        baseline = saved[(saved.arm == 'v2') & (saved.seed == seed)].set_index('sequence_id').loc[validation.sequence_id].probability.to_numpy()
        for arm, probability in [('baseline', baseline), ('ordered', probs)]:
            out = validation[['sequence_id', 'set_id', 'track_id', 'video_id', 'label']].copy()
            out['probability'] = probability
            out['prediction'] = (probability >= .5).astype(int)
            out.to_csv(ROOT / f'predictions/{arm}_seed{seed}_validation.csv', index=False)
            metrics = parent.model_core.compute_metrics(out.label.to_numpy(), probability)
            results.append(dict(arm=arm, seed=seed, dimension=135 if arm == 'baseline' else 471,
                                fit_seconds=0 if arm == 'baseline' else fit_seconds, **metrics))
        print(f'Seed {seed} complete; fit {fit_seconds:.2f}s', flush=True)
    table = pd.DataFrame(results)
    table.to_csv(ROOT / 'metrics/per_seed_results.csv', index=False)
    metric_names = ['f1', 'pr_auc', 'roc_auc', 'accuracy', 'precision', 'recall', 'fp', 'fn', 'fp_ge_0p9_count', 'fn_le_0p1_count', 'fit_seconds']
    summary = table.groupby('arm')[metric_names].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.to_csv(ROOT / 'metrics/summary_results.csv')
    delta = table[table.arm == 'ordered'].set_index('seed')[metric_names] - table[table.arm == 'baseline'].set_index('seed')[metric_names]
    delta.to_csv(ROOT / 'metrics/paired_deltas.csv')
    promising = bool((delta[['f1', 'pr_auc']].mean() > 0).all() and (delta[['f1', 'pr_auc']] >= -.005).all().all())
    write_json(ROOT / 'metrics/decision.json', dict(promising=promising, test_authorized=False,
               all_seed_positive={m: bool((delta[m] > 0).all()) for m in ('f1', 'pr_auc', 'roc_auc')}))
    assert sha(ROOT / 'DESIGN.md') == frozen['design_sha']
    assert sha(Path(__file__)) == frozen['runner_sha']
    assert all(sha(Path(p)) == h for p, h in sources.items())
    print(summary.to_string(), flush=True)
    write_json(ROOT / 'TRAIN_VALIDATION_COMPLETE.json', dict(status='complete', seeds=[42, 43, 44], test_accessed=False))


if __name__ == '__main__':
    main()
