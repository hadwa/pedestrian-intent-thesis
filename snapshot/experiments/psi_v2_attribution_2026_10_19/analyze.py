"""Exact retained-model explanation. Never fits or writes to source experiments."""
from pathlib import Path
import csv
import hashlib
import json
import platform
import re
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT / 'experiments/116_xgboost_v2_core_completion'
REUSED = ROOT / 'experiments/124_psi_matched_lightweight_ablation'
HASHES = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(path):
    path = Path(path)
    HASHES[str(path.relative_to(ROOT))] = sha(path)
    return path


def read(path):
    return pd.read_csv(source(path), float_precision='round_trip')


def write(name, data):
    path = OUT / name
    assert not path.exists(), path
    if isinstance(data, pd.DataFrame):
        data.to_csv(path, index=False)
    else:
        path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def main():
    assert not (OUT / 'verification.json').exists(), 'Completed diagnostic exists'
    source(Path(__file__)); source(OUT / 'DESIGN.md')
    source(SOURCE / 'DESIGN.md'); source(REUSED / 'DESIGN.md')
    cache = SOURCE / 'cache/psi_ego/test'
    rows = read(cache / 'rows.csv')
    names = json.loads(source(cache / 'columns.json').read_text())
    features = np.load(source(cache / 'features.npy'), allow_pickle=False)
    defs = read(ROOT / 'experiments/012_pie_biu_v2_30_60_benchmark/metrics/pie_biu_v2_feature_set_definitions.csv').set_index('feature_set')
    canonical = json.loads(defs.loc['biu_v2_core', 'feature_columns_json'])
    assert names[53:109] == ['v2__' + n for n in canonical]
    assert len(set(names)) == len(names) == 135
    assert features.shape == (20547, 135) and np.isfinite(features).all()
    assert rows.sequence_id.is_unique and rows.track_id.nunique() == 96
    y = rows.label.to_numpy(int)
    assert int(y.sum()) == 15462
    frames = np.array(rows.observed_frames_json.map(json.loads).tolist())
    assert frames.shape == (len(rows), 15) and (np.diff(frames, axis=1) == 1).all()
    np.testing.assert_array_equal(frames[:, -1] + 1, rows.target_frame)

    membership = []
    for i, name in enumerate(names):
        if i < 53:
            block, cue = 'BBox', ''
        elif name.startswith('v2__'):
            block = 'BIU-v2'
            cue = re.sub(r'_(velocity_mean|velocity_std|mean|std|first|last|delta|min|max)$', '', name[4:])
        elif name.startswith('temporal__pose_confidence'):
            block, cue = 'Pose quality', ''
        elif name.startswith('temporal__'):
            block, cue = 'Early/late motion', ''
        elif name.startswith('obd_speed_'):
            block, cue = 'Ego speed', ''
        else:
            raise ValueError(name)
        membership.append(dict(index=i, feature=name, block=block, cue=cue))
    members = pd.DataFrame(membership)
    assert members.groupby('block').size().to_dict() == {
        'BBox': 53, 'BIU-v2': 56, 'Early/late motion': 18, 'Ego speed': 5, 'Pose quality': 3}
    cue_groups = {c: g['index'].to_numpy() for c, g in members[members.block == 'BIU-v2'].groupby('cue')}
    assert len(cue_groups) == 15
    blocks = {b: g['index'].to_numpy() for b, g in members.groupby('block')}
    checks, records, block_rows, feature_rows, configs = [], [], [], [], {}
    for seed in (42, 43, 44):
        model_path = source(SOURCE / f'models/psi_ego_seed{seed}.joblib')
        model = joblib.load(model_path)
        saved = read(SOURCE / f'predictions/psi_ego_test_seed{seed}.csv')
        reused = read(REUSED / f'predictions/test_C_seed{seed}.csv')
        for pred in (saved, reused):
            assert pred.sequence_id.is_unique
            np.testing.assert_array_equal(pred.sequence_id, rows.sequence_id)
            np.testing.assert_array_equal(pred.label, y)
        np.testing.assert_allclose(saved.probability, reused.probability, rtol=0, atol=1e-15)
        params = model.named_steps['model'].get_params()
        assert params['n_estimators'] == 250 and params['max_depth'] == 4 and params['random_state'] == seed
        imputer = model.named_steps['imputer']
        if hasattr(imputer, 'feature_names_in_'):
            np.testing.assert_array_equal(imputer.feature_names_in_, names)
        named = pd.DataFrame(features, columns=names)
        transformed = imputer.transform(named if hasattr(imputer, 'feature_names_in_') else features)
        assert transformed.shape == features.shape
        booster = model.named_steps['model'].get_booster()
        configs[str(seed)] = json.loads(booster.save_config())
        booster.set_param({'nthread': 4})
        dm = xgb.DMatrix(transformed, feature_names=booster.feature_names, nthread=4)
        probability = booster.predict(dm).astype(float)
        margin = booster.predict(dm, output_margin=True).astype(float)
        phi = booster.predict(dm, pred_contribs=True, approx_contribs=False).astype(float)
        assert phi.shape == (len(rows), 136)
        np.testing.assert_allclose(probability, saved.probability, rtol=0, atol=1e-7)
        np.testing.assert_array_equal(probability >= .5, saved.prediction)
        np.testing.assert_allclose(phi.sum(1), margin, rtol=1e-5, atol=2e-5)
        np.testing.assert_allclose(1 / (1 + np.exp(-margin)), probability, rtol=1e-6, atol=1e-7)
        mass = np.abs(phi[:, :-1]).mean(0)
        total = mass.sum(); biu_mass = mass[blocks['BIU-v2']].sum()
        for block, idx in blocks.items():
            block_rows.append(dict(seed=seed, block=block, features=len(idx),
                                   mean_abs_feature_mass=float(mass[idx].sum()), share_pct=float(100 * mass[idx].sum() / total)))
        feature_rows.extend(dict(seed=seed, **m, mean_abs_shap=float(mass[m['index']]),
                                 all_input_pct=float(100 * mass[m['index']] / total)) for m in membership)
        net = np.column_stack([phi[:, idx].sum(1) for idx in cue_groups.values()])
        np.savez_compressed(OUT / f'cue_contributions_seed{seed}.npz',
                            sequence_id=rows.sequence_id.to_numpy(str), track_id=rows.track_id.to_numpy(str),
                            label=y, probability=probability, margin=margin, baseline=phi[:, -1],
                            cue=np.array(list(cue_groups)), contribution=net)
        for subset, mask in [('ALL', np.ones(len(y), bool)), ('positive', y == 1), ('negative', y == 0)]:
            subset_mass = np.abs(phi[mask, :-1]).mean(0)
            for j, (cue, idx) in enumerate(cue_groups.items()):
                aligned = net[mask, j] * (2 * y[mask] - 1)
                records.append(dict(seed=seed, subset=subset, cue=cue, n=int(mask.sum()), features=len(idx),
                    importance_pct=float(100 * subset_mass[idx].sum() / subset_mass[blocks['BIU-v2']].sum()),
                    all_input_pct=float(100 * subset_mass[idx].sum() / subset_mass.sum()),
                    mean_abs_feature_mass=float(subset_mass[idx].sum()),
                    mean_abs_net=float(np.abs(net[mask, j]).mean()),
                    toward_pct=float(100 * np.mean(aligned > 1e-10)), away_pct=float(100 * np.mean(aligned < -1e-10)),
                    neutral_pct=float(100 * np.mean(np.abs(aligned) <= 1e-10)), net_mean=float(aligned.mean())))
        checks.append(dict(seed=seed, windows=len(rows), tracks=rows.track_id.nunique(), features=len(names),
                           max_probability_error=float(np.abs(probability - saved.probability).max()),
                           max_additivity_error=float(np.abs(phi.sum(1) - margin).max()),
                           max_sigmoid_error=float(np.abs(1 / (1 + np.exp(-margin)) - probability).max()),
                           same_exp124_predictions=True, all_decisions_match=True))
        print(f'Seed {seed}: exact model replay and SHAP additivity verified', flush=True)

    d = pd.DataFrame(records); b = pd.DataFrame(block_rows)
    np.testing.assert_allclose(d.toward_pct + d.away_pct + d.neutral_pct, 100, atol=1e-10)
    np.testing.assert_allclose(d.groupby(['seed', 'subset']).importance_pct.sum(), 100, atol=1e-10)
    np.testing.assert_allclose(b.groupby('seed').share_pct.sum(), 100, atol=1e-10)
    metrics = ['importance_pct', 'all_input_pct', 'mean_abs_feature_mass', 'mean_abs_net', 'toward_pct', 'away_pct', 'neutral_pct', 'net_mean']
    summary = d.groupby(['subset', 'cue'])[metrics].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    bs = b.groupby('block')[['mean_abs_feature_mass', 'share_pct']].agg(['mean', 'std'])
    bs.columns = ['_'.join(c) for c in bs.columns]
    write('membership.csv', members); write('per_seed.csv', d); write('summary.csv', summary.reset_index())
    write('block_per_seed.csv', b); write('block_summary.csv', bs.reset_index())
    write('feature_per_seed.csv', pd.DataFrame(feature_rows)); write('checks.csv', pd.DataFrame(checks))
    write('model_configurations.json', configs)
    for path, original in HASHES.items():
        assert sha(ROOT / path) == original, path
    write('source_hashes.json', HASHES)
    write('verification.json', dict(no_training=True, sources_unchanged=True, windows=len(rows), tracks=96,
        positive=15462, negative=5085, observation_frames=15, input_features=135, biu_features=56,
        seeds=[42, 43, 44], method='Exact XGBoost pred_contribs, training-path reference, raw log odds',
        reference_method='experiments/matched_biu_cue_rankings_2026_09_12/analyze.py',
        mean_biu_share_pct=float(bs.loc['BIU-v2', 'share_pct_mean']),
        max_probability_error=max(c['max_probability_error'] for c in checks),
        max_additivity_error=max(c['max_additivity_error'] for c in checks),
        python=platform.python_version(), xgboost=xgb.__version__, numpy=np.__version__, pandas=pd.__version__))
    print(bs.to_string())


if __name__ == '__main__':
    main()
