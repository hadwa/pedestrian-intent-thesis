"""Summarize saved fixed-0.5 confusion counts; never train or select a model."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'experiments/thesis_biu_version_fp_fn_verdict_2026_09_11'
SOURCES = {}
RECORDS = []


def read(num, name):
    folder, = ROOT.glob(f'experiments/{num}_*')
    path = folder / name
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return list(csv.DictReader(path.open()))


def document(num, name='DESIGN.md'):
    folder, = ROOT.glob(f'experiments/{num}_*')
    path = folder / name
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()


def add(setting, rows, key, mapping, prefix='', category='main'):
    for r in rows:
        if r[key] not in mapping:
            continue
        version = mapping[r[key]]
        threshold = r.get('threshold', r.get('selected_threshold', '0.5'))
        assert float(threshold) == .5
        counts = {m: float(r[prefix + m]) for m in ['tn', 'fp', 'fn', 'tp']}
        assert all(v >= 0 and v.is_integer() for v in counts.values())
        n = sum(counts.values())
        assert abs(1 - (counts['fp'] + counts['fn']) / n - float(r[prefix + 'accuracy'])) < 1e-10
        RECORDS.append(dict(category=category, setting=setting, version=version,
                            seed=int(r['seed']), threshold=.5, n=int(n),
                            **{k: int(v) for k, v in counts.items()}))


if __name__ == '__main__':
    # Explicit assembly keeps distinct cohorts and input pathways separate.
    OUT.mkdir(exist_ok=False)
    document('012', 'README.md')
    rows = read('012', 'metrics/pie_biu_v2_benchmark_results.csv')
    for model in ['logreg', 'rf', 'xgb']:
        add(f'PIE BBox+BIU {model} (Exp012)',
            [r for r in rows if r['model_name'] == model and r['threshold_rule'] == 'fixed_0_5'],
            'feature_set', {'bbox_only': 'Baseline', 'biu_v1_plus_bbox': 'v1',
                            'biu_v2_core_plus_bbox': 'v2-core'}, prefix='test_')
    for num, setting, filename, mapping in [
        ('096', 'PIE extended XGBoost (Exp096)', 'test_per_seed_results.csv',
         {'base': 'Baseline', 'v1': 'v1', 'v2': 'v2-core', 'combined': 'Historical Combined'}),
        ('098', 'Clean JAAD BBox+BIU XGBoost (Exp098)', 'per_seed_results.csv',
         {'A': 'Baseline', 'B': 'v1', 'C': 'v2-full', 'D': 'Full union'}),
        ('100', 'Clean JAAD extended XGBoost (Exp100/101)', 'per_seed_results.csv',
         {'v1': 'v1', 'v2': 'v2-core', 'combined': 'Historical Combined'}),
        ('101', 'Clean JAAD extended XGBoost (Exp100/101)', 'per_seed_results.csv',
         {'full63': 'v2-full'}),
    ]:
        document(num)
        add(setting, read(num, 'metrics/' + filename), 'arm', mapping)
    document('102')
    rows = read('102', 'metrics/per_seed_results.csv')
    for cohort in ['jaad_beh', 'jaad_all']:
        add(f'{cohort} API BBox+BIU XGBoost (Exp102)', [r for r in rows if r['cohort'] == cohort],
            'arm', {'A': 'Baseline', 'B': 'v1', 'C': 'v2-core', 'D': 'v2-full', 'E': 'Full union'})
    for num, setting, category in [
        ('084', 'PIE adapted PedCMT (Exp084)', 'main'),
        ('089', 'Clean JAAD adapted PedCMT (Exp089)', 'main'),
        ('085', 'PIE native-protocol PedCMT (Exp085)', 'separate protocol'),
        ('086', 'JAAD native-protocol PedCMT (Exp086)', 'separate protocol'),
    ]:
        document(num)
        rows = read(num, 'metrics/per_seed_results.csv')
        rows = [r for r in rows if r.get('run_type', 'controlled') == 'controlled']
        add(setting, rows, 'variant', {'A': 'Baseline', 'B': 'Raw pose', 'C': 'v1',
                                      'D': 'v2-full', 'E': 'Full union'}, category=category)
    document('095')
    add('PIE adapted MSLSTM BBox53 (Exp095)', read('095', 'metrics/all_controlled_results.csv'),
        'variant', {'G': 'Baseline', 'H': 'v1', 'I': 'v2-full', 'J': 'Raw pose',
                    'K': 'GT Action/Look', 'L': 'Full union'})
    # Authenticate reused MSLSTM and extended-XGBoost arms, and verified JAAD metrics.
    for num, filename, setting, mapping, key in [
        ('053', 'metrics/new_per_seed_results.csv', 'PIE adapted MSLSTM BBox53 (Exp095)',
         {'G': 'Baseline', 'H': 'v1', 'I': 'v2-full'}, 'variant'),
        ('101', 'metrics/per_seed_results.csv', 'Clean JAAD extended XGBoost (Exp100/101)',
         {'core56': 'v2-core'}, 'arm'),
        ('089', 'metrics/verified_per_seed_results.csv', 'Clean JAAD adapted PedCMT (Exp089)',
         {'A': 'Baseline', 'B': 'Raw pose', 'C': 'v1', 'D': 'v2-full', 'E': 'Full union'}, 'variant'),
    ]:
        for r in read(num, filename):
            if r[key] in mapping:
                target, = [x for x in RECORDS if x['setting'] == setting and
                           x['version'] == mapping[r[key]] and x['seed'] == int(r['seed'])]
                assert all(target[m] == float(r[m]) for m in ['fp', 'fn'])
    groups = defaultdict(list)
    for r in RECORDS:
        groups[(r['category'], r['setting'], r['version'])].append(r)
    summaries = []
    for (category, setting, version), rows in groups.items():
        assert sorted(r['seed'] for r in rows) == [42, 43, 44]
        assert len({(r['tn'] + r['fp'], r['tp'] + r['fn']) for r in rows}) == 1
        entry = dict(category=category, setting=setting, version=version, seeds=3,
                     n=rows[0]['n'], positives=rows[0]['tp'] + rows[0]['fn'],
                     negatives=rows[0]['tn'] + rows[0]['fp'])
        for metric in ['fp', 'fn', 'errors', 'fpr', 'fnr']:
            vals = [r[metric] if metric in r else
                    r['fp'] + r['fn'] if metric == 'errors' else
                    r['fp'] / (r['tn'] + r['fp']) if metric == 'fpr' else
                    r['fn'] / (r['tp'] + r['fn']) for r in rows]
            entry[metric + '_mean'] = mean(vals)
            entry[metric + '_sd'] = stdev(vals)
        summaries.append(entry)
    settings = list(dict.fromkeys(r['setting'] for r in summaries))
    winners = []
    for setting in settings:
        rows = [r for r in summaries if r['setting'] == setting]
        assert len({(r['n'], r['positives'], r['negatives']) for r in rows}) == 1
        candidates = [r for r in rows if r['version'] not in ['Baseline', 'Raw pose', 'GT Action/Look']]
        item = dict(category=rows[0]['category'], setting=setting)
        for metric in ['fp', 'fn', 'errors']:
            best = min(r[metric + '_mean'] for r in candidates)
            item[metric + '_winner'] = ', '.join(r['version'] for r in candidates if abs(r[metric + '_mean'] - best) < 1e-10)
            item[metric + '_mean'] = best
        winners.append(item)
    for name, rows in [('per_seed.csv', RECORDS), ('summary.csv', summaries), ('winners.csv', winners)]:
        with (OUT / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ['# BIU version FP/FN audit', '',
             'Descriptive audit of completed saved results; no training, inference, threshold selection or model selection. Fixed threshold 0.5; means across seeds 42/43/44, not a seed ensemble. FP = non-crossing windows predicted crossing; FN = crossing windows predicted non-crossing. Counts are windows, not unique pedestrians. Fractional counts arise from averaging seeds.', '',
             '## Observed minima among BIU versions', '',
             '| Setting | Lowest FP | Lowest FN | Lowest FP+FN |',
             '|---|---:|---:|---:|']
    for r in winners:
        if r['category'] != 'main': continue
        lines.append('| ' + r['setting'] + ' | ' + ' | '.join(f"{r[m + '_winner']} ({r[m + '_mean']:.2f})" for m in ['fp', 'fn', 'errors']) + ' |')
    lines += ['', '## Full counts, including non-BIU controls', '',
              '| Setting | Version/control | N | FP mean ± SD | FN mean ± SD | FP+FN | FPR % | FNR % |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f"| {r['setting']} | {r['version']} | {r['n']} | {r['fp_mean']:.2f} ± {r['fp_sd']:.2f} | {r['fn_mean']:.2f} ± {r['fn_sd']:.2f} | {r['errors_mean']:.2f} | {100*r['fpr_mean']:.2f} | {100*r['fnr_mean']:.2f} |")
    lines += ['', '## Interpretation and boundaries', '',
              '- No version has the lowest FP and FN across all tested settings. These are observed mean minima, not statistical superiority claims or recommendations chosen on test performance.',
              '- In Exp012, RF favors v1 for both errors; XGBoost favors v2-core for both, with a small margin; logistic regression trades one additional FP for six fewer FN using v2-core. Exp012 has no Combined arm.',
              '- In the main PIE PedCMT and MSLSTM controls, v2 has the lowest FN, while the full union has the lowest FP among BIU arms. PedCMT favors union on total errors; MSLSTM favors v2. Native-protocol Exp085/086 are separate supporting rows, not pooled thesis benchmark results.',
              '- Exp096 favors v2-core on FP and total errors among BIU arms; historical Combined averages only 0.67 fewer FN. The no-BIU baseline has fewer total errors (255) than any BIU arm at 0.5, so a version verdict must not be described as universal improvement over the baseline.',
              '- Exp100/101 favors v1 on FP and v2-core on FN and total errors. Exp101 confirms that adding seven metadata/scale features to the core does not reduce either mean error count in this pathway.',
              '- Historical Combined (Exp096/100) deduplicates overlapping feature names. Full union (Exp084/085/086/089/095/098/102) retains source-namespaced v1 and v2 features. They are not identical representations. v2-core is 56 features; v2-full is 63. Extended pathways can also include additional v2 temporal summaries.',
              '- Exp096 is the Exp072 temporal/ego architecture, not a road-context version control. Exp100/101 use categorical vehicle action, not physical ego speed. Exp083 road-context JAAD results have no matched v1/v2/Combined arms here; comparing them to Exp102 would change inputs as well as version.',
              '- JAAD cohorts remain separate; these are in-domain results, not PIE-to-JAAD zero-shot evidence. Their final interpretation and calibration remain pending.',
              '- FP+FN weights both errors equally and depends on class prevalence. It is not a safety-weighted objective. Do not pool counts across different cohorts or treat overlapping windows as independent pedestrians. Seed SD is not a confidence interval.',
              '- Exp009 is excluded from the main verdict: its report records 30 missing 30–60 keypoint rows, and it spans multiple horizons; use the repaired matched JAAD controls above. Exp054 isolates v2 preprocessing, not a matched three-version comparison. Validation-only and fusion-redesign PedCMT trials do not supply interchangeable version controls.', '',
              '## Reproduction and files', '',
              'Command (from the active repository; creates a new folder and refuses to overwrite):', '',
              '```bash', 'python3 scripts/audits/summarize_biu_version_fp_fn.py', '```', '',
              'Outputs: REPORT.md, per_seed.csv, summary.csv, winners.csv, sources.json, verification.json in experiments/thesis_biu_version_fp_fn_verdict_2026_09_11/.', '',
              'Exact source paths and SHA-256 hashes are recorded in sources.json. Counts were reused; seed means, sample SD, rates and minima were recomputed. No original source was modified.', '',
              '## Sources', '']
    lines += [f'- `{p}`' for p in SOURCES]
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    for p, expected in SOURCES.items():
        assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == expected
    (OUT / 'sources.json').write_text(json.dumps(SOURCES, indent=2) + '\n')
    (OUT / 'verification.json').write_text(json.dumps(dict(
        per_seed_rows=len(RECORDS), summary_rows=len(summaries), settings=len(settings),
        source_files_unchanged=len(SOURCES), confusion_accuracy_parity=True,
        three_seeds_per_arm=True, class_counts_matched_within_setting=True,
        reused_mslstm_and_v2_core_counts_authenticated=True,
        jaad_pedcmt_verified_count_parity=True), indent=2) + '\n')
    print('\n'.join(lines[:20]))
    print(OUT)
