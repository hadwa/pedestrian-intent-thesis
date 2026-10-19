"""Verify the rewritten conclusion's numerical headlines against retained sources."""
from pathlib import Path
import csv
import json

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
text = (ROOT / 'thesis/manuscript/05_conclusion.tex').read_text()
checks = []


def check(source, selector, column, expected, digits=3):
    rows = [r for r in csv.DictReader((ROOT / source).open())
            if all(r[k] == v for k, v in selector.items())]
    assert len(rows) == 1, (source, selector, len(rows))
    actual = float(rows[0][column])
    passed = f'{actual:.{digits}f}' == expected and expected in text
    checks.append(dict(source=source, selector=json.dumps(selector), column=column,
                       actual=actual, displayed=expected, passed=passed))
    assert passed, checks[-1]


source = 'experiments/096_pie_exp072_matched_biu_ablation/metrics/test_summary_results.csv'
for arm, ap in [('base', '0.919'), ('v2', '0.939')]:
    check(source, {'arm': arm}, 'pr_auc_mean', ap)
    check(source, {'arm': arm}, 'f1_mean', '0.877')
source = 'experiments/116_xgboost_v2_core_completion/metrics/summary_results.csv'
for metric, expected in [('f1_mean', '0.883'), ('accuracy_mean', '0.935')]:
    check(source, {'case': 'pie_road', 'split': 'test', 'arm': 'core56'}, metric, expected)
source = 'experiments/117_neural_v2_core_completion/metrics/summary_results.csv'
for arm, expected in [('G', '0.810'), ('I', '0.829')]:
    check(source, {'case': 'mslstm_pie', 'split': 'test', 'arm': arm}, 'f1_mean', expected)
source = 'experiments/119_pedcmt_v2_capped_completion/metrics/summary_results.csv'
for arm, expected in [('baseline', '0.673'), ('full63', '0.698')]:
    check(source, {'case': 'jaad', 'split': 'test', 'arm': arm}, 'f1_mean', expected)
check('docs/retained_predictor_complexity_2026_09_16/evidence/latency_summary.csv',
      {'kind': 'xgb_road'}, 'mean_ms', '1.174')
for seed in ['42', '43', '44']:
    check('docs/retained_predictor_complexity_2026_09_16/evidence/serialized_sizes.csv',
          {'kind': 'xgb_road', 'seed': seed}, 'inference_MiB', '0.36', digits=2)
check('experiments/v2_xgboost_runtime_2026_09_13/headline.csv',
      {'model': 'road'}, 'actual_single_including_bbox_ms', '9.64', digits=2)
table = (ROOT / 'thesis/manuscript/Tables/ch4_literature_pie.tex').read_text()
for name, score in [('PCAP-SL: selected ViT-16-s', '0.863'), ('TrajFusionNet', '0.860')]:
    row = next(line for line in table.splitlines() if name in line)
    assert row.split('&')[2].strip() == score and score in text
    checks.append(dict(source='thesis/manuscript/Tables/ch4_literature_pie.tex',
                       selector=name, column='F1', actual=score, displayed=score, passed=True))
with (OUT / 'conclusion_numbers.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(checks[0]))
    writer.writeheader()
    writer.writerows(checks)
print(f'{len(checks)} conclusion-number checks passed. Literature rows checked against existing cited table, not reaudited papers.')
