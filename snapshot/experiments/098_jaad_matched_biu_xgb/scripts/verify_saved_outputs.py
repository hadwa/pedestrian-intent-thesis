#!/usr/bin/env python3
"""Verify original float32 probabilities without altering the frozen runner/results."""

import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


def restore_probabilities(frame):
    # XGBoost returned float32; pandas wrote shortest round-trip float32 decimals.
    return frame.probability.to_numpy(dtype=np.float32)


def main():
    path = Path(__file__).with_name('run_experiment.py')
    spec = importlib.util.spec_from_file_location('exp098_frozen_verification', path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    for phase in ('prepare', 'validate', 'evaluate', 'report'):
        runner.authenticate(phase)
    audited = []
    for split in ('validation', 'test'):
        table_path = runner.ROOT / 'metrics' / ('validation_per_seed_results.csv' if split == 'validation'
                                               else 'per_seed_results.csv')
        table = pd.read_csv(table_path)
        runner.check_grid(table)
        identities = None
        for arm, seed in itertools.product(runner.ARMS, runner.SEEDS):
            file = runner.ROOT / 'predictions' / f'{split}_{arm}_seed{seed}.csv'
            frame = pd.read_csv(file)
            current = list(zip(frame.sequence_id, frame.label, frame.observed_frames_json))
            if identities is None:
                identities = current
            assert current == identities
            assert len(current) == (274 if split == 'validation' else 1501)
            assert frame.threshold.eq(.5).all()
            p = restore_probabilities(frame)
            np.testing.assert_array_equal(frame.prediction, (p >= .5).astype(int))
            raw = runner.core.compute_metrics(frame.label.to_numpy(), frame.probability.to_numpy())
            metrics = runner.core.compute_metrics(frame.label.to_numpy(), p)
            saved = table.loc[(table.arm == arm) & (table.seed == seed)].iloc[0]
            metric_names = [name for name in metrics if name in saved.index]
            for name in metric_names:
                np.testing.assert_allclose(metrics[name], saved[name], rtol=1e-12, atol=1e-12)
            audited.append(dict(scope=split, arm=arm, seed=seed,
                                max_restored_metric_difference=max(abs(metrics[k] - saved[k]) for k in metric_names),
                                csv_float64_brier_difference=raw['brier'] - saved['brier'],
                                probability_file=runner.record(file)))
    output = runner.ROOT / 'metrics/saved_prediction_verification.json'
    runner.write(output, dict(status='passed', procedure='restore original float32 then recompute metrics',
                              frozen_design_or_training_changes=False, checked=audited))
    runner.write(runner.ROOT / 'EXPERIMENT_COMPLETE.json', dict(status='complete', experiment=98,
                 runs=12, test_rows=1501, no_refit=True, verification=runner.record(output),
                 verifier=runner.record(Path(__file__)), report=runner.record(runner.ROOT / 'FINAL_REPORT.md'),
                 classification='retrospective matched ablation',
                 execution_note='EXECUTION_NOTE.md'))
    with (runner.ROOT / 'logs/commands.log').open('a') as handle:
        handle.write(str(Path(__file__)) + '\n')
    print('24 prediction files verified; all original metrics reproduced. EXPERIMENT_COMPLETE.json created.')


if __name__ == '__main__':
    main()
