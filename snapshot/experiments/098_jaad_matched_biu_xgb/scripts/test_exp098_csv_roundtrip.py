import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def test_restore_original_float32_predictions(tmp_path):
    path = Path(__file__).with_name('verify_saved_outputs.py')
    assert path.exists(), 'round-trip verifier is missing'
    spec = importlib.util.spec_from_file_location('csv_roundtrip_verifier', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = np.array([0.123456789, .99987654, .50000006, .49999997], dtype=np.float32)
    file = tmp_path / 'p.csv'
    pd.DataFrame({'probability': original}).to_csv(file, index=False)
    restored = module.restore_probabilities(pd.read_csv(file))
    np.testing.assert_array_equal(restored, original)
