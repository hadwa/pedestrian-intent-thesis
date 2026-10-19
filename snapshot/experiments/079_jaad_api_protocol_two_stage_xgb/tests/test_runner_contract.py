import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_experiment.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("exp079_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_protocol_a_split_contract_is_frozen():
    runner = _load_runner()

    expected = {"train": 8613, "validation": 1265, "test": 6732}
    for split, rows in expected.items():
        frame = runner.load_jaad_split(split)
        assert len(frame) == rows
        assert frame["sequence_id"].is_unique
        assert frame["observation_length"].eq(16).all()
        assert frame["tte_frames"].between(30, 60).all()
        assert frame["protocol"].eq("api_style_reconstruction").all()
    train = runner.load_jaad_split("train")
    gaps = [row for row in train.itertuples(index=False) if runner.observed_frames(row)[-1] - runner.observed_frames(row)[0] != 15]
    assert len(gaps) == 2
    assert runner.observed_frames(gaps[0]) == (56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 202, 203, 204)


def test_model_recipe_matches_exp077():
    runner = _load_runner()

    assert tuple(runner.SEEDS) == (42, 43, 44)
    assert runner.TOTAL_TREE_BUDGET == 250
    assert runner.OOF_SPLITS == 5
    assert runner.FIXED_THRESHOLD == 0.5
    assert runner.EXP077_ZERO_SHOT_MODELS.is_dir()
