import json
from datetime import datetime, timedelta

import numpy as np
import pytest

from hp_fold import HPProblem
from hp_fold.analysis.metrics import compute_metrics
from hp_fold.analysis.result import RunResult
from hp_fold.analysis.store import load_result, save_result


@pytest.fixture(scope="module")
def problem():
    return HPProblem.from_sequence("PHPPH")


def sample_result(problem, **kwargs) -> RunResult:
    xs = [19, 19, 36, 0, 5]
    return RunResult.from_samples(
        "qaoa (p=2)", problem, xs, evaluations=500, seed=7, **kwargs
    )


def read_json(path):
    return json.loads(path.read_text())


def test_save_writes_json_and_npz_and_creates_directory(problem, tmp_path):
    path = save_result(problem, sample_result(problem), tmp_path / "nested" / "runs")
    assert path.suffix == ".json" and path.exists()
    assert path.with_suffix(".npz").exists()


def test_round_trip_is_exact(problem, tmp_path):
    original = sample_result(
        problem, config={"p": 2, "alpha": 0.25, "optimizer": "COBYLA"}
    )
    loaded = load_result(save_result(problem, original, tmp_path))
    assert np.array_equal(loaded.probs, original.probs)
    for field in ("algorithm", "seq", "shots", "evaluations", "seed", "config"):
        assert getattr(loaded, field) == getattr(original, field)


def test_array_extras_round_trip_and_stay_out_of_json(problem, tmp_path):
    history = np.random.default_rng(0).random((4, 64))
    original = sample_result(problem, extras={"history": history, "depth": 12})
    path = save_result(problem, original, tmp_path)
    assert "history" not in read_json(path)["result"]["extras"]
    loaded = load_result(path)
    assert np.array_equal(loaded.extras["history"], history)
    assert loaded.extras["depth"] == 12


def test_numpy_scalars_become_plain_python(problem, tmp_path):
    original = sample_result(
        problem, config={"alpha": np.float64(0.1), "p": np.int64(2)}
    )
    loaded = load_result(save_result(problem, original, tmp_path))
    assert loaded.config == {"alpha": 0.1, "p": 2}
    assert type(loaded.config["p"]) is int


def test_unserializable_config_raises_naming_the_key(problem, tmp_path):
    result = sample_result(problem, config={"backend": object()})
    with pytest.raises(TypeError, match="config.*backend"):
        save_result(problem, result, tmp_path)


def test_record_contains_problem_and_metrics(problem, tmp_path):
    result = sample_result(problem)
    record = read_json(save_result(problem, result, tmp_path))
    assert record["schema_version"] == 1
    assert record["problem"]["seq"] == "PHPPH"
    assert record["problem"]["penalty"] == problem.penalty
    assert record["metrics"]["p_ground"] == pytest.approx(
        compute_metrics(problem, result).p_ground
    )


def test_undefined_metrics_are_null_not_nan(problem, tmp_path):
    invalid_x = int(np.flatnonzero(~problem.valid)[0])
    result = RunResult.from_samples("test", problem, [invalid_x], evaluations=1)
    path = save_result(problem, result, tmp_path)
    strict = json.loads(
        path.read_text(), parse_constant=lambda c: pytest.fail(f"found {c}")
    )
    assert strict["metrics"]["approx_ratio"] is None


def test_provenance_is_recorded(problem, tmp_path):
    record = read_json(save_result(problem, sample_result(problem), tmp_path))
    created = datetime.fromisoformat(record["created_at"])
    assert created.utcoffset() == timedelta(0)
    prov = record["provenance"]
    assert isinstance(prov["python"], str)
    assert isinstance(prov["packages"]["numpy"], str)
    assert prov["git_commit"] is None or isinstance(prov["git_commit"], str)


def test_repeated_saves_get_distinct_safe_names(problem, tmp_path):
    first = save_result(problem, sample_result(problem), tmp_path)
    second = save_result(problem, sample_result(problem), tmp_path)
    assert first != second
    assert all(c.isalnum() or c in "-_." for c in first.name)
    assert load_result(first).algorithm == load_result(second).algorithm


def test_save_rejects_result_for_another_problem(problem, tmp_path):
    other = HPProblem.from_sequence("HHPHH")
    result = RunResult.from_samples("test", other, [0], evaluations=1)
    with pytest.raises(ValueError, match="problem"):
        save_result(problem, result, tmp_path)


def test_load_rejects_unknown_schema_version(problem, tmp_path):
    path = save_result(problem, sample_result(problem), tmp_path)
    record = read_json(path)
    record["schema_version"] = 99
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="schema"):
        load_result(path)
