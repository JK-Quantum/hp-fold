import math

import numpy as np
import pytest

from hp_fold import HPProblem
from hp_fold.analysis.result import RunResult
from hp_fold.analysis.summary import summarize

GROUND = 19  # RDRU, a ground state of PHPPH (E = -1)
STRAIGHT = 0  # RRRR: valid, no contacts (E = 0)


@pytest.fixture(scope="module")
def problem():
    return HPProblem.from_sequence("PHPPH")


@pytest.fixture(scope="module")
def invalid_x(problem):
    return int(np.flatnonzero(~problem.valid)[0])


def run(problem, x, algorithm="alg", config=None, evaluations=10) -> RunResult:
    """A result whose whole probability mass sits on fold x."""
    return RunResult.from_samples(
        algorithm,
        problem,
        [x],
        evaluations=evaluations,
        config=config or {},
    )


def test_empty_input_gives_no_rows(problem):
    assert summarize(problem, []) == []


def test_rows_group_by_algorithm_and_config_in_order_of_first_appearance(problem):
    results = [
        run(problem, GROUND, "random", {"n_samples": 10}),
        run(problem, GROUND, "annealing", {"n_samples": 10}),
        run(problem, GROUND, "random", {"n_samples": 10}),
        run(problem, GROUND, "random", {"n_samples": 20}),
        run(problem, GROUND, "annealing", {"n_samples": 10}),
    ]
    rows = summarize(problem, results)
    assert [(r.algorithm, dict(r.config), r.runs) for r in rows] == [
        ("random", {"n_samples": 10}, 2),
        ("annealing", {"n_samples": 10}, 2),
        ("random", {"n_samples": 20}, 1),
    ]


def test_config_key_order_does_not_split_a_group(problem):
    a = run(problem, GROUND, config={"a": 1, "b": 2})
    b = run(problem, GROUND, config={"b": 2, "a": 1})
    assert [r.runs for r in summarize(problem, [a, b])] == [2]


def test_mean_and_sample_std_across_repeats(problem):
    results = [run(problem, GROUND), run(problem, STRAIGHT)]
    (row,) = summarize(problem, results)
    assert row.p_ground.mean == pytest.approx(0.5)
    assert row.p_ground.std == pytest.approx(math.sqrt(0.5))
    assert row.p_valid.mean == pytest.approx(1.0)
    assert row.approx_ratio.mean == pytest.approx(0.5)


def test_mean_evaluations_per_run(problem):
    results = [
        run(problem, GROUND, evaluations=10),
        run(problem, GROUND, evaluations=30),
    ]
    assert summarize(problem, results)[0].evaluations == pytest.approx(20.0)


def test_a_single_run_has_a_mean_but_no_std(problem):
    (row,) = summarize(problem, [run(problem, GROUND)])
    assert row.p_ground.mean == pytest.approx(1.0)
    assert math.isnan(row.p_ground.std)


def test_undefined_ratios_are_skipped_not_propagated(problem, invalid_x):
    results = [run(problem, GROUND), run(problem, invalid_x)]
    (row,) = summarize(problem, results)
    assert row.approx_ratio.mean == pytest.approx(1.0)
    assert math.isnan(row.approx_ratio.std)
    assert row.p_valid.mean == pytest.approx(0.5)


def test_ratio_is_nan_when_no_run_defines_it(problem, invalid_x):
    (row,) = summarize(problem, [run(problem, invalid_x)])
    assert math.isnan(row.approx_ratio.mean) and math.isnan(row.approx_ratio.std)


def test_results_for_another_sequence_are_rejected(problem):
    other = HPProblem.from_sequence("HHPHH")
    result = RunResult.from_samples("alg", other, [0], evaluations=1)
    with pytest.raises(ValueError, match="problem"):
        summarize(problem, [result])
