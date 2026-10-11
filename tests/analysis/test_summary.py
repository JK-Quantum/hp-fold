import math

import numpy as np
import pytest

from hp_fold import HPProblem
from hp_fold.analysis.result import RunResult
from hp_fold.analysis.summary import series_over_param, summarize

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


def depth_rows(problem):
    return summarize(
        problem,
        [
            run(problem, GROUND, "qaoa", {"p": 2}),
            run(problem, STRAIGHT, "qaoa", {"p": 1}),
            run(problem, GROUND, "random", {"n_samples": 10}),
            run(problem, GROUND, "qaoa_cvar", {"p": 1}),
        ],
    )


def test_series_over_param_orders_each_algorithms_rows_by_the_parameter(problem):
    qaoa = series_over_param(depth_rows(problem), "p")[0]
    assert qaoa.algorithm == "qaoa" and qaoa.param == "p"
    assert qaoa.values == (1.0, 2.0)
    assert [row.p_ground.mean for row in qaoa.rows] == [
        0.0,
        1.0,
    ]


def test_series_over_param_skips_rows_without_the_parameter(problem):
    series = series_over_param(depth_rows(problem), "p")
    assert [s.algorithm for s in series] == ["qaoa", "qaoa_cvar"]


def test_series_over_param_keeps_algorithms_in_order_of_first_appearance(problem):
    rows = summarize(
        problem,
        [
            run(problem, GROUND, "qaoa_cvar", {"p": 1}),
            run(problem, GROUND, "qaoa", {"p": 1}),
        ],
    )
    assert [s.algorithm for s in series_over_param(rows, "p")] == ["qaoa_cvar", "qaoa"]


def test_series_over_param_with_no_matching_rows_is_empty(problem):
    assert series_over_param(depth_rows(problem), "layers") == []


@pytest.mark.parametrize("bad", ["two", True, None])
def test_series_over_param_requires_a_number(problem, bad):
    rows = summarize(problem, [run(problem, GROUND, "qaoa", {"p": bad})])
    with pytest.raises(ValueError, match="number"):
        series_over_param(rows, "p")


def test_series_over_param_rejects_two_rows_at_the_same_value(problem):
    rows = summarize(
        problem,
        [
            run(problem, GROUND, "qaoa", {"p": 1, "shots": 10}),
            run(problem, GROUND, "qaoa", {"p": 1, "shots": 20}),
        ],
    )
    with pytest.raises(ValueError, match="same"):
        series_over_param(rows, "p")
