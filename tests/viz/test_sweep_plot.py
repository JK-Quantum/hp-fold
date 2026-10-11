import numpy as np
import pytest

from hp_fold import HPProblem
from hp_fold.analysis.result import RunResult
from hp_fold.analysis.summary import series_over_param, summarize
from hp_fold.viz.sweep import plot_metrics_vs_param, random_baselines


@pytest.fixture(scope="module")
def problem():
    return HPProblem.from_sequence("PHPPH")


@pytest.fixture(scope="module")
def series(problem):
    ground = int(problem.ground_states[0])
    invalid = int(np.flatnonzero(~problem.valid)[0])
    runs = [
        RunResult.from_samples("qaoa", problem, xs, evaluations=10, config={"p": depth})
        for depth, xs in (
            (1, [ground, invalid]),
            (2, [ground, ground]),
            (3, [ground, ground]),
        )
    ]
    return series_over_param(summarize(problem, runs), "p")


def test_empty_series_are_rejected(problem):
    with pytest.raises(ValueError, match="at least one"):
        plot_metrics_vs_param(problem, [])


def data_line(ax, label):
    return next(line for line in ax.get_lines() if line.get_label() == label)


def test_each_panel_draws_the_matching_metric_against_the_parameter(problem, series):
    fig = plot_metrics_vs_param(problem, series)
    ground_panel, valid_panel = fig.axes
    line = data_line(ground_panel, "qaoa")
    assert np.asarray(line.get_xdata()).tolist() == [1, 2, 3]
    assert np.asarray(line.get_ydata()).tolist() == [
        row.p_ground.mean for row in series[0].rows
    ]
    line = data_line(valid_panel, "qaoa")
    assert np.asarray(line.get_ydata()).tolist() == [
        row.p_valid.mean for row in series[0].rows
    ]


def test_random_sampling_baselines_come_from_the_problem(problem):
    baselines = random_baselines(problem)
    assert baselines["p_ground"] == len(problem.ground_states) / 2**problem.n_qubits
    assert baselines["p_valid"] == int(problem.valid.sum()) / 2**problem.n_qubits


def test_each_panel_marks_the_random_sampling_baseline(problem, series):
    fig = plot_metrics_vs_param(problem, series)
    expected = random_baselines(problem)
    for ax, field in zip(fig.axes, ("p_ground", "p_valid"), strict=True):
        reference = data_line(ax, "_random")
        assert np.asarray(reference.get_ydata()).tolist() == [expected[field]] * 2


def test_panels_are_scaled_so_that_nothing_is_clipped(problem, series):
    fig = plot_metrics_vs_param(problem, series)
    for ax, field in zip(fig.axes, ("p_ground", "p_valid"), strict=True):
        highest = max(getattr(row, field).mean for row in series[0].rows)
        low, high = ax.get_ylim()
        assert low == 0 and high > highest


def test_the_x_axis_label_can_be_chosen(problem, series):
    fig = plot_metrics_vs_param(problem, series, x_label="QAOA depth p")
    assert fig.axes[-1].get_xlabel() == "QAOA depth p"
