import numpy as np
import pytest

from hp_fold import HPProblem
from hp_fold.analysis.result import RunResult
from hp_fold.analysis.summary import series_over_param, summarize
from hp_fold.viz.sweep import plot_metrics_vs_param


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


def test_each_panel_draws_the_matching_metric_against_the_parameter(problem, series):
    fig = plot_metrics_vs_param(problem, series)
    ground_panel, valid_panel = fig.axes
    (line,) = ground_panel.get_lines()
    assert np.asarray(line.get_xdata()).tolist() == [1, 2, 3]
    assert np.asarray(line.get_ydata()).tolist() == [
        row.p_ground.mean for row in series[0].rows
    ]
    (line,) = valid_panel.get_lines()
    assert np.asarray(line.get_ydata()).tolist() == [
        row.p_valid.mean for row in series[0].rows
    ]
