import numpy as np
import pytest

from hp_fold import HPProblem
from hp_fold.analysis.result import RunResult
from hp_fold.viz.convergence import plot_convergence


@pytest.fixture(scope="module")
def problem():
    return HPProblem.from_sequence("PHPPH")


def traced(problem, algorithm, history) -> RunResult:
    extras = {} if history is None else {"objective_history": history}
    return RunResult.from_samples(algorithm, problem, [0], evaluations=1, extras=extras)


@pytest.fixture(scope="module")
def results(problem):
    return [
        ("qaoa", traced(problem, "qaoa", [5.0, 4.0, 4.0, 2.5])),
        ("qaoa_cvar", traced(problem, "qaoa_cvar", [3.0, 2.0])),
    ]


def data_line(ax, label):
    return next(line for line in ax.get_lines() if line.get_label() == label)


def test_empty_results_are_rejected(problem):
    with pytest.raises(ValueError, match="at least one"):
        plot_convergence(problem, [])


def test_one_panel_per_result_drawing_its_trace_against_the_iteration(problem, results):
    fig = plot_convergence(problem, results)
    assert len(fig.axes) == 2
    first = data_line(fig.axes[0], "qaoa")
    assert np.asarray(first.get_xdata()).tolist() == [1, 2, 3, 4]
    assert np.asarray(first.get_ydata()).tolist() == [5.0, 4.0, 4.0, 2.5]
    second = data_line(fig.axes[1], "qaoa_cvar")
    assert np.asarray(second.get_ydata()).tolist() == [3.0, 2.0]


def test_the_ground_state_energy_is_drawn_as_a_reference_in_every_panel(
    problem, results
):
    fig = plot_convergence(problem, results)
    for ax in fig.axes:
        reference = data_line(ax, "_ground")
        assert np.asarray(reference.get_ydata()).tolist() == [problem.ground_energy] * 2


def test_the_final_value_is_written_at_the_end_of_the_trace(problem, results):
    fig = plot_convergence(problem, results)
    assert "2.50" in [t.get_text() for t in fig.axes[0].texts]


def test_a_result_without_a_trace_is_rejected_and_named(problem):
    with pytest.raises(ValueError, match="qaoa.*objective_history"):
        plot_convergence(problem, [("qaoa", traced(problem, "qaoa", None))])


@pytest.mark.parametrize("bad", [[], [1.0, float("nan")], [[1.0, 2.0]]])
def test_a_malformed_trace_is_rejected(problem, bad):
    with pytest.raises(ValueError, match="objective_history"):
        plot_convergence(problem, [("qaoa", traced(problem, "qaoa", bad))])
