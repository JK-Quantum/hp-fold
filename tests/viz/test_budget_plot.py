import matplotlib.colors as mcolors
import numpy as np
import pytest

from hp_fold import HPProblem
from hp_fold.analysis.metrics import success_vs_evaluations
from hp_fold.analysis.result import RunResult
from hp_fold.viz.budget import budget_grid, plot_success_vs_evaluations, success_rows
from hp_fold.viz.theme import algorithm_colors, get_theme


@pytest.fixture(scope="module")
def problem():
    return HPProblem.from_sequence("PHPPH")


def run(problem, algorithm, p_ground, shots, evaluations, training=None) -> RunResult:
    size = 2**problem.n_qubits
    ground = problem.ground_states
    probs = np.full(size, (1 - p_ground) / (size - len(ground)))
    probs[ground] = p_ground / len(ground)
    extras = {} if training is None else {"training_evaluations": training}
    return RunResult(algorithm, problem.seq, probs, shots, evaluations, extras=extras)


@pytest.fixture(scope="module")
def results(problem):
    return [
        ("random (seed 0)", run(problem, "random", 0.06, 200, 200)),
        ("qaoa (seed 0)", run(problem, "qaoa", 0.4, 1000, 1600, training=600)),
    ]


def test_grid_is_integer_ascending_and_spans_one_to_the_largest_budget(results):
    grid = budget_grid(results)
    assert grid[0] == 1 and grid[-1] == 1600
    assert np.issubdtype(grid.dtype, np.integer)
    assert (np.diff(grid) > 0).all()


def test_grid_of_nothing_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        budget_grid([])


def test_rows_are_the_plotted_numbers_as_a_table(problem, results):
    grid = budget_grid(results)
    rows = success_rows(problem, results, grid)
    assert len(rows) == len(results) * len(grid)
    assert {r["algorithm"] for r in rows} == {"random (seed 0)", "qaoa (seed 0)"}
    qaoa = [r["p_success"] for r in rows if r["algorithm"] == "qaoa (seed 0)"]
    assert qaoa == pytest.approx(
        success_vs_evaluations(problem, results[1][1], grid).tolist()
    )
    assert set(rows[0]) == {"algorithm", "evaluations", "p_success"}


def test_empty_results_are_rejected(problem):
    with pytest.raises(ValueError, match="at least one"):
        plot_success_vs_evaluations(problem, [])


def test_one_line_per_result_carrying_the_success_curve(problem, results):
    fig = plot_success_vs_evaluations(problem, results)
    ax = fig.axes[0]
    assert [line.get_label() for line in ax.get_lines()] == [
        label for label, _ in results
    ]
    for line, (_, result) in zip(ax.get_lines(), results, strict=True):
        x, y = line.get_xdata(), line.get_ydata()
        assert np.asarray(y).tolist() == pytest.approx(
            success_vs_evaluations(problem, result, np.asarray(x)).tolist()
        )


def test_an_algorithm_keeps_the_color_assigned_to_it(problem, results):
    theme = get_theme("light")
    expected = algorithm_colors(theme, ["random", "qaoa"])
    fig = plot_success_vs_evaluations(problem, results, theme="light")
    for line, (_, result) in zip(fig.axes[0].get_lines(), results, strict=True):
        assert mcolors.same_color(line.get_color(), expected[result.algorithm])


def test_lines_are_two_points_wide(problem, results):
    fig = plot_success_vs_evaluations(problem, results)
    assert all(line.get_linewidth() == 2 for line in fig.axes[0].get_lines())


def test_budget_axis_is_logarithmic_and_probability_axis_shows_zero_to_one(
    problem,
    results,
):
    ax = plot_success_vs_evaluations(problem, results).axes[0]
    assert ax.get_xscale() == "log"
    low, high = ax.get_ylim()
    assert low < 0 and high > 1


def test_legend_lists_every_result_in_order(problem, results):
    fig = plot_success_vs_evaluations(problem, results)
    assert [t.get_text() for t in fig.legends[0].get_texts()] == [
        label for label, _ in results
    ]


def test_a_single_result_needs_no_legend(problem, results):
    fig = plot_success_vs_evaluations(problem, results[:1])
    assert fig.legends == []


@pytest.mark.parametrize("name", ["light", "dark"])
def test_the_figure_uses_the_theme_surface(problem, results, name):
    fig = plot_success_vs_evaluations(problem, results, theme=name)
    assert mcolors.same_color(fig.get_facecolor(), get_theme(name).surface)


def test_two_results_of_one_algorithm_are_rejected(problem):
    twice = [
        ("a", run(problem, "qaoa", 0.4, 10, 20)),
        ("b", run(problem, "qaoa", 0.5, 10, 20)),
    ]
    with pytest.raises(ValueError, match="once"):
        plot_success_vs_evaluations(problem, twice)


def test_a_result_without_a_draw_count_is_rejected(problem):
    uncounted = [("x", run(problem, "random", 0.1, None, 20))]
    with pytest.raises(ValueError, match="shots"):
        plot_success_vs_evaluations(problem, uncounted)
