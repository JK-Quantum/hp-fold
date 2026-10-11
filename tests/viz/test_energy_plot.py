import matplotlib.colors as mcolors
import numpy as np
import pytest

from hp_fold import HPProblem
from hp_fold.analysis.distribution import energy_distribution
from hp_fold.analysis.metrics import compute_metrics
from hp_fold.analysis.result import RunResult
from hp_fold.viz.energy import (
    GROUND_LABEL,
    INVALID_LABEL,
    VALID_LABEL,
    plot_energy_distributions,
    save_figure,
)
from hp_fold.viz.theme import get_theme


@pytest.fixture(scope="module")
def problem():
    return HPProblem.from_sequence("PHPPH")


def uniform(problem, name="uniform") -> RunResult:
    size = 2**problem.n_qubits
    return RunResult(name, problem.seq, np.full(size, 1 / size), None, 1)


def ground_only(problem, name="ideal") -> RunResult:
    x = int(problem.ground_states[0])
    return RunResult.from_samples(name, problem, [x], evaluations=1)


@pytest.fixture(scope="module")
def results(problem):
    return [("random", uniform(problem)), ("ideal", ground_only(problem))]


def test_empty_results_are_rejected(problem):
    with pytest.raises(ValueError, match="at least one"):
        plot_energy_distributions(problem, [])


def test_one_panel_per_result(problem, results):
    fig = plot_energy_distributions(problem, results)
    assert len(fig.axes) == len(results)
    assert [ax.get_title(loc="left") for ax in fig.axes] == ["random", "ideal"]


def test_ground_bars_carry_the_ground_probability(problem, results):
    fig = plot_energy_distributions(problem, results)
    for ax, (_, result) in zip(fig.axes, results, strict=True):
        ground = next(c for c in ax.containers if c.get_label() == GROUND_LABEL)
        heights = [bar.get_height() for bar in ground]
        expected = energy_distribution(problem, result).p_ground
        assert heights == pytest.approx(expected.tolist())


def test_legend_names_the_three_classes_once(problem, results):
    fig = plot_energy_distributions(problem, results)
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert labels == [GROUND_LABEL, VALID_LABEL, INVALID_LABEL]


def test_panels_share_the_same_axes(problem, results):
    fig = plot_energy_distributions(problem, results)
    assert fig.axes[0].get_xticks().tolist() == fig.axes[1].get_xticks().tolist()
    assert fig.axes[0].get_xlim() == fig.axes[1].get_xlim()
    assert fig.axes[0].get_ylim() == fig.axes[1].get_ylim()


def test_each_panel_states_its_ground_state_probability(problem, results):
    fig = plot_energy_distributions(problem, results)
    for ax, (_, result) in zip(fig.axes, results, strict=True):
        expected = f"P(ground) {compute_metrics(problem, result).p_ground:.3f}"
        assert expected in [t.get_text() for t in ax.texts]


@pytest.mark.parametrize("name", ["light", "dark"])
def test_the_figure_uses_the_theme_surface_and_accent(problem, results, name):
    theme = get_theme(name)
    fig = plot_energy_distributions(problem, results, theme=name)
    assert mcolors.same_color(fig.get_facecolor(), theme.surface)
    ground = next(c for c in fig.axes[0].containers if c.get_label() == GROUND_LABEL)
    assert mcolors.same_color(ground[0].get_facecolor(), theme.accent)


def test_unknown_theme_is_rejected(problem, results):
    with pytest.raises(ValueError, match="theme"):
        plot_energy_distributions(problem, results, theme="sepia")


def test_save_figure_writes_a_png_and_creates_folders(problem, results, tmp_path):
    fig = plot_energy_distributions(problem, results)
    path = save_figure(fig, tmp_path / "figs" / "energy.png", dpi=50)
    assert path.read_bytes().startswith(b"\x89PNG")
