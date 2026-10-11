import numpy as np
import pytest

from hp_fold.analysis.landscape import Landscape
from hp_fold.viz.landscape import plot_landscape


@pytest.fixture(scope="module")
def landscape():
    gammas = np.linspace(0.0, 6.0, 7)
    betas = np.linspace(0.0, 3.0, 4)
    values = np.add.outer(betas, gammas) + 1.0
    return Landscape(gammas, betas, values)


def legend_labels(fig):
    return [text.get_text() for text in fig.legends[0].get_texts()]


def test_the_grid_minimum_is_marked_at_its_position_and_value(landscape):
    ax = plot_landscape(landscape).axes[0]
    (marker,) = ax.get_lines()
    assert np.asarray(marker.get_xdata()).tolist() == [0.0]
    assert np.asarray(marker.get_ydata()).tolist() == [0.0]
    assert "1.00" in str(marker.get_label())


def test_the_optimizers_result_is_marked_only_when_it_is_known(landscape):
    assert len(plot_landscape(landscape).axes[0].get_lines()) == 1
    fig = plot_landscape(landscape, found=(2.0, 1.0))
    found = fig.axes[0].get_lines()[1]
    assert np.asarray(found.get_xdata()).tolist() == [2.0]
    assert np.asarray(found.get_ydata()).tolist() == [1.0]
    assert legend_labels(fig)[1] == "optimizer's result"


def test_reference_energies_outside_the_plotted_range_are_left_out(landscape):
    fig = plot_landscape(landscape, {"inside": 5.0, "outside": 99.0})
    texts = [t.get_text() for t in fig.axes[1].texts]
    assert texts == ["inside"]
