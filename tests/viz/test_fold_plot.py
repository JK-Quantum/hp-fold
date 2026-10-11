import numpy as np
import pytest

from hp_fold import HPProblem
from hp_fold.viz.fold import plot_fold


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_the_drawn_chain_is_the_decoded_fold_and_contacts_match_the_energy(theme):
    problem = HPProblem.from_sequence("HPPH")
    x = int(problem.ground_states[0])
    ax = plot_fold(problem, x, theme).axes[0]
    assert np.array_equal(ax.lines[0].get_xydata(), problem.decode(x))
    assert len(ax.lines) - 1 == -problem.ground_energy == 1
    assert len(ax.collections) == problem.n_residues
