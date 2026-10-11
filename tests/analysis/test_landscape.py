import numpy as np
import pytest

from hp_fold.analysis.landscape import Landscape

GAMMAS = np.array([0.0, 1.0, 2.0])
BETAS = np.array([0.0, 0.5])
VALUES = np.array([[3.0, 2.0, 1.0], [4.0, 5.0, 0.5]])


def test_the_best_point_is_reported_as_gamma_beta_value():
    assert Landscape(GAMMAS, BETAS, VALUES).best() == (2.0, 0.5, 0.5)


def test_rows_follow_beta_and_columns_follow_gamma():
    values = np.array([[9.0, 9.0, 9.0], [9.0, 0.0, 9.0]])
    assert Landscape(GAMMAS, BETAS, values).best() == (1.0, 0.5, 0.0)


def test_a_tie_for_the_lowest_value_goes_to_the_first_row_then_column():
    values = np.array([[2.0, 1.0, 1.0], [1.0, 1.0, 3.0]])
    assert Landscape(GAMMAS, BETAS, values).best() == (1.0, 0.0, 1.0)


def test_the_landscape_keeps_its_own_read_only_copy_of_the_data():
    values = VALUES.copy()
    landscape = Landscape(GAMMAS, BETAS, values)
    values[0, 0] = -100.0
    assert landscape.values[0, 0] == 3.0
    with pytest.raises(ValueError, match="read-only"):
        landscape.values[0, 0] = 0.0


def test_values_must_be_laid_out_as_betas_by_gammas():
    with pytest.raises(ValueError, match="shape"):
        Landscape(GAMMAS, BETAS, VALUES.T)


@pytest.mark.parametrize("bad", [[0.0, 0.0, 1.0], [2.0, 1.0, 0.0], [0.0, np.nan, 2.0]])
def test_angles_must_be_finite_and_strictly_increasing(bad):
    with pytest.raises(ValueError, match="gammas"):
        Landscape(np.array(bad), BETAS, VALUES)


def test_each_axis_needs_at_least_two_angles():
    with pytest.raises(ValueError, match="betas"):
        Landscape(GAMMAS, np.array([0.0]), VALUES[:1])


def test_angles_must_be_one_dimensional():
    with pytest.raises(ValueError, match="gammas"):
        Landscape(GAMMAS.reshape(1, 3), BETAS, VALUES)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_values_must_be_finite(bad):
    values = VALUES.copy()
    values[1, 1] = bad
    with pytest.raises(ValueError, match="finite"):
        Landscape(GAMMAS, BETAS, values)
