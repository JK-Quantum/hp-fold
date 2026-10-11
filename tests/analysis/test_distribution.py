import numpy as np
import pytest

from hp_fold import HPProblem
from hp_fold.analysis.distribution import energy_distribution
from hp_fold.analysis.metrics import compute_metrics
from hp_fold.analysis.result import RunResult

GROUND = 19  # RDRU, a ground state of PHPPH (E = -1)
STRAIGHT = 0  # RRRR: valid, no contacts (E = 0)


@pytest.fixture(scope="module")
def problem():
    return HPProblem.from_sequence("PHPPH")


@pytest.fixture(scope="module")
def invalid_x(problem):
    return int(np.flatnonzero(~problem.valid)[0])


def uniform(problem) -> RunResult:
    size = 2**problem.n_qubits
    return RunResult("uniform", problem.seq, np.full(size, 1 / size), None, 1)


def mass_on(problem, x: int) -> RunResult:
    return RunResult.from_samples("point", problem, [x], evaluations=1)


def test_energies_are_the_distinct_fold_energies_in_ascending_order(problem):
    dist = energy_distribution(problem, uniform(problem))
    assert dist.energies.tolist() == sorted(set(problem.cost.tolist()))
    assert dist.energies[0] == -1.0


def test_all_probability_is_accounted_for(problem):
    dist = energy_distribution(problem, uniform(problem))
    total = dist.p_ground.sum() + dist.p_valid.sum() + dist.p_invalid.sum()
    assert total == pytest.approx(1.0)


def test_class_totals_agree_with_the_metrics(problem):
    result = uniform(problem)
    dist, metrics = (
        energy_distribution(problem, result),
        compute_metrics(problem, result),
    )
    assert dist.p_ground.sum() == pytest.approx(metrics.p_ground)
    assert (dist.p_ground + dist.p_valid).sum() == pytest.approx(metrics.p_valid)


def test_ground_state_mass_sits_at_the_ground_energy(problem):
    dist = energy_distribution(problem, mass_on(problem, GROUND))
    assert dist.p_ground[0] == pytest.approx(1.0)
    assert dist.p_valid.sum() == 0 and dist.p_invalid.sum() == 0


def test_valid_folds_without_contacts_are_not_counted_as_ground(problem):
    dist = energy_distribution(problem, mass_on(problem, STRAIGHT))
    level = int(np.flatnonzero(dist.energies == 0.0)[0])
    assert dist.p_valid[level] == pytest.approx(1.0)
    assert dist.p_ground.sum() == 0


def test_overlapping_folds_land_at_their_own_energy_in_the_invalid_class(
    problem, invalid_x
):
    dist = energy_distribution(problem, mass_on(problem, invalid_x))
    level = int(np.flatnonzero(dist.energies == problem.cost[invalid_x])[0])
    assert dist.p_invalid[level] == pytest.approx(1.0)
    assert dist.p_ground.sum() == 0 and dist.p_valid.sum() == 0


def test_uniform_distribution_spreads_mass_in_proportion_to_fold_counts(problem):
    dist = energy_distribution(problem, uniform(problem))
    assert dist.p_ground[0] == pytest.approx(4 / 64)
    zero_level = int(np.flatnonzero(dist.energies == 0.0)[0])
    valid_zero = int(np.sum(problem.valid & (problem.cost == 0.0)))
    assert dist.p_valid[zero_level] == pytest.approx(valid_zero / 64)


def test_rows_are_a_plain_table_of_the_same_numbers(problem):
    dist = energy_distribution(problem, mass_on(problem, GROUND))
    rows = dist.rows()
    assert len(rows) == dist.energies.size
    assert rows[0] == {
        "energy": -1.0,
        "p_ground": 1.0,
        "p_valid": 0.0,
        "p_invalid": 0.0,
    }
    assert all(set(r) == {"energy", "p_ground", "p_valid", "p_invalid"} for r in rows)


def test_a_result_for_another_sequence_is_rejected(problem):
    other = HPProblem.from_sequence("HHPHH")
    result = RunResult.from_samples("test", other, [0], evaluations=1)
    with pytest.raises(ValueError, match="problem"):
        energy_distribution(problem, result)
