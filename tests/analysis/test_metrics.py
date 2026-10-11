import math

import numpy as np
import pytest

from hp_fold import HPProblem
from hp_fold.analysis.metrics import compute_metrics, success_vs_shots
from hp_fold.analysis.result import RunResult

GROUND = 19  # RDRU, a ground state of PHPPH (E = -1)
STRAIGHT = 0  # RRRR: valid, no contacts (E = 0)


@pytest.fixture(scope="module")
def problem():
    return HPProblem.from_sequence("PHPPH")


@pytest.fixture(scope="module")
def invalid_x(problem):
    return int(np.flatnonzero(~problem.valid)[0])


def point_mass(problem, weights: dict[int, float]) -> RunResult:
    probs = np.zeros(2**problem.n_qubits)
    for x, w in weights.items():
        probs[x] = w
    return RunResult("test", problem.seq, probs, shots=None, evaluations=1)


def test_pinned_example_assumptions(problem, invalid_x):
    assert problem.fold(GROUND) == "RDRU" and problem.cost[GROUND] == -1
    assert problem.fold(STRAIGHT) == "RRRR" and problem.cost[STRAIGHT] == 0
    assert not problem.valid[invalid_x]


def test_all_mass_on_a_ground_state(problem):
    m = compute_metrics(problem, point_mass(problem, {GROUND: 1.0}))
    assert m.p_ground == pytest.approx(1.0)
    assert m.p_valid == pytest.approx(1.0)
    assert m.expected_energy == pytest.approx(-1.0)
    assert m.best_energy == -1.0
    assert m.mean_valid_energy == pytest.approx(-1.0)
    assert m.approx_ratio == pytest.approx(1.0)


def test_valid_fold_without_contacts_has_ratio_zero(problem):
    m = compute_metrics(problem, point_mass(problem, {STRAIGHT: 1.0}))
    assert (m.p_ground, m.p_valid) == (0.0, 1.0)
    assert m.best_energy == 0.0
    assert m.approx_ratio == pytest.approx(0.0)


def test_invalid_fold_only(problem, invalid_x):
    m = compute_metrics(problem, point_mass(problem, {invalid_x: 1.0}))
    assert (m.p_ground, m.p_valid) == (0.0, 0.0)
    assert m.expected_energy == pytest.approx(problem.cost[invalid_x])
    assert math.isnan(m.mean_valid_energy)
    assert math.isnan(m.approx_ratio)


def test_ratio_ignores_invalid_mass(problem, invalid_x):
    m = compute_metrics(problem, point_mass(problem, {GROUND: 0.5, invalid_x: 0.5}))
    assert m.p_ground == pytest.approx(0.5)
    assert m.p_valid == pytest.approx(0.5)
    assert m.expected_energy == pytest.approx(0.5 * -1 + 0.5 * problem.cost[invalid_x])
    assert m.mean_valid_energy == pytest.approx(-1.0)
    assert m.approx_ratio == pytest.approx(1.0)
    assert m.best_energy == -1.0


def test_uniform_distribution_matches_counting(problem):
    uniform = np.full(2**problem.n_qubits, 1 / 2**problem.n_qubits)
    result = RunResult("test", problem.seq, uniform, shots=None, evaluations=1)
    m = compute_metrics(problem, result)
    assert m.p_ground == pytest.approx(4 / 64)
    assert m.p_valid == pytest.approx(25 / 64)
    assert m.expected_energy == pytest.approx(problem.cost.mean())
    assert m.mean_valid_energy == pytest.approx(problem.cost[problem.valid].mean())
    assert m.approx_ratio == pytest.approx(m.mean_valid_energy / -1.0)
    assert m.best_energy == -1.0


def test_ratio_is_nan_when_ground_energy_is_zero():
    flat = HPProblem.from_sequence("PPPPP")
    assert flat.ground_energy == 0
    result = RunResult.from_samples(
        "test",
        flat,
        [int(flat.ground_states[0])],
        evaluations=1,
    )
    assert math.isnan(compute_metrics(flat, result).approx_ratio)


def test_rejects_result_for_another_sequence(problem):
    other = HPProblem.from_sequence("HHPHH")
    result = RunResult.from_samples("test", other, [0], evaluations=1)
    with pytest.raises(ValueError, match="problem"):
        compute_metrics(problem, result)


def test_rejects_result_with_wrong_qubit_count(problem):
    probs = np.full(8, 1 / 8)
    result = RunResult("test", problem.seq, probs, shots=None, evaluations=1)
    with pytest.raises(ValueError, match="qubits"):
        compute_metrics(problem, result)


def test_success_vs_shots_matches_closed_form(problem):
    result = point_mass(problem, {GROUND: 0.5, STRAIGHT: 0.5})
    got = success_vs_shots(problem, result, [1, 2, 3])
    assert got == pytest.approx([0.5, 0.75, 0.875])


def test_success_vs_shots_agrees_with_simulation(problem):
    uniform = np.full(2**problem.n_qubits, 1 / 2**problem.n_qubits)
    result = RunResult("test", problem.seq, uniform, shots=None, evaluations=1)
    rng = np.random.default_rng(0)
    draws = rng.integers(0, 64, size=(4000, 5))
    hit = np.isin(draws, problem.ground_states).any(axis=1).mean()
    assert success_vs_shots(problem, result, [5])[0] == pytest.approx(hit, abs=0.03)


def test_success_vs_shots_is_monotone_and_approaches_one(problem):
    uniform = np.full(2**problem.n_qubits, 1 / 2**problem.n_qubits)
    result = RunResult("test", problem.seq, uniform, shots=None, evaluations=1)
    curve = success_vs_shots(problem, result, np.arange(1, 1001))
    assert (np.diff(curve) >= 0).all()
    assert curve[0] == pytest.approx(4 / 64)
    assert curve[-1] > 0.999


def test_success_vs_shots_extremes(problem):
    always = point_mass(problem, {GROUND: 1.0})
    never = point_mass(problem, {STRAIGHT: 1.0})
    assert (success_vs_shots(problem, always, [1, 10]) == 1.0).all()
    assert (success_vs_shots(problem, never, [1, 10]) == 0.0).all()


@pytest.mark.parametrize("bad", [[0, 1], [-3], [1.5]])
def test_success_vs_shots_rejects_non_positive_or_fractional_shots(problem, bad):
    with pytest.raises(ValueError, match="shots"):
        success_vs_shots(problem, point_mass(problem, {GROUND: 1.0}), bad)


def test_success_vs_shots_rejects_incompatible_result(problem):
    other = HPProblem.from_sequence("HHPHH")
    result = RunResult.from_samples("test", other, [0], evaluations=1)
    with pytest.raises(ValueError, match="problem"):
        success_vs_shots(problem, result, [1])
