import numpy as np
import pytest

from hp_fold.folding.problem import HPProblem
from hp_fold.quantum.numpy_qaoa import NumpyQAOA
from hp_fold.quantum.objectives import CVaRObjective
from hp_fold.quantum.optimize import minimize_with_restarts


def bowl(x: np.ndarray) -> float:
    return float(((x - np.array([1.0, 2.0])) ** 2).sum())


def test_finds_the_minimum_of_a_simple_function():
    result = minimize_with_restarts(bowl, 2, restarts=3, rng=np.random.default_rng(0))
    assert result.value == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(result.params, [1.0, 2.0], atol=1e-3)
    assert len(result.restart_values) == 3
    assert result.n_evaluations >= len(result.history)


def test_is_reproducible_for_a_seed():
    a = minimize_with_restarts(bowl, 2, restarts=2, rng=np.random.default_rng(5))
    b = minimize_with_restarts(bowl, 2, restarts=2, rng=np.random.default_rng(5))
    assert np.array_equal(a.params, b.params)


def test_history_is_one_best_so_far_curve_over_every_evaluation():
    result = minimize_with_restarts(bowl, 2, restarts=3, rng=np.random.default_rng(1))
    assert len(result.history) == result.n_evaluations
    assert result.history[-1] == pytest.approx(result.value)
    pairs = zip(result.history, result.history[1:], strict=False)
    assert all(a >= b for a, b in pairs)


def test_extra_starts_are_used():
    result = minimize_with_restarts(
        bowl, 2, restarts=0, rng=np.random.default_rng(0), extra_starts=[np.ones(2)]
    )
    assert result.value == pytest.approx(0.0, abs=1e-6)
    assert len(result.restart_values) == 1


def test_needs_at_least_one_start():
    with pytest.raises(ValueError, match="start"):
        minimize_with_restarts(bowl, 2, restarts=0, rng=np.random.default_rng(0))


def test_cvar_qaoa_end_to_end_beats_uniform_and_finds_ground_mass():
    problem = HPProblem.from_sequence("PHPPH")
    circuit = NumpyQAOA(problem, 2)
    objective = CVaRObjective(circuit, alpha=0.1)
    result = minimize_with_restarts(objective, restarts=4, rng=np.random.default_rng(0))
    probs = circuit.probabilities(result.params)
    assert result.value >= problem.ground_energy - 1e-9
    assert result.value < problem.cost.mean()
    assert (
        probs[problem.ground_states].sum()
        > problem.ground_states.size / problem.cost.size
    )


def test_n_params_comes_from_the_objective_when_available():
    circuit = NumpyQAOA(HPProblem.from_sequence("PHPPH"), 3)
    objective = CVaRObjective(circuit, alpha=0.25)
    assert objective.n_params == circuit.n_params == 6
    result = minimize_with_restarts(
        objective, restarts=1, rng=np.random.default_rng(0), maxiter=20
    )
    assert result.params.shape == (6,)


def test_n_params_is_required_for_a_plain_function():
    with pytest.raises(ValueError, match="n_params"):
        minimize_with_restarts(bowl, restarts=1, rng=np.random.default_rng(0))
