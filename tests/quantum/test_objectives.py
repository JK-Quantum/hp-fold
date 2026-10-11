import numpy as np
import pytest

from hp_fold.folding.problem import HPProblem
from hp_fold.quantum.numpy_qaoa import NumpyQAOA
from hp_fold.quantum.objectives import CVaRObjective, ExpectationObjective, cvar


def test_cvar_hand_computed_cases():
    probs, cost = np.array([0.5, 0.5]), np.array([0.0, 2.0])
    assert cvar(probs, cost, 0.25) == pytest.approx(0.0)
    assert cvar(probs, cost, 0.5) == pytest.approx(0.0)
    assert cvar(probs, cost, 0.75) == pytest.approx((0.5 * 0 + 0.25 * 2) / 0.75)
    assert cvar(probs, cost, 1.0) == pytest.approx(1.0)


def test_cvar_ignores_the_input_order_of_folds():
    probs, cost = np.array([0.2, 0.5, 0.3]), np.array([3.0, -1.0, 2.0])
    perm = np.array([2, 0, 1])
    # The cost -1 fold alone holds 0.5 >= 0.4 of the mass.
    assert cvar(probs, cost, 0.4) == pytest.approx(-1.0)
    assert cvar(probs[perm], cost[perm], 0.4) == pytest.approx(-1.0)


def test_cvar_alpha_one_equals_expectation_and_is_monotone_in_alpha():
    rng = np.random.default_rng(0)
    n = 2**6
    probs = rng.dirichlet(np.ones(n))
    cost = rng.normal(size=n)
    assert cvar(probs, cost, 1.0) == pytest.approx(probs @ cost)
    values = [cvar(probs, cost, a) for a in (0.05, 0.1, 0.25, 0.5, 1.0)]
    assert values == sorted(values)
    assert values[0] >= cost.min()


def test_cvar_of_a_state_on_the_ground_fold_is_the_ground_energy():
    cost = np.array([1.0, -3.0, 2.0])
    assert cvar(np.array([0.0, 1.0, 0.0]), cost, 0.05) == pytest.approx(-3.0)


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.01])
def test_cvar_rejects_bad_alpha(alpha):
    circuit = NumpyQAOA(HPProblem.from_sequence("PHPPH"), 1)
    with pytest.raises(ValueError, match="alpha"):
        cvar(np.array([1.0]), np.array([0.0]), alpha)
    with pytest.raises(ValueError, match="alpha"):
        CVaRObjective(circuit, alpha)


def test_objectives_on_a_real_circuit():
    problem = HPProblem.from_sequence("PHPPH")
    circuit = NumpyQAOA(problem, 1)
    params = np.array([0.4, 0.3])
    expectation = ExpectationObjective(circuit)(params)
    assert expectation == pytest.approx(circuit.expectation(params))
    full = CVaRObjective(circuit, 1.0)(params)
    assert full == pytest.approx(expectation)
    small = CVaRObjective(circuit, 0.1)(params)
    assert problem.ground_energy <= small <= expectation
