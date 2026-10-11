import numpy as np
import pytest

from hp_fold.folding.problem import HPProblem
from hp_fold.quantum.numpy_qaoa import NumpyQAOA
from hp_fold.quantum.qaoa import QAOACircuit


@pytest.fixture(scope="module")
def problem() -> HPProblem:
    return HPProblem.from_sequence("PHPPH")


@pytest.mark.parametrize("p", [0, 1, 2, 3])
def test_agrees_with_the_qiskit_circuit(problem, p):
    params = np.random.default_rng(p).uniform(-1.5, 1.5, 2 * p)
    numpy_probs = NumpyQAOA(problem, p).probabilities(params)
    qiskit_probs = QAOACircuit(problem, p).probabilities(params)
    assert np.allclose(numpy_probs, qiskit_probs)


def test_expectation_matches_qiskit_circuit(problem):
    params = np.array([0.4, -0.2, 0.7, 0.1])
    assert NumpyQAOA(problem, 2).expectation(params) == pytest.approx(
        QAOACircuit(problem, 2).expectation(params)
    )


def test_depth_zero_is_uniform(problem):
    qaoa = NumpyQAOA(problem, 0)
    assert np.allclose(qaoa.probabilities(np.array([])), 1 / problem.cost.size)
    assert qaoa.expectation(np.array([])) == pytest.approx(problem.cost.mean())


def test_centering_changes_only_the_global_phase(problem):
    params = np.array([0.37, -0.81, 0.22, 0.54])
    a = NumpyQAOA(problem, 2, center_cost=True).probabilities(params)
    b = NumpyQAOA(problem, 2, center_cost=False).probabilities(params)
    assert np.allclose(a, b)


def test_sample_is_reproducible_and_in_range(problem):
    qaoa = NumpyQAOA(problem, 1)
    a = qaoa.sample(np.array([0.4, 0.3]), 50, np.random.default_rng(1))
    b = qaoa.sample(np.array([0.4, 0.3]), 50, np.random.default_rng(1))
    assert np.array_equal(a, b)
    assert a.min() >= 0 and a.max() < problem.cost.size


def test_invalid_inputs(problem):
    with pytest.raises(ValueError, match="depth"):
        NumpyQAOA(problem, -1)
    with pytest.raises(ValueError, match="parameters"):
        NumpyQAOA(problem, 2).statevector(np.zeros(3))
    with pytest.raises(ValueError, match="shots"):
        NumpyQAOA(problem, 1).sample(np.zeros(2), 0, np.random.default_rng(0))
