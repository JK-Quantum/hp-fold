from functools import reduce

import numpy as np
import pytest
from scipy.optimize import minimize

from hp_fold.folding.problem import HPProblem
from hp_fold.quantum.qaoa import QAOACircuit

N_QUBITS = 6
DIM = 2**N_QUBITS


@pytest.fixture(scope="module")
def problem() -> HPProblem:
    return HPProblem.from_sequence("PHPPH")


def dense_reference_probabilities(
    cost: np.ndarray, gammas: list[float], betas: list[float]
) -> np.ndarray:
    """Independent oracle: full 64x64 matrices, no Qiskit, no tricks."""
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    identity = np.eye(2, dtype=complex)
    mixer_generator = sum(
        reduce(np.kron, [x if q == k else identity for q in range(N_QUBITS)])
        for k in range(N_QUBITS)
    )
    eigvals, eigvecs = np.linalg.eigh(mixer_generator)
    psi = np.full(DIM, DIM**-0.5, dtype=complex)
    for gamma, beta in zip(gammas, betas, strict=True):
        psi = np.exp(-1j * gamma * cost) * psi
        mixer = eigvecs @ np.diag(np.exp(-1j * beta * eigvals)) @ eigvecs.conj().T
        psi = mixer @ psi
    return np.abs(psi) ** 2


def test_depth_zero_is_uniform_with_mean_cost(problem):
    circuit = QAOACircuit(problem, p=0)
    assert np.allclose(circuit.probabilities(np.array([])), 1 / DIM)
    assert circuit.expectation(np.array([])) == pytest.approx(problem.cost.mean())


def test_probabilities_are_normalized(problem):
    circuit = QAOACircuit(problem, p=3)
    params = np.random.default_rng(0).uniform(-1, 1, circuit.n_params)
    assert circuit.probabilities(params).sum() == pytest.approx(1.0)


def test_matches_independent_dense_numpy_reference(problem):
    gammas, betas = [0.37, -0.81], [0.22, 0.54]
    actual = QAOACircuit(problem, p=2).probabilities(np.array(gammas + betas))
    expected = dense_reference_probabilities(problem.cost, gammas, betas)
    assert np.allclose(actual, expected)


def test_cost_layer_phases_land_on_the_right_fold_index(problem):
    # beta = 0 makes the mixer the identity, so the state is the raw phase vector.
    circuit = QAOACircuit(problem, p=1, center_cost=False)
    psi = circuit.statevector(np.array([0.5, 0.0]))
    assert np.allclose(psi, np.exp(-1j * 0.5 * problem.cost) / 8)


def test_centering_does_not_change_probabilities(problem):
    params = np.array([0.37, -0.81, 0.22, 0.54])
    centered = QAOACircuit(problem, p=2, center_cost=True)
    raw = QAOACircuit(problem, p=2, center_cost=False)
    assert np.allclose(centered.probabilities(params), raw.probabilities(params))
    assert centered.expectation(params) == pytest.approx(raw.expectation(params))


def test_mixer_alone_leaves_uniform_state_invariant(problem):
    probs = QAOACircuit(problem, p=1).probabilities(np.array([0.0, 0.3]))
    assert np.allclose(probs, 1 / DIM)


def test_extra_layer_with_zero_angles_changes_nothing(problem):
    p1 = QAOACircuit(problem, p=1).probabilities(np.array([0.4, 0.3]))
    p2 = QAOACircuit(problem, p=2).probabilities(np.array([0.4, 0.0, 0.3, 0.0]))
    assert np.allclose(p1, p2)


def test_probabilities_are_symmetric_under_sign_flip_of_all_angles(problem):
    params = np.array([0.4, 0.9, 0.3, 0.7])
    circuit = QAOACircuit(problem, p=2)
    assert np.allclose(circuit.probabilities(params), circuit.probabilities(-params))


def test_beta_has_period_pi(problem):
    circuit = QAOACircuit(problem, p=1)
    a = circuit.probabilities(np.array([0.4, 0.3]))
    b = circuit.probabilities(np.array([0.4, 0.3 + np.pi]))
    assert np.allclose(a, b)


def test_optimized_qaoa_beats_uniform_and_never_goes_below_ground(problem):
    circuit = QAOACircuit(problem, p=2)
    x0 = np.random.default_rng(0).uniform(0, np.pi, circuit.n_params)
    best = minimize(circuit.expectation, x0, method="COBYLA").fun
    assert problem.ground_energy <= best < problem.cost.mean()


def test_sample_is_reproducible_and_in_range(problem):
    circuit = QAOACircuit(problem, p=1)
    params = np.array([0.4, 0.3])
    a = circuit.sample(params, 100, np.random.default_rng(1))
    b = circuit.sample(params, 100, np.random.default_rng(1))
    assert np.array_equal(a, b)
    assert a.min() >= 0 and a.max() < DIM


def test_invalid_inputs(problem):
    with pytest.raises(ValueError, match="depth"):
        QAOACircuit(problem, p=-1)
    with pytest.raises(ValueError, match="parameters"):
        QAOACircuit(problem, p=2).statevector(np.zeros(3))
