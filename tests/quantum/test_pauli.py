import numpy as np
import pytest

from hp_fold.folding.problem import HPProblem
from hp_fold.quantum.pauli import cost_to_pauli, pauli_stats, walsh_hadamard


def test_walsh_hadamard_matches_the_definition():
    values = np.random.default_rng(0).normal(size=16)
    expected = np.array(
        [
            sum(values[x] * (-1) ** bin(x & s).count("1") for x in range(16))
            for s in range(16)
        ]
    )
    assert np.allclose(walsh_hadamard(values), expected)


def test_walsh_hadamard_rejects_non_power_of_two():
    with pytest.raises(ValueError, match="power of two"):
        walsh_hadamard(np.zeros(6))


def test_expansion_diagonal_equals_cost_for_the_worked_example():
    problem = HPProblem.from_sequence("PHPPH")
    op = cost_to_pauli(problem.cost)
    assert np.allclose(np.real(op.to_matrix().diagonal()), problem.cost)


def test_expansion_is_exact_for_random_costs_of_several_sizes():
    rng = np.random.default_rng(1)
    for n_qubits in (1, 2, 3, 6):
        cost = rng.normal(size=2**n_qubits)
        op = cost_to_pauli(cost)
        assert np.allclose(np.real(op.to_matrix().diagonal()), cost)


def test_bit_order_is_little_endian():
    # cost = 1 exactly when bit 0 of x is set, i.e. (I - Z_0) / 2. Qiskit writes
    # qubit 0 as the rightmost character.
    op = cost_to_pauli(np.array([0.0, 1.0, 0.0, 1.0]))
    terms = dict(zip(op.paulis.to_labels(), np.real(op.coeffs), strict=True))
    assert terms == {"II": pytest.approx(0.5), "IZ": pytest.approx(-0.5)}


def test_pruning_drops_small_terms_with_bounded_error():
    problem = HPProblem.from_sequence("PHPPH")
    exact = cost_to_pauli(problem.cost, tol=0.0)
    pruned = cost_to_pauli(problem.cost, tol=0.05)
    assert len(pruned) < len(exact)
    error = np.abs(np.real(pruned.to_matrix().diagonal()) - problem.cost).max()
    assert error <= 0.05 * len(exact)  # each dropped term shifts a value by <= tol


def test_constant_cost_is_a_single_identity_term():
    op = cost_to_pauli(np.full(8, 2.5))
    assert op.paulis.to_labels() == ["III"]
    assert np.real(op.coeffs[0]) == pytest.approx(2.5)


def test_stats_counts_weights():
    problem = HPProblem.from_sequence("PHPPH")
    stats = pauli_stats(cost_to_pauli(problem.cost))
    assert stats.n_qubits == problem.n_qubits
    assert stats.n_terms == sum(stats.weight_histogram.values())
    assert 1 <= stats.max_weight <= problem.n_qubits
    assert stats.cnot_per_layer == sum(
        2 * (w - 1) * n for w, n in stats.weight_histogram.items() if w >= 2
    )
