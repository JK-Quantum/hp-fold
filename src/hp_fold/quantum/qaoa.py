import numpy as np
from hp_fold.folding.problem import HPProblem
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import DiagonalGate

class QAOACircuit:
    """A depth-p QAOA for one HPProblem, reusable across parameter values."""

    def __init__(self, problem: HPProblem, p: int, *, center_cost: bool = True) -> None:
        if p < 0:
            raise ValueError(f"depth p must be >= 0, got {p}")
        self._problem = problem
        self._p = p
        self._center_cost = center_cost
        self._offset = float(problem.cost.mean()) if center_cost else 0.0
        self._phase_cost = problem.cost - self._offset

    def __repr__(self) -> str:
        return (
            f"QAOACircuit(seq={self._problem.seq!r}, p={self._p}, "
            f"n_qubits={self.n_qubits}, center_cost={self._center_cost})"
        )

    @property
    def n_qubits(self) -> int:
        return self._problem.n_qubits

    @property
    def n_params(self) -> int:
        return 2 * self._p

    def split_params(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        params = np.asarray(params, dtype=float)
        if params.shape != (self.n_params,):
            raise ValueError(
                f"expected {self.n_params} parameters, got shape {params.shape}"
            )
        return params[: self._p], params[self._p :]

    def build_qaoa_circuit(self, params: np.ndarray) -> QuantumCircuit:
        gammas, betas = self.split_params(params)
        n = self.n_qubits
        qc = QuantumCircuit(n)

        qc.h(range(n))
        for gamma, beta in zip(gammas, betas, strict=True):
            diagonal_phases = np.exp(-1j * gamma * self._phase_cost)
            qc.append(DiagonalGate(diagonal_phases), range(n))
            for qubit in range(n):
                qc.rx(2 * beta, qubit)

        return qc

    def statevector(self, params: np.ndarray) -> np.ndarray:
        """statevector as a numpy array."""
        qc = self.build_qaoa_circuit(params)
        return Statevector.from_instruction(qc).data

    def probabilities(self, params: np.ndarray) -> np.ndarray:
        """probability of measuring each fold x."""
        probs = np.abs(self.statevector(params)) ** 2
        return probs / probs.sum()

    def expectation(self, params: np.ndarray) -> float:
        """expected cost."""
        probs = self.probabilities(params)
        return float(probs @ self._problem.cost)

    def sample(
        self, params: np.ndarray, shots: int, rng: np.random.Generator
    ) -> np.ndarray:
        """draw shots fold indices. Returns an int array of indices."""
        probs = self.probabilities(params)
        return rng.choice(probs.size, size=shots, p=probs)