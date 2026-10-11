import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import Statevector

from hp_fold.folding.problem import HPProblem
from hp_fold.quantum.pauli import cost_to_pauli


class QAOACircuit:
    """A depth-p QAOA for one HPProblem, reusable across parameter values."""

    def __init__(self, problem: HPProblem, p: int, *, center_cost: bool = True) -> None:
        if p < 0:
            raise ValueError(f"depth p must be >= 0, got {p}")
        self._problem = problem
        self._p = p
        self._center_cost = center_cost
        self._offset = float(problem.cost.mean()) if center_cost else 0.0
        self._ansatz = self._build_ansatz(problem.cost - self._offset)
        by_name = {param.name: param for param in self._ansatz.parameters}
        self._gammas = [by_name[f"γ[{i}]"] for i in range(p)]
        self._betas = [by_name[f"β[{i}]"] for i in range(p)]

    def _build_ansatz(self, phase_cost: np.ndarray) -> QuantumCircuit:
        if self._p == 0:
            circuit = QuantumCircuit(self.n_qubits)
            circuit.h(range(self.n_qubits))
            return circuit
        return QAOAAnsatz(cost_to_pauli(phase_cost), reps=self._p)

    def __repr__(self) -> str:
        return (
            f"QAOACircuit(seq={self._problem.seq!r}, p={self._p}, "
            f"n_qubits={self.n_qubits}, center_cost={self._center_cost})"
        )

    @property
    def problem(self) -> HPProblem:
        return self._problem

    @property
    def n_qubits(self) -> int:
        return self._problem.n_qubits

    @property
    def n_params(self) -> int:
        return 2 * self._p

    @property
    def ansatz(self) -> QuantumCircuit:
        """The unbound parameterized circuit."""
        return self._ansatz

    def split_params(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        params = np.asarray(params, dtype=float)
        if params.shape != (self.n_params,):
            raise ValueError(
                f"expected {self.n_params} parameters, got shape {params.shape}"
            )
        return params[: self._p], params[self._p :]

    def build_qaoa_circuit(self, params: np.ndarray) -> QuantumCircuit:
        """The circuit with angles bound (no measurements)."""
        gammas, betas = self.split_params(params)
        binding = dict(zip(self._gammas, gammas, strict=True))
        binding.update(zip(self._betas, betas, strict=True))
        return self._ansatz.assign_parameters(binding)

    def measured_circuit(self, params: np.ndarray) -> QuantumCircuit:
        """The bound circuit with all qubits measured, ready to run on a backend."""
        circuit = self.build_qaoa_circuit(params).copy()
        circuit.measure_all()
        return circuit

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
