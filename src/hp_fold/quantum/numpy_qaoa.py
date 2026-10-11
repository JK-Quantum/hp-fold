import numpy as np

from hp_fold.folding.problem import HPProblem


class NumpyQAOA:
    """Depth-p QAOA for one HPProblem, simulated with a numpy statevector."""

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
            f"NumpyQAOA(seq={self._problem.seq!r}, p={self._p}, "
            f"n_qubits={self.n_qubits}, center_cost={self._center_cost})"
        )

    @property
    def problem(self) -> HPProblem:
        return self._problem

    @property
    def p(self) -> int:
        return self._p

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

    def statevector(self, params: np.ndarray) -> np.ndarray:
        gammas, betas = self.split_params(params)
        dim = self._problem.cost.size
        psi = np.full(dim, dim**-0.5, dtype=complex)
        for gamma, beta in zip(gammas, betas, strict=True):
            psi *= np.exp(-1j * gamma * self._phase_cost)
            psi = self._mix(psi, beta)
        return psi

    def probabilities(self, params: np.ndarray) -> np.ndarray:
        probs = np.abs(self.statevector(params)) ** 2
        return probs / probs.sum()

    def expectation(self, params: np.ndarray) -> float:
        return float(self.probabilities(params) @ self._problem.cost)

    def sample(
        self, params: np.ndarray, shots: int, rng: np.random.Generator
    ) -> np.ndarray:
        if shots < 1:
            raise ValueError(f"shots must be >= 1, got {shots}")
        return rng.choice(
            self._problem.cost.size, size=shots, p=self.probabilities(params)
        )

    def _mix(self, psi: np.ndarray, beta: float) -> np.ndarray:
        """apply RX(2*beta) to every qubit."""
        c, s = np.cos(beta), -1j * np.sin(beta)
        rx = np.array([[c, s], [s, c]])
        n = self.n_qubits
        psi = psi.reshape((2,) * n)
        for axis in range(n):
            psi = np.moveaxis(np.tensordot(rx, psi, axes=([1], [axis])), 0, axis)
        return psi.reshape(-1)
