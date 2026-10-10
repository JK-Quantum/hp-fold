import numpy as np
from hp_fold.folding.problem import HPProblem

class QAOACircuit:
    """
    qaoa
    """
    def __init__(self, problem: HPProblem, p, center_cost) -> None:
        if p < 0:
            raise ValueError(f'deth p must be > 0, got {p}')
        self._problem = problem
        self._p = p
        self._center_cost = center_cost

    def __repr__(self) -> str:
        return (
            f"QAOACircuit(seq={self._problem.seq!r}, p={self._p}, "
            f"n_qubits={self.n_qubits}, center_cost={self._center_cost})"
        )

    @property
    def problem(self) -> HPProblem:
        return self._problem

    @property
    def p(self) -> HPProblem:
        return self._p

    @property
    def n_qubits(self) -> int:
        return self._problem.n_qubits