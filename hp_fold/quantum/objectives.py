from typing import Protocol

import numpy as np

from hp_fold.folding.problem import HPProblem


class Circuit(Protocol):
    @property
    def problem(self) -> HPProblem: ...

    @property
    def n_params(self) -> int: ...

    def probabilities(self, params: np.ndarray) -> np.ndarray: ...


def cvar(probs: np.ndarray, cost: np.ndarray, alpha: float) -> float:
    """Exact CVaR_alpha of the distribution `probs` over `cost`."""
    _check_alpha(alpha)
    return _cvar_sorted(probs, cost, np.argsort(cost, kind="stable"), alpha)


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")


def _cvar_sorted(
    probs: np.ndarray, cost: np.ndarray, order: np.ndarray, alpha: float
) -> float:
    p, c = probs[order], cost[order]
    cumulative = np.cumsum(p)
    taken = np.clip(alpha - (cumulative - p), 0.0, p)
    return float(taken @ c / alpha)


class ExpectationObjective:
    """<C> of the circuit's output distribution."""

    def __init__(self, circuit: Circuit) -> None:
        self._circuit = circuit
        self._cost = circuit.problem.cost

    @property
    def n_params(self) -> int:
        return self._circuit.n_params

    def __call__(self, params: np.ndarray) -> float:
        return float(self._circuit.probabilities(params) @ self._cost)


class CVaRObjective:
    """CVaR_alpha of the circuit's output distribution."""

    def __init__(self, circuit: Circuit, alpha: float) -> None:
        _check_alpha(alpha)
        self._circuit = circuit
        self._alpha = alpha
        self._cost = circuit.problem.cost
        self._order = np.argsort(self._cost, kind="stable")  # fixed: sort once

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def n_params(self) -> int:
        return self._circuit.n_params

    def __call__(self, params: np.ndarray) -> float:
        probs = self._circuit.probabilities(params)
        return _cvar_sorted(probs, self._cost, self._order, self._alpha)
