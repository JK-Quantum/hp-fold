from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class OptimizationResult:
    params: np.ndarray
    value: float
    n_evaluations: int
    history: list[float] = field(repr=False)
    restart_values: list[float] = field(repr=False)


def minimize_with_restarts(
    objective: Callable[[np.ndarray], float],
    n_params: int | None = None,
    *,
    restarts: int,
    rng: np.random.Generator,
    maxiter: int = 300,
    extra_starts: Sequence[np.ndarray] = (),
    low: float = 0.0,
    high: float = np.pi,
) -> OptimizationResult:
    """Minimize objective with COBYLA from restarts random starts."""
    if n_params is None:
        n_params = getattr(objective, "n_params", None)
        if n_params is None:
            raise ValueError("pass n_params: the objective does not define it")
    if restarts < 0 or restarts + len(extra_starts) == 0:
        raise ValueError("need at least one start (restarts >= 1 or extra_starts)")
    starts = [rng.uniform(low, high, n_params) for _ in range(restarts)]
    starts += [np.asarray(x, dtype=float) for x in extra_starts]

    best: tuple[np.ndarray, float, list[float]] | None = None
    n_evaluations, restart_values = 0, []
    for x0 in starts:
        history: list[float] = []

        def tracked(x: np.ndarray, history: list[float] = history) -> float:
            value = objective(x)
            history.append(min(value, history[-1]) if history else value)
            return value

        result = minimize(tracked, x0, method="COBYLA", options={"maxiter": maxiter})
        n_evaluations += len(history)
        restart_values.append(float(result.fun))
        if best is None or result.fun < best[1]:
            best = (np.asarray(result.x), float(result.fun), history)

    assert best is not None
    return OptimizationResult(best[0], best[1], n_evaluations, best[2], restart_values)
