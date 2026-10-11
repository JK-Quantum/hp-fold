"""Scalar metrics of a RunResult against its problem."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from hp_fold.analysis.result import RunResult
from hp_fold.folding.problem import HPProblem

_SUPPORT_TOL = 1e-12


@dataclass(frozen=True, slots=True)
class Metrics:
    p_ground: float
    p_valid: float
    expected_energy: float
    best_energy: float
    mean_valid_energy: float
    approx_ratio: float


def check_compatible(problem: HPProblem, result: RunResult) -> None:
    if result.seq != problem.seq:
        raise ValueError(f"result is for {result.seq!r}, problem is {problem.seq!r}")
    if result.probs.size != problem.cost.size:
        raise ValueError(
            f"result has {result.n_qubits} qubits, problem has {problem.n_qubits}"
        )


def _p_ground(problem: HPProblem, result: RunResult) -> float:
    return float(result.probs[problem.ground_states].sum())


def compute_metrics(problem: HPProblem, result: RunResult) -> Metrics:
    check_compatible(problem, result)
    probs = result.probs
    p_valid = float(probs[problem.valid].sum())
    if p_valid > 0:
        mean_valid = float(probs[problem.valid] @ problem.cost[problem.valid] / p_valid)
    else:
        mean_valid = math.nan
    ratio = (
        mean_valid / problem.ground_energy if problem.ground_energy != 0 else math.nan
    )
    return Metrics(
        p_ground=_p_ground(problem, result),
        p_valid=p_valid,
        expected_energy=float(probs @ problem.cost),
        best_energy=float(problem.cost[probs > _SUPPORT_TOL].min()),
        mean_valid_energy=mean_valid,
        approx_ratio=ratio,
    )


def success_vs_shots(
    problem: HPProblem,
    result: RunResult,
    shots: ArrayLike,
) -> np.ndarray:
    """P(at least one of k independent draws is a ground state), for each k in `shots`.

    Treats result.probs as the distribution of a single draw, so it is exact.
    """
    check_compatible(problem, result)
    k = np.asarray(shots)
    if k.size and (k.min() < 1 or not np.issubdtype(k.dtype, np.integer)):
        raise ValueError("shots must be positive integers")
    return _at_least_one(_p_ground(problem, result), k)


def success_vs_evaluations(
    problem: HPProblem,
    result: RunResult,
    budgets: ArrayLike,
) -> np.ndarray:
    """P(at least one ground state) when `budgets` cost-function evaluations are spent.

    Cost model: `result.extras["training_evaluations"]` (default 0) is paid once, before
    the distribution is fixed.
    The rest of `result.evaluations` is spread evenly over the `result.shots` draws.
    Only complete draws count, so the curve is a step function that is 0 until training
    and the first draw are paid for.
    """
    check_compatible(problem, result)
    if result.shots is None:
        raise ValueError(
            "success_vs_evaluations needs result.shots (the number of draws)"
        )
    training = float(result.extras.get("training_evaluations", 0))
    if not 0 <= training < result.evaluations:
        raise ValueError(
            f"extras['training_evaluations'] must be in [0, evaluations), "
            f"got {training:g} with {result.evaluations} evaluations"
        )
    budget = np.asarray(budgets, dtype=float)
    if budget.size and budget.min() < 0:
        raise ValueError("budgets must be non-negative")
    per_draw = (result.evaluations - training) / result.shots
    draws = np.floor(np.maximum(budget - training, 0.0) / per_draw)
    return _at_least_one(_p_ground(problem, result), draws)


def _at_least_one(p: float, draws: np.ndarray) -> np.ndarray:
    """1 - (1 - p)**draws, accurate for tiny p; exactly 0 for zero draws."""
    if p >= 1.0:
        return (draws >= 1).astype(float)
    return -np.expm1(draws * np.log1p(-p))
