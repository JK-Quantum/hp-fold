"""Scalar metrics of a RunResult against its problem."""

import math
from dataclasses import dataclass

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


def compute_metrics(problem: HPProblem, result: RunResult) -> Metrics:
    if result.seq != problem.seq:
        raise ValueError(f"result is for {result.seq!r}, problem is {problem.seq!r}")
    if result.probs.size != problem.cost.size:
        raise ValueError(
            f"result has {result.n_qubits} qubits, problem has {problem.n_qubits}"
        )
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
        p_ground=float(probs[problem.ground_states].sum()),
        p_valid=p_valid,
        expected_energy=float(probs @ problem.cost),
        best_energy=float(problem.cost[probs > _SUPPORT_TOL].min()),
        mean_valid_energy=mean_valid,
        approx_ratio=ratio,
    )
