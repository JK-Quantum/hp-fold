"""Where a run's probability mass sits, grouped by fold energy and fold class."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from hp_fold.analysis.metrics import check_compatible
from hp_fold.analysis.result import RunResult
from hp_fold.folding.problem import HPProblem


@dataclass(frozen=True, slots=True)
class EnergyDistribution:
    """Probability per distinct fold energy, split into three disjoint classes."""

    energies: np.ndarray
    p_ground: np.ndarray
    p_valid: np.ndarray
    p_invalid: np.ndarray

    def rows(self) -> list[dict[str, Any]]:
        """The same numbers as a table (the text twin of the plot)."""
        return [
            {
                "energy": float(e),
                "p_ground": float(g),
                "p_valid": float(v),
                "p_invalid": float(i),
            }
            for e, g, v, i in zip(
                self.energies,
                self.p_ground,
                self.p_valid,
                self.p_invalid,
                strict=True,
            )
        ]


def energy_distribution(problem: HPProblem, result: RunResult) -> EnergyDistribution:
    check_compatible(problem, result)
    energies, level = np.unique(problem.cost, return_inverse=True)
    is_ground = np.zeros(problem.cost.size, dtype=bool)
    is_ground[problem.ground_states] = True

    def mass(mask: np.ndarray) -> np.ndarray:
        return np.bincount(
            level[mask],
            weights=result.probs[mask],
            minlength=energies.size,
        )

    return EnergyDistribution(
        energies=energies,
        p_ground=mass(is_ground),
        p_valid=mass(problem.valid & ~is_ground),
        p_invalid=mass(~problem.valid),
    )
