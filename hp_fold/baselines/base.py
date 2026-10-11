from typing import Protocol

import numpy as np

from hp_fold.folding.problem import HPProblem


class Sampler(Protocol):
    """Anything that proposes folds.

    Returns integers x (see `hp_fold.folding.lattice`).
    """

    @property
    def evaluations_per_sample(self) -> int:
        """Cost-function evaluations spent per returned sample, for fair comparisons."""
        ...

    def sample(
        self,
        problem: HPProblem,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray: ...
