import numpy as np

from hp_fold.folding.problem import HPProblem


class RandomSampler:
    """Uniform random folds.

    The floor every other method has to beat.
    """

    evaluations_per_sample = 1

    def sample(
        self,
        problem: HPProblem,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return rng.integers(0, 2**problem.n_qubits, size=n_samples)
