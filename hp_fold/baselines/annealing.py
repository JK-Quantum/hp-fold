import math

import numpy as np

from hp_fold.folding.problem import HPProblem


def metropolis_accept(delta: float, temperature: float, u: float) -> bool:
    """Accept downhill moves always, uphill moves with probability exp(-delta / T)."""
    return delta <= 0 or u < math.exp(-delta / temperature)


class SimulatedAnnealing:
    """Single-bit-flip annealing on the tabulated cost, geometric cooling.

    Each sample is the best state seen during one independent run.
    Defaults are untuned guesses: valid-fold energy gaps are 1, so
    T = 1 is "warm" and T = 0.05 is "nearly greedy".
    """

    def __init__(
        self,
        n_steps: int = 200,
        t_start: float = 1.0,
        t_end: float = 0.05,
    ) -> None:
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        if not 0 < t_end <= t_start:
            raise ValueError("temperatures must satisfy 0 < t_end <= t_start")
        self._n_steps = n_steps
        self._t_start = t_start
        self._t_end = t_end

    @property
    def evaluations_per_sample(self) -> int:
        return self._n_steps + 1

    def sample(
        self,
        problem: HPProblem,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        runs = [self._run(problem, rng) for _ in range(n_samples)]
        return np.array(runs, dtype=np.int64)

    def _run(self, problem: HPProblem, rng: np.random.Generator) -> int:
        n = problem.n_qubits
        x = int(rng.integers(2**n))
        energy = problem.cost[x]
        best_x, best_energy = x, energy
        for temperature in np.geomspace(self._t_start, self._t_end, self._n_steps):
            candidate = x ^ (1 << int(rng.integers(n)))
            candidate_energy = problem.cost[candidate]
            if metropolis_accept(candidate_energy - energy, temperature, rng.random()):
                x, energy = candidate, candidate_energy
                if energy < best_energy:
                    best_x, best_energy = x, energy
        return best_x
