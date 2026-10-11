"""An expectation-value landscape over the two QAOA angles (gamma, beta)
of a depth-1 circuit.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Landscape:
    """<C> sampled on a regular grid: values[i, j] belongs to betas[i] and gammas[j]."""

    gammas: np.ndarray
    betas: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        gammas = np.array(self.gammas, dtype=float)
        betas = np.array(self.betas, dtype=float)
        values = np.array(self.values, dtype=float)
        for name, axis in (("gammas", gammas), ("betas", betas)):
            if axis.ndim != 1 or axis.size < 2:
                raise ValueError(
                    f"{name} must be a 1-D sequence of at least two angles"
                )
            if not np.isfinite(axis).all() or (np.diff(axis) <= 0).any():
                raise ValueError(f"{name} must be finite and strictly increasing")
        expected = (betas.size, gammas.size)
        if values.shape != expected:
            raise ValueError(
                f"values must have shape (len(betas), len(gammas)) = {expected}, "
                f"got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("values must be finite")
        for array in (gammas, betas, values):
            array.setflags(write=False)
        object.__setattr__(self, "gammas", gammas)
        object.__setattr__(self, "betas", betas)
        object.__setattr__(self, "values", values)

    def best(self) -> tuple[float, float, float]:
        """(gamma, beta, value) of the lowest grid point.

        The first one in row order on ties.
        """
        i, j = np.unravel_index(int(np.argmin(self.values)), self.values.shape)
        return float(self.gammas[j]), float(self.betas[i]), float(self.values[i, j])
