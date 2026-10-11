"""Standard record of one algorithm run.

Calculates a distribution over fold integers x.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from hp_fold.folding.problem import HPProblem

_TOL = 1e-9


@dataclass(frozen=True, slots=True, eq=False)
class RunResult:
    """What an algorithm produced, in a form every metric understands.

    probs        dense distribution over x, shape (2**n_qubits,); probs[x] = P(x)
    shots        number of samples behind probs; None means exact probabilities
    evaluations  cost evaluations (classical) or circuit executions (quantum)
    config       inputs that define the run (p, alpha, optimizer, ...)
    extras       outputs that are not a distribution (optimizer history, depth, ...)
    """

    algorithm: str
    seq: str
    probs: np.ndarray
    shots: int | None
    evaluations: int
    seed: int | None = None
    config: Mapping[str, Any] = field(default_factory=dict)
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        probs = np.array(self.probs, dtype=float)
        if probs.ndim != 1 or probs.size < 2 or probs.size & (probs.size - 1):
            raise ValueError(
                f"probs length must be a power of two, got shape {probs.shape}"
            )
        if (probs < 0).any():
            raise ValueError("probs contains negative entries")
        if abs(probs.sum() - 1.0) > _TOL:
            raise ValueError(f"probs must sum to 1, sum is {probs.sum():.12g}")
        if self.shots is not None and self.shots < 1:
            raise ValueError(f"shots must be positive or None, got {self.shots}")
        probs.flags.writeable = False
        object.__setattr__(self, "probs", probs)

    @property
    def n_qubits(self) -> int:
        return self.probs.size.bit_length() - 1

    @classmethod
    def from_samples(
        cls,
        algorithm: str,
        problem: HPProblem,
        xs: ArrayLike,
        evaluations: int,
        **kwargs: Any,
    ) -> RunResult:
        """Empirical distribution of sampled integers (SA runs, random draws, shots)."""
        samples = np.asarray(xs, dtype=np.int64).ravel()
        if samples.size == 0:
            raise ValueError("cannot build a result from empty samples")
        size = 2**problem.n_qubits
        if samples.min() < 0 or samples.max() >= size:
            raise ValueError(f"sample out of range for {problem.n_qubits} qubits")
        probs = np.bincount(samples, minlength=size) / samples.size
        return cls(
            algorithm,
            problem.seq,
            probs,
            int(samples.size),
            evaluations,
            **kwargs,
        )
