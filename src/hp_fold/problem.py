from dataclasses import dataclass

import numpy as np

from hp_fold import lattice
from hp_fold.energy import count_overlaps, energy


def default_penalty(n_residues: int) -> float:
    """Default overlap penalty: one unit per residue.

    Measured worst-case requirement (smallest penalty that keeps every optimum
    valid) is about N/4: exhaustive over all sequences for N <= 8, sampled up to
    N = 10. N therefore leaves a wide margin, and HPProblem.from_sequence still
    checks the result, so a too-small custom value fails loudly.
    """
    return float(n_residues)


def _read_only(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


@dataclass(frozen=True, slots=True, eq=False)
class HPProblem:
    """The HP folding problem for one sequence, tabulated over all 2**n_qubits folds.

    Index convention: `x` is the integer encoding of a fold (see hp_fold.lattice);
    qubit q is bit q of x. `cost[x]` is the energy of fold x including the overlap
    penalty, and `valid[x]` tells whether fold x is free of overlaps.
    """

    seq: str
    penalty: float
    cost: np.ndarray
    valid: np.ndarray
    ground_energy: float
    ground_states: np.ndarray

    @classmethod
    def from_sequence(cls, seq: str, penalty: float | None = None) -> HPProblem:
        n_residues = len(seq)
        if penalty is None:
            penalty = default_penalty(n_residues)

        folds = [
            lattice.fold_of(x, n_residues)
            for x in range(2 ** lattice.n_qubits(n_residues))
        ]
        cost = np.array([energy(seq, fold, penalty) for fold in folds], dtype=float)
        valid = np.array([count_overlaps(fold) == 0 for fold in folds], dtype=bool)

        ground_energy = float(cost.min())
        ground_states = np.flatnonzero(cost == ground_energy)
        if not valid[ground_states].all():
            raise ValueError(
                f"penalty={penalty} is too small for {seq!r}: "
                "an overlapping fold is among the optima; increase the penalty"
            )
        return cls(
            seq,
            penalty,
            _read_only(cost),
            _read_only(valid),
            ground_energy,
            _read_only(ground_states),
        )

    @property
    def n_residues(self) -> int:
        return len(self.seq)

    @property
    def n_qubits(self) -> int:
        return lattice.n_qubits(len(self.seq))

    def fold(self, x: int) -> str:
        return lattice.fold_of(x, self.n_residues)

    def decode(self, x: int) -> np.ndarray:
        """Lattice coordinates of fold x, shape (n_residues, 2)."""
        return lattice.positions(self.fold(x))
