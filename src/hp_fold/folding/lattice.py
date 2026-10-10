"""Folds on the 2D square lattice: fold strings, their integer encoding, and geometry.

Conventions (shared with the quantum side):
- A fold is a string over "RULD" (right, up, left, down): one move per chain bond,
  in chain order. The first move is always "R", which removes rotational symmetry.

- The integer x encodes the remaining n_residues - 2 moves, two bits each, with
  free move k in bits 2k and 2k+1. Equivalently, qubit q is bit q of x, which is
  also Qiskit's own indexing convention, so no reordering is needed there.

Direction-per-bond with two bits per bond follows Perdomo-Ortiz et al. (2012).
Using absolute directions and fixing the first bond is a design choice of this project.
"""

import numpy as np

MOVES = "RULD"
_VECTORS = np.array([(1, 0), (0, 1), (-1, 0), (0, -1)])


def _check_n_residues(n_residues: int) -> None:
    if n_residues < 3:
        raise ValueError(f"need at least 3 residues, got {n_residues}")


def _check_fold(fold: str) -> None:
    _check_n_residues(len(fold) + 1)
    if fold[0] != "R":
        raise ValueError(f"first move must be 'R', got {fold[0]!r}")
    invalid = set(fold) - set(MOVES)
    if invalid:
        raise ValueError(f"invalid moves {sorted(invalid)}; use only {MOVES!r}")


def n_qubits(n_residues: int) -> int:
    _check_n_residues(n_residues)
    return 2 * (n_residues - 2)


def x_of(fold: str) -> int:
    _check_fold(fold)
    return sum(MOVES.index(move) << (2 * k) for k, move in enumerate(fold[1:]))


def fold_of(x: int, n_residues: int) -> str:
    n = n_qubits(n_residues)
    if not 0 <= x < 2**n:
        raise ValueError(
            f"x={x} out of range for {n_residues} residues (0 <= x < {2**n})"
        )
    return "R" + "".join(MOVES[(x >> (2 * k)) & 3] for k in range(n_residues - 2))


def positions(fold: str) -> np.ndarray:
    """Lattice coordinates, shape (n_residues, 2), starting at the origin."""
    _check_fold(fold)
    steps = _VECTORS[[MOVES.index(move) for move in fold]]
    return np.vstack([[0, 0], np.cumsum(steps, axis=0)])
