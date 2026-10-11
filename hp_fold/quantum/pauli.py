from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp


def walsh_hadamard(values: np.ndarray) -> np.ndarray:
    """Unnormalized fast Walsh-Hadamard transform."""
    a = np.array(values, dtype=float)
    n = a.size
    if n == 0 or n & (n - 1):
        raise ValueError(f"length must be a power of two, got {n}")
    h = 1
    while h < n:
        a = a.reshape(-1, 2, h)
        a = np.stack((a[:, 0] + a[:, 1], a[:, 0] - a[:, 1]), axis=1)
        h *= 2
    return a.reshape(n)


def cost_to_pauli(cost: np.ndarray, tol: float = 1e-9) -> SparsePauliOp:
    """Z-string expansion of a diagonal cost vector."""
    cost = np.asarray(cost, dtype=float)
    n_qubits = int(np.log2(cost.size)) if cost.size else 0
    coefficients = walsh_hadamard(cost) / cost.size

    keep = np.flatnonzero(np.abs(coefficients) > tol)
    if keep.size == 0:
        return SparsePauliOp("I" * n_qubits, coeffs=np.zeros(1))
    labels = [_label(int(s), n_qubits) for s in keep]
    return SparsePauliOp(labels, coeffs=coefficients[keep])


def _label(subset: int, n_qubits: int) -> str:
    return "".join("Z" if subset >> q & 1 else "I" for q in reversed(range(n_qubits)))


@dataclass(frozen=True)
class PauliStats:
    n_qubits: int
    n_terms: int
    has_identity: bool
    weight_histogram: dict[int, int]
    max_weight: int
    mean_weight: float
    cnot_per_layer: int


def pauli_stats(op: SparsePauliOp) -> PauliStats:
    weights = np.array(
        [sum(c == "Z" for c in label) for label in op.paulis.to_labels()], dtype=int
    )
    nonidentity = weights[weights > 0]
    histogram = {int(w): int((nonidentity == w).sum()) for w in np.unique(nonidentity)}
    multi = nonidentity[nonidentity >= 2]
    return PauliStats(
        n_qubits=int(op.num_qubits or 0),
        n_terms=int(nonidentity.size),
        has_identity=bool((weights == 0).any()),
        weight_histogram=histogram,
        max_weight=int(nonidentity.max()) if nonidentity.size else 0,
        mean_weight=float(nonidentity.mean()) if nonidentity.size else 0.0,
        cnot_per_layer=int((2 * (multi - 1)).sum()),
    )
