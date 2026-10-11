from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile

from hp_fold.quantum.numpy_qaoa import NumpyQAOA
from hp_fold.quantum.qaoa import QAOACircuit


@dataclass(frozen=True)
class CircuitStats:
    """Size of the circuit after transpiling for the target backend."""

    depth: int
    n_gates: int
    n_two_qubit_gates: int
    ops: dict[str, int]


@dataclass(frozen=True)
class RunResult:
    backend: str
    noise_model: str | None
    job_id: str | None
    shots: int
    params: list[float]
    stats: CircuitStats
    counts: dict[int, int]
    p_ground: float
    p_valid: float
    p_ground_given_valid: float | None
    mean_energy: float
    best_sampled_energy: float
    tv_distance: float
    tv_distance_shot_noise: float
    probabilities: np.ndarray = field(repr=False)
    exact_probabilities: np.ndarray = field(repr=False)

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("probabilities")
        data.pop("exact_probabilities")
        data["counts"] = {str(x): c for x, c in sorted(self.counts.items())}
        return data


@dataclass(frozen=True)
class BatchEntry:
    """One circuit of a batched run: fold counts and its transpiled size."""

    counts: dict[int, int]
    stats: CircuitStats


_OPAQUE_BLOCKS = ["QAOA", "PauliEvolution"]


def expand_opaque_blocks(circuit: QuantumCircuit) -> QuantumCircuit:
    """Decompose the QAOA block and its PauliEvolution gates into plain gates."""
    for _ in range(8):
        if not set(_OPAQUE_BLOCKS) & set(circuit.count_ops()):
            return circuit
        circuit = circuit.decompose(gates_to_decompose=_OPAQUE_BLOCKS)
    raise RuntimeError("could not expand the QAOA circuit into plain gates")


class BackendRunner:
    """Transpile, run and score one QAOA circuit on a backend."""

    def __init__(
        self,
        circuit: QAOACircuit,
        backend: Any,
        *,
        noise_model: str | None = None,
        optimization_level: int = 1,
        run_options: dict[str, Any] | None = None,
    ) -> None:
        self._circuit = circuit
        self._backend = backend
        self._noise_model = noise_model
        self._optimization_level = optimization_level
        self._run_options = dict(run_options or {})

    @property
    def backend_name(self) -> str:
        return str(getattr(self._backend, "name", self._backend))

    def transpile(self, params: np.ndarray) -> QuantumCircuit:
        """The measured circuit compiled for the backend's gate set."""
        return transpile(
            expand_opaque_blocks(self._circuit.measured_circuit(params)),
            self._backend,
            optimization_level=self._optimization_level,
        )

    @staticmethod
    def stats_of(transpiled: QuantumCircuit) -> CircuitStats:
        ops = {str(k): int(v) for k, v in transpiled.count_ops().items()}
        return CircuitStats(
            depth=int(transpiled.depth()),
            n_gates=int(
                sum(v for k, v in ops.items() if k not in ("measure", "barrier"))
            ),
            n_two_qubit_gates=int(transpiled.num_nonlocal_gates()),
            ops=ops,
        )

    def run(self, params: np.ndarray, shots: int, *, seed: int = 0) -> RunResult:
        if shots < 1:
            raise ValueError(f"shots must be >= 1, got {shots}")
        params = np.asarray(params, dtype=float)
        transpiled = self.transpile(params)
        job = self._backend.run(transpiled, shots=shots, **self._run_options)
        counts = self.counts_to_folds(job.result().get_counts())
        job_id = job.job_id() if hasattr(job, "job_id") else None
        return self._score(params, transpiled, counts, shots, job_id, seed)

    @staticmethod
    def counts_to_folds(raw_counts: dict[str, int]) -> dict[int, int]:
        """Bitstring counts -> {fold index x: count}."""
        folds: dict[int, int] = {}
        for bits, count in raw_counts.items():
            x = int(bits.replace(" ", ""), 2)
            folds[x] = folds.get(x, 0) + int(count)
        return folds

    def _score(
        self,
        params: np.ndarray,
        transpiled: QuantumCircuit,
        counts: dict[int, int],
        shots: int,
        job_id: str | None,
        seed: int,
    ) -> RunResult:
        problem = self._circuit.problem
        total = sum(counts.values())
        measured = np.zeros(problem.cost.size)
        for x, count in counts.items():
            measured[x] = count / total
        exact = NumpyQAOA(problem, self._circuit.n_params // 2).probabilities(params)

        rng = np.random.default_rng(seed)
        ideal_sample = rng.multinomial(total, exact) / total
        p_valid = float(measured[problem.valid].sum())
        p_ground = float(measured[problem.ground_states].sum())
        seen = np.flatnonzero(measured)
        return RunResult(
            backend=self.backend_name,
            noise_model=self._noise_model,
            job_id=job_id,
            shots=shots,
            params=params.tolist(),
            stats=self.stats_of(transpiled),
            counts=counts,
            p_ground=p_ground,
            p_valid=p_valid,
            p_ground_given_valid=p_ground / p_valid if p_valid > 0 else None,
            mean_energy=float(measured @ problem.cost),
            best_sampled_energy=float(problem.cost[seen].min()),
            tv_distance=0.5 * float(np.abs(measured - exact).sum()),
            tv_distance_shot_noise=0.5 * float(np.abs(ideal_sample - exact).sum()),
            probabilities=measured,
            exact_probabilities=exact,
        )


def run_batch(
    backend: Any,
    circuits: Sequence[QAOACircuit],
    params: Sequence[np.ndarray],
    shots: int,
    optimization_level: int = 1,
) -> list[BatchEntry]:
    """Run every circuit with its angles in ONE `backend.run` call (a single job)."""
    if shots < 1:
        raise ValueError(f"shots must be >= 1, got {shots}")
    if not circuits or len(circuits) != len(params):
        raise ValueError("need one parameter vector per circuit, and at least one")
    measured = [
        expand_opaque_blocks(circuit.measured_circuit(angles))
        for circuit, angles in zip(circuits, params, strict=True)
    ]
    transpiled = transpile(measured, backend, optimization_level=optimization_level)
    result = backend.run(transpiled, shots=shots).result()
    return [
        BatchEntry(
            BackendRunner.counts_to_folds(result.get_counts(index)),
            BackendRunner.stats_of(circuit),
        )
        for index, circuit in enumerate(transpiled)
    ]
