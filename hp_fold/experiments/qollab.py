"""Train QAOA / CVaR-QAOA locally, then evaluate the trained circuits on a backend.

The angles are trained on the exact statevector. Every trained circuit is then sent to
the backend together, in one `backend.run` call (one job), and its counts become a
second result named `<algorithm>_backend` next to the ideal one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hp_fold.analysis.result import RunResult
from hp_fold.analysis.summary import SummaryRow, summarize
from hp_fold.experiments.algorithms import resolve
from hp_fold.experiments.config import load_config
from hp_fold.experiments.plots import FigureSet, build_figures
from hp_fold.experiments.problem_source import resolve_sequence
from hp_fold.folding.problem import HPProblem
from hp_fold.quantum.backend_runner import run_batch
from hp_fold.quantum.qaoa import QAOACircuit
from hp_fold.viz.report import newest_per_config

TRAINED = ("qaoa", "qaoa_cvar")


@dataclass(frozen=True, slots=True)
class QollabReport:
    problem: HPProblem
    origin: str
    backend_name: str
    results: tuple[RunResult, ...]
    rows: tuple[SummaryRow, ...]
    figures: FigureSet


def run_on_backend(
    config_path: str | Path,
    shots: int,
    backend: Any,
    base_dir: str | Path = ".",
) -> QollabReport:
    """Run the config's algorithms, evaluate the trained circuits on `backend`.

    `shots` is both the number of ideal samples and the backend shots of every trained
    circuit. Repeat i of every run uses seed config.seed + i.
    """
    config = load_config(config_path, base_dir=Path(base_dir))
    source = resolve_sequence(config)
    problem = HPProblem.from_sequence(source.sequence, config.penalty)

    ideal: list[RunResult] = []
    for spec in config.runs:
        runner = resolve(spec.algorithm)
        params = (
            {**spec.params, "shots": shots}
            if spec.algorithm in TRAINED
            else spec.params
        )
        for repeat in range(spec.repeats):
            ideal.append(runner(problem, params, config.seed + repeat))

    results = [*ideal, *_evaluate_on_backend(problem, ideal, backend, shots)]
    rows = summarize(problem, results)
    figures = build_figures(problem, newest_per_config(results))
    return QollabReport(
        problem,
        source.origin,
        str(getattr(backend, "name", backend)),
        tuple(results),
        tuple(rows),
        figures,
    )


def _evaluate_on_backend(
    problem: HPProblem,
    trained: list[RunResult],
    backend: Any,
    shots: int,
) -> list[RunResult]:
    trained = [result for result in trained if result.algorithm in TRAINED]
    if not trained:
        return []
    entries = run_batch(
        backend,
        [QAOACircuit(problem, result.config["p"]) for result in trained],
        [_angles(result.extras) for result in trained],
        shots,
    )
    return [
        _backend_result(problem, result, entry.counts, entry.stats, shots)
        for result, entry in zip(trained, entries, strict=True)
    ]


def _angles(extras: Mapping[str, Any]) -> np.ndarray:
    return np.array([*extras["gammas"], *extras["betas"]], dtype=float)


def _backend_result(
    problem: HPProblem,
    ideal: RunResult,
    counts: Mapping[int, int],
    stats: Any,
    shots: int,
) -> RunResult:
    total = sum(counts.values())
    probs = np.zeros(2**problem.n_qubits)
    for fold, count in counts.items():
        probs[fold] = count / total
    training = ideal.extras["training_evaluations"]
    return RunResult(
        f"{ideal.algorithm}_backend",
        problem.seq,
        probs,
        total,
        training + shots,
        seed=ideal.seed,
        config=dict(ideal.config),
        extras={
            "training_evaluations": training,
            "gammas": list(ideal.extras["gammas"]),
            "betas": list(ideal.extras["betas"]),
            "circuit_depth": stats.depth,
            "two_qubit_gates": stats.n_two_qubit_gates,
        },
    )
