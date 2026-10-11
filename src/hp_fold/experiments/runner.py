"""Run every entry of an ExperimentConfig, save each result, and report."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from hp_fold.analysis.result import RunResult
from hp_fold.analysis.store import save_result
from hp_fold.experiments.algorithms import Runner, resolve
from hp_fold.experiments.config import ExperimentConfig
from hp_fold.folding.problem import HPProblem


@dataclass(frozen=True, slots=True)
class RunFailure:
    algorithm: str
    repeat: int | None
    seed: int | None
    error: str


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    saved: tuple[Path, ...]
    failures: tuple[RunFailure, ...]

    @property
    def ok(self) -> bool:
        return not self.failures


def run_experiment(
    config: ExperimentConfig,
    resolver: Callable[[str], Runner] = resolve,
) -> ExperimentReport:
    """Run all configured runs; one failing run never stops the others.

    Repeat i of every run uses seed config.seed + i.
    Failures are collected in the report.
    """
    problem = HPProblem.from_sequence(config.sequence, config.penalty)
    saved: list[Path] = []
    failures: list[RunFailure] = []
    for spec in config.runs:
        try:
            runner = resolver(spec.algorithm)
        except ValueError as exc:
            failures.append(RunFailure(spec.algorithm, None, None, str(exc)))
            continue
        for repeat in range(spec.repeats):
            seed = config.seed + repeat
            try:
                result = runner(problem, spec.params, seed)
                if not isinstance(result, RunResult):
                    raise TypeError(
                        f"returned {type(result).__name__}, expected RunResult"
                    )
                saved.append(save_result(problem, result, config.output))
            except Exception as exc:
                failures.append(
                    RunFailure(
                        spec.algorithm, repeat, seed, f"{type(exc).__name__}: {exc}"
                    )
                )
    return ExperimentReport(tuple(saved), tuple(failures))
