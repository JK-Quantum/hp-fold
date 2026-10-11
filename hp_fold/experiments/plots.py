"""Build every figure a set of runs of one problem can support.

Figures that need data the runs do not have (a depth sweep, an optimizer trace, a
landscape, ...) are skipped with a reason.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from matplotlib.figure import Figure

from hp_fold.analysis.distribution import energy_distribution
from hp_fold.analysis.result import RunResult
from hp_fold.analysis.summary import series_over_param, summarize
from hp_fold.folding.problem import HPProblem
from hp_fold.viz.budget import budget_grid, plot_success_vs_evaluations, success_rows
from hp_fold.viz.convergence import HISTORY_KEY, objective_history, plot_convergence
from hp_fold.viz.energy import plot_energy_distributions
from hp_fold.viz.fold import plot_fold
from hp_fold.viz.landscape import plot_landscape
from hp_fold.viz.report import (
    label_of,
    landscape_of,
    representatives,
)
from hp_fold.viz.sweep import plot_metrics_vs_param

Labeled = Sequence[tuple[str, RunResult]]
Made = tuple[Callable[[str], Figure], list[dict[str, Any]]]


class SkipError(Exception):
    """A figure this data cannot support (not an error)."""


@dataclass(frozen=True, slots=True)
class FigureSet:
    figures: tuple[tuple[str, Figure], ...]
    skipped: Mapping[str, str]
    errors: Mapping[str, str]


def build_figures(
    problem: HPProblem, results: Sequence[RunResult], theme: str = "light"
) -> FigureSet:
    """The figures of `results` (oldest first), kept in memory."""
    figures: list[tuple[str, Figure]] = []
    skipped: dict[str, str] = {}
    errors: dict[str, str] = {}
    for name, make in _makers(problem, results):
        try:
            build, _ = make()
        except SkipError as why:
            skipped[name] = str(why)
            continue
        except ValueError as exc:
            errors[name] = str(exc)
            continue
        figures.append((name, build(theme)))
    return FigureSet(tuple(figures), skipped, errors)


def _makers(
    problem: HPProblem, results: Sequence[RunResult]
) -> tuple[tuple[str, Callable[[], Made]], ...]:
    labeled = [(label_of(r), r) for r in representatives(results)]
    return (
        ("fold", lambda: _fold(problem)),
        ("energy", lambda: _energy(problem, labeled)),
        ("budget", lambda: _budget(problem, labeled)),
        ("convergence", lambda: _convergence(problem, labeled)),
        ("sweep", lambda: _sweep(problem, results)),
        ("landscape", lambda: _landscape(problem, results)),
    )


def _fold(problem: HPProblem) -> Made:
    ground = int(problem.ground_states[0])
    return (lambda theme: plot_fold(problem, ground, theme)), []


def _energy(problem: HPProblem, labeled: Labeled) -> Made:
    rows = [
        {"algorithm": label, **row}
        for label, result in labeled
        for row in energy_distribution(problem, result).rows()
    ]
    return (lambda theme: plot_energy_distributions(problem, labeled, theme)), rows


def _budget(problem: HPProblem, labeled: Labeled) -> Made:
    rows = success_rows(problem, labeled, budget_grid(labeled))
    return (lambda theme: plot_success_vs_evaluations(problem, labeled, theme)), rows


def _convergence(problem: HPProblem, labeled: Labeled) -> Made:
    traced = [(label, r) for label, r in labeled if HISTORY_KEY in r.extras]
    if not traced:
        raise SkipError(
            "no run recorded an optimizer trace (extras['objective_history'])"
        )
    rows = [
        {"algorithm": label, "iteration": i, "best_energy": float(value)}
        for label, result in traced
        for i, value in enumerate(objective_history(label, result), start=1)
    ]
    return (lambda theme: plot_convergence(problem, traced, theme)), rows


def _sweep(problem: HPProblem, results: Sequence[RunResult]) -> Made:
    try:
        series = series_over_param(summarize(problem, results), "p")
    except ValueError as exc:
        raise SkipError(str(exc)) from exc
    if not series:
        raise SkipError("no run sets a depth `p` in its params")
    rows = [
        {
            "algorithm": s.algorithm,
            "p": value,
            "runs": row.runs,
            "p_ground_mean": row.p_ground.mean,
            "p_ground_std": row.p_ground.std,
            "p_valid_mean": row.p_valid.mean,
            "p_valid_std": row.p_valid.std,
        }
        for s in series
        for value, row in zip(s.values, s.rows, strict=True)
    ]
    return (
        lambda theme: plot_metrics_vs_param(problem, series, theme, "circuit depth p")
    ), rows


def _landscape(problem: HPProblem, results: Sequence[RunResult]) -> Made:
    try:
        found = landscape_of(results)
    except ValueError as exc:
        raise SkipError(f"unusable landscape data ({exc})") from exc
    if found is None:
        raise SkipError(
            "no run carries "
            "extras['landscape_gammas'/'landscape_betas'/'landscape_values']"
        )
    _, landscape, angles = found
    references = {"random guess": float(problem.cost.mean())}

    def build(theme: str) -> Figure:
        return plot_landscape(
            landscape, references, angles, f"QAOA landscape: {problem.seq}", theme
        )

    return build, []

