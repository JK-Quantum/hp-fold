"""Draw every figure a set of runs of one problem can support.

Figures that need data the runs do not have (a depth sweep, an optimizer trace, a
landscape, ...) are skipped with a reason. For each figure one PNG per theme is
written, plus the plotted numbers as CSV where the figure is a chart of a table.
"""

import csv
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure

from hp_fold.analysis.distribution import energy_distribution
from hp_fold.analysis.result import RunResult
from hp_fold.analysis.summary import series_over_param, summarize
from hp_fold.folding.problem import HPProblem
from hp_fold.viz.budget import budget_grid, plot_success_vs_evaluations, success_rows
from hp_fold.viz.convergence import HISTORY_KEY, objective_history, plot_convergence
from hp_fold.viz.energy import plot_energy_distributions, save_figure
from hp_fold.viz.landscape import plot_landscape
from hp_fold.viz.report import (
    label_of,
    landscape_of,
    representatives,
)
from hp_fold.viz.sweep import plot_metrics_vs_param

THEMES = ("light", "dark")

Labeled = Sequence[tuple[str, RunResult]]
Made = tuple[Callable[[str], Figure], list[dict[str, Any]]]


class SkipError(Exception):
    """A figure this data cannot support (not an error)."""


@dataclass(frozen=True, slots=True)
class FigureReport:
    written: tuple[Path, ...]
    skipped: Mapping[str, str]
    errors: Mapping[str, str]


def draw_figures(
    problem: HPProblem, results: Sequence[RunResult], directory: Path
) -> FigureReport:
    """Write the figures of `results` (oldest first) into `directory`."""
    labeled = [(label_of(r), r) for r in representatives(results)]
    written: list[Path] = []
    skipped: dict[str, str] = {}
    errors: dict[str, str] = {}
    for name, make in (
        ("energy", lambda: _energy(problem, labeled)),
        ("budget", lambda: _budget(problem, labeled)),
        ("convergence", lambda: _convergence(problem, labeled)),
        ("sweep", lambda: _sweep(problem, results)),
        ("landscape", lambda: _landscape(problem, results)),
    ):
        try:
            build, rows = make()
        except SkipError as why:
            skipped[name] = str(why)
            continue
        except ValueError as exc:
            errors[name] = str(exc)
            continue
        for theme in THEMES:
            target = directory / f"{name}_{problem.seq}_{theme}.png"
            written.append(save_figure(build(theme), target))
        if rows:
            written.append(write_csv(directory / f"{name}_{problem.seq}.csv", rows))
    return FigureReport(tuple(written), skipped, errors)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _tidy(v) for k, v in row.items()})
    return path


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


def _tidy(value: Any) -> Any:
    if isinstance(value, float):
        return "" if math.isnan(value) else round(value, 10)
    return value
