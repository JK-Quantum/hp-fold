"""Draw every figure the saved runs of the most recent problem can support.

  $ uv run scripts/plot_results.py                      # output of configs/hello.toml
  $ uv run scripts/plot_results.py configs/hello.toml   # optional: another config

Run scripts/run_experiment.py first (or merge Dev B's runs into the same folder).
Figures that need data the runs do not have (a depth sweep, an optimizer trace, a
landscape, ...) are skipped with a note.
For each figure one PNG per theme is written, plus the plotted numbers as CSV where
the figure is a chart of a table, into <output>/../figures.
Exit code: 0 ok, 1 if a figure failed, 2 if nothing could be read.
"""

import csv
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure

from hp_fold.analysis.distribution import energy_distribution
from hp_fold.analysis.result import RunResult
from hp_fold.analysis.store import load_problem, load_result, problem_runs
from hp_fold.analysis.summary import series_over_param, summarize
from hp_fold.experiments.config import load_config
from hp_fold.folding.problem import HPProblem
from hp_fold.viz.budget import budget_grid, plot_success_vs_evaluations, success_rows
from hp_fold.viz.convergence import HISTORY_KEY, objective_history, plot_convergence
from hp_fold.viz.energy import plot_energy_distributions, save_figure
from hp_fold.viz.landscape import plot_landscape
from hp_fold.viz.report import (
    label_of,
    landscape_of,
    newest_per_config,
    representatives,
)
from hp_fold.viz.sweep import plot_metrics_vs_param

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "hello.toml"
THEMES = ("light", "dark")

Labeled = Sequence[tuple[str, RunResult]]
Made = tuple[Callable[[str], Figure], list[dict[str, Any]]]


class SkipError(Exception):
    """A figure this data cannot support (not an error)."""


def main() -> int:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    try:
        config = load_config(config_path, base_dir=ROOT)
        paths = problem_runs(config.output)
        every = newest_per_config([load_result(path) for path in paths])
        problem = load_problem(paths[0])
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    figures = Path(config.output).parent / "figures"
    chosen = representatives(every)
    labeled = [(label_of(r), r) for r in chosen]
    names = ", ".join(label for label, _ in labeled)
    print(f"{problem.seq}: {len(every)} runs, {len(chosen)} algorithms ({names})")

    failed = False
    for name, make in (
        ("energy", lambda: _energy(problem, labeled)),
        ("budget", lambda: _budget(problem, labeled)),
        ("convergence", lambda: _convergence(problem, labeled)),
        ("sweep", lambda: _sweep(problem, every)),
        ("landscape", lambda: _landscape(problem, every)),
    ):
        try:
            build, rows = make()
        except SkipError as why:
            print(f"skipped {name}: {why}")
            continue
        except ValueError as exc:
            print(f"error: {name}: {exc}", file=sys.stderr)
            failed = True
            continue
        for theme in THEMES:
            target = figures / f"{name}_{problem.seq}_{theme}.png"
            print(f"wrote {save_figure(build(theme), target)}")
        if rows:
            print(f"wrote {_write_csv(figures / f'{name}_{problem.seq}.csv', rows)}")
    return 1 if failed else 0


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


def _sweep(problem: HPProblem, every: Sequence[RunResult]) -> Made:
    series = series_over_param(summarize(problem, every), "p")
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


def _landscape(problem: HPProblem, every: Sequence[RunResult]) -> Made:
    try:
        found = landscape_of(every)
    except ValueError as exc:
        raise SkipError(f"unusable landscape data ({exc})") from exc
    if found is None:
        raise SkipError(
            "no run carries "
            "extras['landscape_gammas'/'landscape_betas'/'landscape_values']"
        )
    result, landscape, angles = found
    gamma, beta, value = landscape.best()
    print(
        f"landscape of {label_of(result)}: grid minimum {value:.3f} at "
        f"gamma={gamma:.3f}, "
        f"beta={beta:.3f}"
    )
    references = {"random guess": float(problem.cost.mean())}

    def build(theme: str) -> Figure:
        return plot_landscape(
            landscape, references, angles, f"QAOA landscape: {problem.seq}", theme
        )

    return build, []


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _tidy(v) for k, v in row.items()})
    return path


def _tidy(value: Any) -> Any:
    if isinstance(value, float):
        return "" if math.isnan(value) else round(value, 10)
    return value


if __name__ == "__main__":
    sys.exit(main())
