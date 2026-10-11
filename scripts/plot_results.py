"""Plot the most recent saved runs (run scripts/run_experiment.py first).

  $ uv run scripts/plot_results.py                      # output of configs/hello.toml
  $ uv run scripts/plot_results.py configs/other.toml   # optional: another config

Shows the newest run of each algorithm for the most recent problem in the config output
folder.
For each figure it writes one PNG per theme plus the plotted numbers as CSV, into
<output>/../figures.
"""

import csv
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hp_fold.analysis.distribution import energy_distribution
from hp_fold.analysis.result import RunResult
from hp_fold.analysis.store import latest_runs, load_problem, load_result
from hp_fold.experiments.config import load_config
from hp_fold.folding.problem import HPProblem
from hp_fold.viz.budget import budget_grid, plot_success_vs_evaluations, success_rows
from hp_fold.viz.energy import plot_energy_distributions, save_figure

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "hello.toml"
THEMES = ("light", "dark")


def main() -> int:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    try:
        config = load_config(config_path, base_dir=ROOT)
        paths = latest_runs(config.output)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    problem = load_problem(paths[0])
    results = [load_result(path) for path in paths]
    labeled = [(f"{r.algorithm} (seed {r.seed})", r) for r in results]
    figures = Path(config.output).parent / "figures"
    seq = problem.seq

    try:
        for theme in THEMES:
            fig = plot_energy_distributions(problem, labeled, theme)
            print(f"wrote {save_figure(fig, figures / f'energy_{seq}_{theme}.png')}")
            fig = plot_success_vs_evaluations(problem, labeled, theme)
            print(f"wrote {save_figure(fig, figures / f'budget_{seq}_{theme}.png')}")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        "wrote "
        f"{write_csv(figures / f'energy_{seq}.csv', energy_rows(problem, labeled))}"
    )
    rows = success_rows(problem, labeled, budget_grid(labeled))
    print(f"wrote {write_csv(figures / f'budget_{seq}.csv', rows)}")
    return 0


def energy_rows(
    problem: HPProblem,
    labeled: Sequence[tuple[str, RunResult]],
) -> list[dict[str, Any]]:
    return [
        {"algorithm": label, **row}
        for label, result in labeled
        for row in energy_distribution(problem, result).rows()
    ]


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: round(v, 10) if isinstance(v, float) else v for k, v in row.items()}
            )
    return path


if __name__ == "__main__":
    sys.exit(main())
