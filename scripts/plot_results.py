"""Plot the most recent saved runs (run scripts/run_experiment.py first).

  $ uv run scripts/plot_results.py                      # output of configs/hello.toml
  $ uv run scripts/plot_results.py configs/other.toml   # optional: another config

Shows the newest run of each algorithm for the most recent problem in the config output
folder, and writes one PNG per theme (plus the numbers as CSV) to <output>/../figures.
"""

import csv
import sys
from pathlib import Path

from hp_fold.analysis.distribution import energy_distribution
from hp_fold.analysis.store import latest_runs, load_problem, load_result
from hp_fold.experiments.config import load_config
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

    for theme in THEMES:
        figure = plot_energy_distributions(problem, labeled, theme)
        out = save_figure(figure, figures / f"energy_{problem.seq}_{theme}.png")
        print(f"wrote {out}")

    table = figures / f"energy_{problem.seq}.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "algorithm",
                "energy",
                "p_ground",
                "p_valid",
                "p_invalid",
            ],
        )
        writer.writeheader()
        for label, result in labeled:
            for row in energy_distribution(problem, result).rows():
                rounded = {key: round(value, 10) for key, value in row.items()}
                writer.writerow({"algorithm": label, **rounded})
    print(f"wrote {table}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
