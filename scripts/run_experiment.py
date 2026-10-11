"""Run the experiment described by a config file and print a summary.

  $ uv run scripts/run_experiment.py                      # runs configs/hello.toml
  $ uv run scripts/run_experiment.py configs/other.toml   # optional: another config

Everything that defines the experiment lives in the config file, so the config
reproduces the results. Exit code is 1 if any run failed.
"""

import math
import sys
from pathlib import Path

from hp_fold.analysis.store import load_result
from hp_fold.analysis.summary import MeanStd, SummaryRow, summarize
from hp_fold.experiments.config import load_config
from hp_fold.experiments.runner import run_experiment
from hp_fold.folding.problem import HPProblem

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "hello.toml"
PER_RUN = False


def main() -> int:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    config = load_config(config_path, base_dir=ROOT)
    report = run_experiment(config)

    problem = HPProblem.from_sequence(config.sequence, config.penalty)
    print(f"config   {config_path}")
    print(
        f"sequence {problem.seq}: {problem.n_qubits} qubits, {2**problem.n_qubits} "
        f"folds, "
        f"{int(problem.valid.sum())} valid, ground energy {problem.ground_energy:g} "
        f"({len(problem.ground_states)} ground states), penalty {problem.penalty:g}"
    )
    print(f"saved    {len(report.saved)} runs to {config.output}")

    results = [load_result(path) for path in report.saved]
    if PER_RUN:
        rows = [summarize(problem, [r])[0] for r in results]
    else:
        rows = summarize(problem, results)
    print_table(rows, label_seed=PER_RUN, seeds=[r.seed for r in results])

    for failure in report.failures:
        where = (
            ""
            if failure.repeat is None
            else f" (repeat {failure.repeat}, seed {failure.seed})"
        )
        print(f"FAILED   {failure.algorithm}{where}: {failure.error}", file=sys.stderr)
    return 0 if report.ok else 1


def print_table(
    rows: list[SummaryRow], label_seed: bool, seeds: list[int | None]
) -> None:
    labels = [_label(row) for row in rows]
    if label_seed:
        labels = [
            f"{label} seed={seed}" for label, seed in zip(labels, seeds, strict=True)
        ]
    width = max([len("algorithm"), *map(len, labels)])
    print(
        f"\n{'algorithm':<{width}}  {'runs':>4}  {'evals/run':>9}  "
        f"{'P(ground)':>15}  {'P(valid)':>15}  {'ratio':>15}"
    )
    for label, row in zip(labels, rows, strict=True):
        print(
            f"{label:<{width}}  {row.runs:>4}  {row.evaluations:>9.0f}  "
            f"{_fmt(row.p_ground):>15}  {_fmt(row.p_valid):>15}  "
            f"{_fmt(row.approx_ratio):>15}"
        )


def _label(row: SummaryRow) -> str:
    params = " ".join(f"{k}={v}" for k, v in row.config.items())
    return f"{row.algorithm} {params}".strip()


def _fmt(value: MeanStd) -> str:
    if math.isnan(value.mean):
        return "-"
    if math.isnan(value.std):
        return f"{value.mean:.3f}"
    return f"{value.mean:.3f} ± {value.std:.3f}"


if __name__ == "__main__":
    sys.exit(main())
