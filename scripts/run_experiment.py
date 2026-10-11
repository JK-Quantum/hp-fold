"""Run every experiment in configs/*.toml, end to end.

  $ uv run scripts/run_experiment.py

For each config: define the problem, run the algorithms, take the metrics (printed and
saved as summary.csv) and draw the figures. Everything lives under the config's
`output` folder. A config that fails does not stop the others.
Exit code: 0 ok, 1 if any config had a failing run or figure, 2 if nothing could run.
"""

import sys
from pathlib import Path

from hp_fold.experiments.pipeline import PipelineReport, format_table, run_pipeline

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def main() -> int:
    configs = sorted(CONFIG_DIR.glob("*.toml"))
    if not configs:
        print(f"error: no configs in {CONFIG_DIR}", file=sys.stderr)
        return 2

    reports: list[PipelineReport] = []
    failed: list[str] = []
    for path in configs:
        print(f"\n=== {path.name} ===", flush=True)
        try:
            report = run_pipeline(path, ROOT)
        except (OSError, ValueError) as exc:
            print(f"error: {path.name}: {exc}", file=sys.stderr)
            failed.append(path.name)
            continue
        show(report)
        reports.append(report)
        if not report.ok:
            failed.append(path.name)

    print(f"\n{len(reports)} of {len(configs)} configs ran")
    if failed:
        print(f"problems in: {', '.join(failed)}", file=sys.stderr)
    if not reports:
        return 2
    return 1 if failed else 0


def show(report: PipelineReport) -> None:
    problem = report.problem
    print(f"origin   {report.origin}")
    print(
        f"sequence {problem.seq}: {problem.n_qubits} qubits, "
        f"{int(problem.valid.sum())} valid folds of {2**problem.n_qubits}, "
        f"ground energy {problem.ground_energy:g} "
        f"({len(problem.ground_states)} ground states), penalty {problem.penalty:g}"
    )
    print(f"saved    {report.saved} runs")
    if report.rows:
        print(f"\n{format_table(report.rows)}\n")
    for failure in report.failures:
        where = (
            ""
            if failure.repeat is None
            else f" (repeat {failure.repeat}, seed {failure.seed})"
        )
        print(f"FAILED   {failure.algorithm}{where}: {failure.error}", file=sys.stderr)
    for name, reason in report.figures.skipped.items():
        print(f"skipped  {name}: {reason}")
    for name, reason in report.figures.errors.items():
        print(f"FAILED   {name}: {reason}", file=sys.stderr)
    if report.summary_path:
        print(f"wrote    {report.summary_path}")
    for path in report.figures.written:
        print(f"wrote    {path}")


if __name__ == "__main__":
    sys.exit(main())
