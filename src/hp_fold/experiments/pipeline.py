"""One config, end to end: define the problem, run, take metrics, draw the figures."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from hp_fold.analysis.store import load_result
from hp_fold.analysis.summary import MeanStd, SummaryRow, summarize
from hp_fold.experiments.config import load_config
from hp_fold.experiments.plots import FigureReport, draw_figures, write_csv
from hp_fold.experiments.runner import RunFailure, run_experiment
from hp_fold.folding.problem import HPProblem
from hp_fold.viz.report import newest_per_config

FIGURES_DIR = "figures"
SUMMARY_FILE = "summary.csv"


@dataclass(frozen=True, slots=True)
class PipelineReport:
    name: str
    origin: str
    problem: HPProblem
    saved: int
    failures: tuple[RunFailure, ...]
    rows: tuple[SummaryRow, ...]
    summary_path: Path | None
    figures: FigureReport

    @property
    def ok(self) -> bool:
        return not self.failures and not self.figures.errors


def run_pipeline(config_path: Path, base_dir: Path) -> PipelineReport:
    """Run every entry of the config; its runs, summary and figures go to `output`.

    Raises OSError or ValueError if the config cannot be read or the problem cannot
    be built.
    """
    config = load_config(config_path, base_dir=base_dir)
    report = run_experiment(config)
    problem = HPProblem.from_sequence(report.sequence, config.penalty)
    results = [load_result(path) for path in report.saved]
    rows = summarize(problem, results)
    summary_path = None
    if rows:
        summary_path = write_csv(config.output / SUMMARY_FILE, _summary_rows(rows))
    figures = (
        draw_figures(problem, newest_per_config(results), config.output / FIGURES_DIR)
        if results
        else FigureReport((), {}, {})
    )
    return PipelineReport(
        Path(config_path).stem,
        report.origin,
        problem,
        len(report.saved),
        report.failures,
        tuple(rows),
        summary_path,
        figures,
    )


def format_table(rows: Sequence[SummaryRow]) -> str:
    labels = [_label(row) for row in rows]
    width = max([len("algorithm"), *map(len, labels)])
    lines = [
        f"{'algorithm':<{width}}  {'runs':>4}  {'evals/run':>9}  "
        f"{'P(ground)':>15}  {'P(valid)':>15}  {'ratio':>15}"
    ]
    for label, row in zip(labels, rows, strict=True):
        lines.append(
            f"{label:<{width}}  {row.runs:>4}  {row.evaluations:>9.0f}  "
            f"{_fmt(row.p_ground):>15}  {_fmt(row.p_valid):>15}  "
            f"{_fmt(row.approx_ratio):>15}"
        )
    return "\n".join(lines)


def _summary_rows(rows: Sequence[SummaryRow]) -> list[dict[str, object]]:
    return [
        {
            "algorithm": row.algorithm,
            "config": _params(row),
            "runs": row.runs,
            "evaluations": row.evaluations,
            "p_ground_mean": row.p_ground.mean,
            "p_ground_std": row.p_ground.std,
            "p_valid_mean": row.p_valid.mean,
            "p_valid_std": row.p_valid.std,
            "approx_ratio_mean": row.approx_ratio.mean,
            "approx_ratio_std": row.approx_ratio.std,
        }
        for row in rows
    ]


def _params(row: SummaryRow) -> str:
    return " ".join(f"{k}={v}" for k, v in row.config.items())


def _label(row: SummaryRow) -> str:
    return f"{row.algorithm} {_params(row)}".strip()


def _fmt(value: MeanStd) -> str:
    if math.isnan(value.mean):
        return "-"
    if math.isnan(value.std):
        return f"{value.mean:.3f}"
    return f"{value.mean:.3f} ± {value.std:.3f}"
