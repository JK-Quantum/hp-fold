"""The metrics table of an experiment as text."""

from __future__ import annotations

import math
from collections.abc import Sequence

from hp_fold.analysis.summary import MeanStd, SummaryRow


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


def params_text(row: SummaryRow) -> str:
    return " ".join(f"{k}={v}" for k, v in row.config.items())


def _label(row: SummaryRow) -> str:
    return f"{row.algorithm} {params_text(row)}".strip()


def _fmt(value: MeanStd) -> str:
    if math.isnan(value.mean):
        return "-"
    if math.isnan(value.std):
        return f"{value.mean:.3f}"
    return f"{value.mean:.3f} ± {value.std:.3f}"
