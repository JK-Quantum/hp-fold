"""Aggregate repeated runs of the same algorithm into one comparable row."""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from hp_fold.analysis.metrics import compute_metrics
from hp_fold.analysis.result import RunResult
from hp_fold.folding.problem import HPProblem


@dataclass(frozen=True, slots=True)
class MeanStd:
    mean: float
    std: float


@dataclass(frozen=True, slots=True)
class SummaryRow:
    algorithm: str
    config: Mapping[str, Any]
    runs: int
    evaluations: float
    p_ground: MeanStd
    p_valid: MeanStd
    approx_ratio: MeanStd


def summarize(problem: HPProblem, results: Sequence[RunResult]) -> list[SummaryRow]:
    """One row per distinct (algorithm, config), in order of first appearance."""
    groups: dict[tuple[str, str], list[RunResult]] = {}
    for result in results:
        key = (result.algorithm, json.dumps(dict(result.config), sort_keys=True))
        groups.setdefault(key, []).append(result)

    rows = []
    for members in groups.values():
        metrics = [compute_metrics(problem, r) for r in members]
        rows.append(
            SummaryRow(
                algorithm=members[0].algorithm,
                config=dict(members[0].config),
                runs=len(members),
                evaluations=float(np.mean([r.evaluations for r in members])),
                p_ground=_mean_std([m.p_ground for m in metrics]),
                p_valid=_mean_std([m.p_valid for m in metrics]),
                approx_ratio=_mean_std([m.approx_ratio for m in metrics]),
            )
        )
    return rows


def _mean_std(values: Sequence[float]) -> MeanStd:
    defined = [v for v in values if not math.isnan(v)]
    if not defined:
        return MeanStd(math.nan, math.nan)
    std = float(np.std(defined, ddof=1)) if len(defined) > 1 else math.nan
    return MeanStd(float(np.mean(defined)), std)


@dataclass(frozen=True, slots=True)
class ParamSeries:
    """One algorithm's summary rows ordered by a numeric config parameter."""

    algorithm: str
    param: str
    values: tuple[float, ...]
    rows: tuple[SummaryRow, ...]


def series_over_param(rows: Sequence[SummaryRow], param: str) -> list[ParamSeries]:
    """Group rows by algorithm, keep those that set `param`, and order them by value.

    Algorithms in order of first appearance. Rows without the parameter are skipped; an
    algorithm with two rows at the same value is ambiguous and rejected.
    """
    grouped: dict[str, list[tuple[float, SummaryRow]]] = {}
    for row in rows:
        if param not in row.config:
            continue
        value = row.config[param]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(
                f"{row.algorithm}: {param!r} must be a number, got {value!r}"
            )
        grouped.setdefault(row.algorithm, []).append((float(value), row))

    series = []
    for algorithm, pairs in grouped.items():
        pairs.sort(key=lambda pair: pair[0])
        values = tuple(value for value, _ in pairs)
        if len(set(values)) != len(values):
            raise ValueError(
                f"{algorithm}: several rows have the same {param!r} value; "
                "filter the other config differences first"
            )
        series.append(
            ParamSeries(algorithm, param, values, tuple(row for _, row in pairs))
        )
    return series
