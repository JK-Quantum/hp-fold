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
