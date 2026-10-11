"""Success probability against the evaluation budget."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from hp_fold.analysis.metrics import success_vs_evaluations
from hp_fold.analysis.result import RunResult
from hp_fold.folding.problem import HPProblem
from hp_fold.viz.theme import Theme, algorithm_colors, get_theme

_LINE_WIDTH = 2.0
_Y_LIMITS = (-0.04, 1.08)


def budget_grid(
    results: Sequence[tuple[str, RunResult]],
    points: int = 200,
) -> np.ndarray:
    """Integer budgets, log-spaced from 1 to the largest budget any run spent."""
    if not results:
        raise ValueError("need at least one result")
    top = max(result.evaluations for _, result in results)
    return np.unique(np.round(np.logspace(0, np.log10(top), points)).astype(int))


def success_rows(
    problem: HPProblem,
    results: Sequence[tuple[str, RunResult]],
    budgets: np.ndarray,
) -> list[dict[str, Any]]:
    """The plotted numbers as a table (the text twin of the figure)."""
    rows: list[dict[str, Any]] = []
    for label, result in results:
        curve = success_vs_evaluations(problem, result, budgets)
        for budget, p in zip(budgets, curve, strict=True):
            rows.append(
                {
                    "algorithm": label,
                    "evaluations": int(budget),
                    "p_success": float(p),
                }
            )
    return rows


def plot_success_vs_evaluations(
    problem: HPProblem,
    results: Sequence[tuple[str, RunResult]],
    theme: str = "light",
) -> Figure:
    """One line per (label, result):
    P(at least one ground state) vs evaluations spent.
    """
    if not results:
        raise ValueError("need at least one result to plot")
    algorithms = [result.algorithm for _, result in results]
    if len(set(algorithms)) != len(algorithms):
        raise ValueError(
            "each algorithm can appear once (a color stands for an algorithm)"
        )
    colors = get_theme(theme)
    series = algorithm_colors(colors, algorithms)
    budgets = budget_grid(results)

    fig = Figure(figsize=(7.4, 4.4), facecolor=colors.surface, layout="constrained")
    ax = fig.subplots()
    _style_axes(ax, colors)

    for label, result in results:
        ax.plot(
            budgets,
            success_vs_evaluations(problem, result, budgets),
            label=label,
            color=series[result.algorithm],
            linewidth=_LINE_WIDTH,
            solid_joinstyle="round",
            solid_capstyle="round",
            drawstyle="steps-post",
            zorder=3,
        )

    ax.set_xscale("log")
    ax.set_xlim(budgets[0], budgets[-1])
    ax.set_ylim(*_Y_LIMITS)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.xaxis.set_major_formatter(FuncFormatter(_compact))
    ax.set_xlabel(
        "cost-function evaluations spent (log scale)",
        color=colors.ink_secondary,
        fontsize=9,
    )
    ax.set_ylabel("P(ground state found)", color=colors.ink_secondary, fontsize=9)
    if len(results) > 1:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="outside lower center",
            ncol=min(len(results), 4),
            frameon=False,
            fontsize=9,
            labelcolor=colors.ink_secondary,
            handlelength=1.6,
        )
    fig.suptitle(
        f"Evaluations needed to find the ground state: {problem.seq}",
        x=0.0,
        ha="left",
        color=colors.ink,
        fontsize=12,
        fontweight="bold",
    )
    return fig


def _style_axes(ax: Axes, colors: Theme) -> None:
    ax.set_facecolor(colors.surface)
    ax.grid(axis="y", color=colors.grid, linewidth=1.0, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(colors.axis)
    ax.tick_params(axis="both", length=0, labelsize=9, labelcolor=colors.ink_muted)


def _compact(value: float, _position: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}K"
    return f"{value:g}"
