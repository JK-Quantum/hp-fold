"""Optimizer progress.

Best expected energy so far, per iteration, one panel per algorithm.
"""

from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from hp_fold.analysis.result import RunResult
from hp_fold.folding.problem import HPProblem
from hp_fold.viz.theme import Theme, algorithm_colors, get_theme

HISTORY_KEY = "objective_history"
_LINE_WIDTH = 2.0
_MARKER_SIZE = 8.0
_PANEL_HEIGHT = 2.3  # inches


def objective_history(label: str, result: RunResult) -> np.ndarray:
    """The run's optimizer trace (extras["objective_history"]) as an 1-D array."""
    if HISTORY_KEY not in result.extras:
        raise ValueError(
            f"{label}: result has no extras[{HISTORY_KEY!r}] (the optimizer trace)"
        )
    history = np.asarray(result.extras[HISTORY_KEY], dtype=float)
    if history.ndim != 1 or history.size == 0 or not np.isfinite(history).all():
        raise ValueError(
            f"{label}: extras[{HISTORY_KEY!r}] must be a non-empty 1-D finite list"
        )
    return history


def plot_convergence(
    problem: HPProblem,
    results: Sequence[tuple[str, RunResult]],
    theme: str = "light",
) -> Figure:
    """One panel per (label, result).

    Each has its own axes since objectives can differ.
    """
    if not results:
        raise ValueError("need at least one result to plot")
    histories = [objective_history(label, result) for label, result in results]
    colors = get_theme(theme)
    by_algorithm = algorithm_colors(colors, [result.algorithm for _, result in results])

    fig = Figure(
        figsize=(7.4, 1.0 + _PANEL_HEIGHT * len(results)),
        facecolor=colors.surface,
        layout="constrained",
    )
    axes = fig.subplots(len(results), 1, squeeze=False)[:, 0]
    for ax, (label, result), history in zip(axes, results, histories, strict=True):
        _style_axes(ax, colors)
        ax.set_title(
            label, loc="left", color=colors.ink, fontsize=10, fontweight="bold"
        )
        x = np.arange(1, history.size + 1)
        color = by_algorithm[result.algorithm]
        ax.plot(
            x,
            history,
            color=color,
            linewidth=_LINE_WIDTH,
            solid_joinstyle="round",
            solid_capstyle="round",
            drawstyle="steps-post",
            zorder=3,
            label=label,
        )
        ax.plot(
            [x[-1]],
            [history[-1]],
            marker="o",
            markersize=_MARKER_SIZE,
            color=color,
            markeredgecolor=colors.surface,
            markeredgewidth=2,
            linestyle="none",
            zorder=4,
            clip_on=False,
            label="_end",
        )
        ax.annotate(
            f"{history[-1]:.2f}",
            (x[-1], history[-1]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="right",
            color=colors.ink,
            fontsize=9,
        )
        ground = problem.ground_energy
        ax.axhline(
            ground, color=colors.ink_muted, linewidth=1, zorder=2, label="_ground"
        )
        ax.annotate(
            "ground state energy",
            (0.995, ground),
            xycoords=("axes fraction", "data"),
            xytext=(0, 5),
            textcoords="offset points",
            ha="right",
            color=colors.ink_secondary,
            fontsize=9,
        )
        ax.set_xlim(1, max(2, x[-1]))
        ax.set_xlabel("optimizer iteration", color=colors.ink_secondary, fontsize=9)
        ax.set_ylabel("best energy so far", color=colors.ink_secondary, fontsize=9)
    fig.suptitle(
        f"How the optimizer converges: {problem.seq}",
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
