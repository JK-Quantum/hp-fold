"""Probability by fold energy, one panel per algorithm."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from hp_fold.analysis.distribution import energy_distribution
from hp_fold.analysis.metrics import Metrics, compute_metrics
from hp_fold.analysis.result import RunResult
from hp_fold.folding.problem import HPProblem
from hp_fold.viz.theme import Theme, get_theme

GROUND_LABEL = "ground state"
VALID_LABEL = "other valid fold"
INVALID_LABEL = "overlapping fold (invalid)"

_PANEL_HEIGHT = 1.9  # inches
_BAR_WIDTH = 0.55


def plot_energy_distributions(
    problem: HPProblem,
    results: Sequence[tuple[str, RunResult]],
    theme: str = "light",
) -> Figure:
    """Stacked probability by energy level

    One panel per (label, result), shared axes.
    """
    if not results:
        raise ValueError("need at least one result to plot")
    colors = get_theme(theme)
    distributions = [energy_distribution(problem, result) for _, result in results]
    levels = distributions[0].energies
    n = len(results)

    fig = Figure(
        figsize=(7.4, 1.2 + _PANEL_HEIGHT * n),
        facecolor=colors.surface,
        layout="constrained",
    )
    axes = fig.subplots(n, 1, sharex=True, sharey=True, squeeze=False)[:, 0]
    x = np.arange(levels.size)

    for ax, (label, result), dist in zip(axes, results, distributions, strict=True):
        _style_axes(ax, colors)
        bottom = np.zeros(levels.size)
        for values, color, name in (
            (dist.p_ground, colors.accent, GROUND_LABEL),
            (dist.p_valid, colors.neutral, VALID_LABEL),
            (dist.p_invalid, colors.alert, INVALID_LABEL),
        ):
            ax.bar(
                x,
                values,
                _BAR_WIDTH,
                bottom=bottom,
                color=color,
                label=name,
                edgecolor=colors.surface,
                linewidth=1.5,
                zorder=3,
            )
            bottom = bottom + values
        ax.set_title(
            label, loc="left", color=colors.ink, fontsize=10, fontweight="bold"
        )
        _annotate(ax, compute_metrics(problem, result), colors)

    axes[0].set_ylim(0, 1.18)
    axes[0].set_yticks([0, 0.5, 1])
    axes[0].set_yticklabels(["0", "0.5", "1"])
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([f"{e:g}" for e in levels])
    axes[-1].set_xlabel(
        "fold energy (lower is better)",
        color=colors.ink_secondary,
        fontsize=9,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        labelcolor=colors.ink_secondary,
        handlelength=1.0,
        handleheight=1.0,
    )
    fig.supylabel("probability", color=colors.ink_secondary, fontsize=9)
    fig.suptitle(
        f"Where the probability goes: {problem.seq}",
        x=0.0,
        ha="left",
        color=colors.ink,
        fontsize=12,
        fontweight="bold",
    )
    return fig


def save_figure(fig: Figure, path: str | Path, dpi: int = 200) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    return path


def _style_axes(ax: Axes, colors: Theme) -> None:
    ax.set_facecolor(colors.surface)
    ax.grid(axis="y", color=colors.grid, linewidth=1.0, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(colors.axis)
    ax.tick_params(axis="both", length=0, labelsize=9, labelcolor=colors.ink_muted)


def _annotate(ax: Axes, metrics: Metrics, colors: Theme) -> None:
    """The two numbers in a fixed corner so they never collide with bars."""
    ax.text(
        0.995,
        0.97,
        f"P(ground) {metrics.p_ground:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=colors.ink,
        fontsize=9,
    )
    ax.text(
        0.995,
        0.80,
        f"P(valid) {metrics.p_valid:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=colors.ink_secondary,
        fontsize=9,
    )
