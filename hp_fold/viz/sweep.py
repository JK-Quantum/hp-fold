"""Metrics against a swept parameter (e.g. QAOA depth), one line per algorithm."""

from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from hp_fold.analysis.summary import MeanStd, ParamSeries
from hp_fold.folding.problem import HPProblem
from hp_fold.viz.theme import Theme, algorithm_colors, get_theme

_PANELS = (("P(ground state)", "p_ground"), ("P(valid fold)", "p_valid"))
_LINE_WIDTH = 2.0
_MARKER_SIZE = 8.0
_HEADROOM = 1.3


def random_baselines(problem: HPProblem) -> dict[str, float]:
    """What picking a fold uniformly at random achieves, for each plotted metric."""
    return {
        "p_ground": len(problem.ground_states) / problem.cost.size,
        "p_valid": float(problem.valid.mean()),
    }


def plot_metrics_vs_param(
    problem: HPProblem,
    series: Sequence[ParamSeries],
    theme: str = "light",
    x_label: str | None = None,
) -> Figure:
    """Two stacked panels sharing the x axis; the band around a line is mean +/- std.

    Each panel has its own y axis and a thin reference line for random sampling,
    so the gain over chance is readable directly.
    """
    if not series:
        raise ValueError("need at least one series to plot")
    colors = get_theme(theme)
    by_algorithm = algorithm_colors(colors, [s.algorithm for s in series])
    param = series[0].param
    baselines = random_baselines(problem)

    fig = Figure(
        figsize=(7.4, 1.2 + 2.3 * len(_PANELS)),
        facecolor=colors.surface,
        layout="constrained",
    )
    axes = fig.subplots(len(_PANELS), 1, sharex=True, squeeze=False)[:, 0]
    for ax, (title, field) in zip(axes, _PANELS, strict=True):
        _style_axes(ax, colors)
        ax.set_title(
            title,
            loc="left",
            color=colors.ink,
            fontsize=10,
            fontweight="bold",
        )
        top = baselines[field]
        for s in series:
            top = max(top, _draw(ax, s, field, by_algorithm[s.algorithm], colors))
        ax.axhline(
            baselines[field],
            color=colors.ink_muted,
            linewidth=1,
            zorder=2,
            label="_random",
        )
        ax.annotate(
            f"random sampling: {baselines[field]:.3f}",
            (0.995, baselines[field]),
            xycoords=("axes fraction", "data"),
            xytext=(0, 5),
            textcoords="offset points",
            ha="right",
            color=colors.ink_secondary,
            fontsize=9,
        )
        ax.set_ylim(0, min(1.05, top * _HEADROOM))

    ticks = sorted({v for s in series for v in s.values})
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([f"{t:g}" for t in ticks])
    axes[-1].set_xlabel(x_label or param, color=colors.ink_secondary, fontsize=9)
    if len(series) > 1:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="outside lower center",
            ncol=min(len(series), 4),
            frameon=False,
            fontsize=9,
            labelcolor=colors.ink_secondary,
            handlelength=1.6,
        )
    fig.supylabel("probability", color=colors.ink_secondary, fontsize=9)
    fig.suptitle(
        f"Effect of {x_label or param}: {problem.seq}",
        x=0.0,
        ha="left",
        color=colors.ink,
        fontsize=12,
        fontweight="bold",
    )
    return fig


def _draw(
    ax: Axes,
    series: ParamSeries,
    field: str,
    color: str,
    colors: Theme,
) -> float:
    """Draw one line (and its spread band); return the highest value reached."""
    stats: list[MeanStd] = [getattr(row, field) for row in series.rows]
    x = np.asarray(series.values)
    mean = np.array([m.mean for m in stats])
    std = np.array([m.std for m in stats])
    spread = np.nan_to_num(std)
    if not np.isnan(std).all():
        ax.fill_between(
            x,
            mean - spread,
            mean + spread,
            color=color,
            alpha=0.10,
            linewidth=0,
        )
    ax.plot(
        x,
        mean,
        color=color,
        label=series.algorithm,
        linewidth=_LINE_WIDTH,
        solid_joinstyle="round",
        solid_capstyle="round",
        marker="o",
        markersize=_MARKER_SIZE,
        markeredgecolor=colors.surface,
        markeredgewidth=2,
        zorder=3,
    )
    return float(np.nanmax(mean + spread))


def _style_axes(ax: Axes, colors: Theme) -> None:
    ax.set_facecolor(colors.surface)
    ax.grid(axis="y", color=colors.grid, linewidth=1.0, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(colors.axis)
    ax.tick_params(axis="both", length=0, labelsize=9, labelcolor=colors.ink_muted)
