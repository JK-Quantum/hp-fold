"""A fold drawn on the square lattice: the chain, its H/P residues and H-H contacts."""

import numpy as np
from matplotlib.figure import Figure

from hp_fold.folding.problem import HPProblem
from hp_fold.viz.theme import get_theme

_NODE_SIZE = 520
_MARGIN = 0.7


def plot_fold(problem: HPProblem, x: int, theme: str = "light") -> Figure:
    """Fold `x` of `problem`: H residues filled, P residues grey, contacts dashed."""
    colors = get_theme(theme)
    positions = problem.decode(x)
    xs, ys = positions[:, 0], positions[:, 1]

    fig = Figure(figsize=(4.6, 4.6), facecolor=colors.surface, layout="constrained")
    ax = fig.subplots()
    ax.set_facecolor(colors.surface)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.plot(xs, ys, color=colors.ink_muted, linewidth=2.5, zorder=1)
    contacts = _contacts(problem.seq, positions)
    for i, j in contacts:
        ax.plot(
            [xs[i], xs[j]],
            [ys[i], ys[j]],
            color=colors.alert,
            linewidth=2,
            linestyle=(0, (2, 2)),
            zorder=2,
        )
    for index, (kind, px, py) in enumerate(zip(problem.seq, xs, ys, strict=True)):
        ax.scatter(
            [px],
            [py],
            s=_NODE_SIZE,
            color=colors.accent if kind == "H" else colors.neutral,
            edgecolors=colors.surface,
            linewidths=2,
            zorder=3,
        )
        ax.text(
            px,
            py,
            str(index + 1),
            ha="center",
            va="center",
            color=colors.surface,
            fontsize=9,
            fontweight="bold",
            zorder=4,
        )
    ax.set_xlim(xs.min() - _MARGIN, xs.max() + _MARGIN)
    ax.set_ylim(ys.min() - _MARGIN, ys.max() + _MARGIN)
    fig.suptitle(
        f"{problem.seq}: {problem.fold(x)}, energy {problem.cost[x]:g}, "
        f"{len(contacts)} H-H contact(s)",
        color=colors.ink,
        fontsize=11,
        fontweight="bold",
    )
    return fig


def _contacts(seq: str, positions: np.ndarray) -> list[tuple[int, int]]:
    """Pairs of non-consecutive H residues on neighbouring lattice sites."""
    return [
        (i, j)
        for i in range(len(seq))
        for j in range(i + 2, len(seq))
        if seq[i] == seq[j] == "H" and np.abs(positions[i] - positions[j]).sum() == 1
    ]
