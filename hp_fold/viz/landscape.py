"""Heat map of <C> over (gamma, beta): where the optimizer has to search."""

from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from hp_fold.analysis.landscape import Landscape
from hp_fold.viz.theme import Theme, get_theme

_RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")


def plot_landscape(
    landscape: Landscape,
    reference_energies: dict[str, float] | None = None,
    found: tuple[float, float] | None = None,
    title: str = "QAOA energy landscape",
    theme: str = "light",
) -> Figure:
    """<C> against gamma and beta. `found` is the optimizer's (gamma, beta), if known.

    `reference_energies` maps a name to an energy (e.g. the random-guess mean and ground
    energy).
    """
    colors = get_theme(theme)
    fig = Figure(figsize=(7.4, 5.4), facecolor=colors.surface, layout="constrained")
    ax = fig.subplots()
    ax.set_facecolor(colors.surface)

    values = landscape.values
    cmap = LinearSegmentedColormap.from_list("energy", _ramp(theme))
    mesh = ax.pcolormesh(
        landscape.gammas,
        landscape.betas,
        values,
        cmap=cmap,
        shading="gouraud",
        vmin=float(values.min()),
        vmax=float(values.max()),
        rasterized=True,
    )

    best_gamma, best_beta, best_value = landscape.best()
    _mark(ax, best_gamma, best_beta, "o", colors, f"grid minimum  {best_value:.2f}")
    if found is not None:
        _mark(ax, found[0], found[1], "D", colors, "optimizer's result")

    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0, labelsize=9, labelcolor=colors.ink_muted)
    ax.set_xlabel("gamma (cost-layer angle)", color=colors.ink_secondary, fontsize=9)
    ax.set_ylabel("beta (mixer angle)", color=colors.ink_secondary, fontsize=9)
    ax.set_xlim(float(landscape.gammas[0]), float(landscape.gammas[-1]))
    ax.set_ylim(float(landscape.betas[0]), float(landscape.betas[-1]))

    bar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.05)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=9, labelcolor=colors.ink_muted)
    bar.set_label(
        "expected energy <C>  (lower is better)",
        color=colors.ink_secondary,
        fontsize=9,
    )
    bar.ax.invert_yaxis()
    for name, energy in (reference_energies or {}).items():
        if values.min() <= energy <= values.max():
            bar.ax.axhline(energy, color=colors.ink, linewidth=1)
            bar.ax.annotate(
                name,
                (0, energy),
                xycoords=("axes fraction", "data"),
                xytext=(-6, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                color=colors.ink_secondary,
                fontsize=8,
            )

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=len(labels),
        frameon=False,
        fontsize=9,
        labelcolor=colors.ink_secondary,
    )
    fig.suptitle(
        title,
        x=0.0,
        ha="left",
        color=colors.ink,
        fontsize=12,
        fontweight="bold",
    )
    return fig


def _ramp(theme: str) -> list[str]:
    return list(reversed(_RAMP)) if theme == "light" else list(_RAMP)


def _mark(
    ax: Axes,
    gamma: float,
    beta: float,
    marker: str,
    colors: Theme,
    label: str,
) -> None:
    ax.plot(
        [gamma],
        [beta],
        marker=marker,
        markersize=9,
        linestyle="none",
        label=label,
        markerfacecolor=colors.alert,
        markeredgecolor=colors.surface,
        markeredgewidth=2,
        zorder=5,
    )
