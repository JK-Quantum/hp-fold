from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from hp_fold.quantum.experiment import ExperimentResult, QAOAExperiment

DPI = 150


class QAOAReport:
    """Builds and saves the standard figures for one experiment result."""

    def __init__(self, experiment: QAOAExperiment, result: ExperimentResult) -> None:
        self._experiment = experiment
        self._result = result
        self._problem = result.problem

    def save_all(self, out_dir: Path) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        figures = {
            f"distribution_p{r.p}.png": self.distribution(r.p)
            for r in self._result.depths
        }
        figures["convergence.png"] = self.convergence()
        figures["metrics_vs_depth.png"] = self.metrics_vs_depth()
        figures["landscape_p1.png"] = self.landscape()
        paths = []
        for name, fig in figures.items():
            path = out_dir / name
            fig.savefig(path, dpi=DPI)
            paths.append(path)
        return paths

    def distribution(self, p: int) -> Figure:
        problem, probs = self._problem, self._result.probabilities[p]
        order = np.argsort(problem.cost, kind="stable")
        ground = set(problem.ground_states.tolist())
        colors = [
            "tab:green"
            if x in ground
            else "tab:blue"
            if problem.valid[x]
            else "tab:red"
            for x in order.tolist()
        ]
        fig = Figure(figsize=(9, 4), layout="tight")
        ax = fig.subplots()
        ax.bar(np.arange(order.size), probs[order], color=colors, width=1.0)
        ax.axhline(1 / problem.cost.size, color="k", ls="--", lw=0.8, label="uniform")
        ax.set(
            xlabel="folds, sorted by cost (low to high)",
            ylabel="probability",
            title=f"{problem.seq}: output distribution, p={p} "
            "(green = ground, blue = valid, red = overlap)",
        )
        ax.legend()
        return fig

    def convergence(self) -> Figure:
        fig = Figure(figsize=(7, 4), layout="tight")
        ax = fig.subplots()
        for r in self._result.depths:
            ax.plot(r.history, label=f"p={r.p}")
        ax.axhline(
            self._problem.ground_energy,
            color="k",
            ls="--",
            lw=0.8,
            label="ground energy",
        )
        ax.set(
            xlabel="evaluation",
            ylabel="best objective so far",
            title="Optimizer convergence",
        )
        ax.legend()
        return fig

    def metrics_vs_depth(self) -> Figure:
        depths = self._result.depths
        ps = [r.p for r in depths]
        fig = Figure(figsize=(12, 3.8), layout="tight")
        axes = fig.subplots(1, 3)
        axes[0].plot(ps, [r.expectation for r in depths], "o-")
        axes[0].set(xlabel="p", ylabel="<C>", title="Expected cost")
        axes[1].plot(ps, [r.p_ground for r in depths], "o-", label="P(ground)")
        axes[1].plot(ps, [r.p_valid for r in depths], "s-", label="P(valid)")
        axes[1].set(xlabel="p", ylabel="probability", title="Success probabilities")
        axes[1].legend()
        axes[2].plot(ps, [r.improvement_over_random for r in depths], "o-")
        axes[2].set(
            xlabel="p", ylabel="(mean - <C>)/(mean - ground)", title="Improvement"
        )
        for ax in axes:
            ax.set_xticks(ps)
        return fig

    def landscape(self, grid: int = 60) -> Figure:
        gammas, betas, z = self._experiment.landscape(grid)
        fig = Figure(figsize=(6, 4.5), layout="tight")
        ax = fig.subplots()
        mesh = ax.pcolormesh(gammas, betas, z, shading="auto", cmap="viridis")
        fig.colorbar(mesh, ax=ax, label="<C>")
        ax.set(
            xlabel="gamma", ylabel="beta", title=f"{self._problem.seq}: p=1 landscape"
        )
        return fig
