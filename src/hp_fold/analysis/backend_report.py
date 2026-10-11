import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from hp_fold.folding.problem import HPProblem
from hp_fold.quantum.backend_runner import RunResult

DPI = 150


class BackendReport:
    """Saves the JSON and the ideal-vs-measured figure for one `RunResult`."""

    def __init__(self, problem: HPProblem, result: RunResult) -> None:
        self._problem = problem
        self._result = result

    def comparison(self) -> Figure:
        problem, r = self._problem, self._result
        order = np.argsort(problem.cost, kind="stable")
        x = np.arange(order.size)
        fig = Figure(figsize=(10, 4.2), layout="tight")
        ax = fig.subplots()
        ax.bar(x - 0.2, r.exact_probabilities[order], width=0.4, label="exact (ideal)")
        ax.bar(x + 0.2, r.probabilities[order], width=0.4, label=f"{r.backend}")
        ground = np.flatnonzero(np.isin(order, problem.ground_states))
        for g in ground:
            ax.axvspan(g - 0.5, g + 0.5, color="tab:green", alpha=0.15, lw=0)
        ax.set(
            xlabel="folds, sorted by cost (green band = ground states)",
            ylabel="probability",
            title=(
                f"{problem.seq}: {r.backend}, {r.shots} shots, "
                f"TV distance {r.tv_distance:.3f} "
                f"(shot-noise floor {r.tv_distance_shot_noise:.3f})"
            ),
        )
        ax.legend()
        return fig

    def save(self, out_dir: Path, label: str) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{label}.json"
        json_path.write_text(json.dumps(self._result.to_dict(), indent=2))
        png_path = out_dir / f"{label}.png"
        self.comparison().savefig(png_path, dpi=DPI)
        return [json_path, png_path]
