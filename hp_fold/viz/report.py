"""Choose which saved runs feed which figure.

Optional fields a run may carry:

    config["p"]                 circuit depth (selects the deepest run, feeds the sweep)
    extras["objective_history"] optimizer trace (convergence figure)
    extras["landscape_gammas"]  1-D angles, strictly increasing   -+
    extras["landscape_betas"]   1-D angles, strictly increasing    +- energy landscape
    extras["landscape_values"]  <C>, shape (len(betas), len(gammas)) -+
    extras["landscape_found"]   [gamma, beta] the optimizer reached (optional marker)
"""

import json
from collections.abc import Sequence

import numpy as np

from hp_fold.analysis.landscape import Landscape
from hp_fold.analysis.result import RunResult

LANDSCAPE_KEYS = ("landscape_gammas", "landscape_betas", "landscape_values")
FOUND_KEY = "landscape_found"


def depth_of(result: RunResult) -> int | None:
    """The circuit depth `p` from the run's config, if it set one."""
    depth = result.config.get("p")
    if isinstance(depth, bool) or not isinstance(depth, int):
        return None
    return depth


def label_of(result: RunResult) -> str:
    """Panel/legend name: the algorithm, plus the depth and CVaR level when set."""
    depth = depth_of(result)
    label = result.algorithm if depth is None else f"{result.algorithm}, p = {depth}"
    alpha = result.config.get("cvar_alpha")
    return label if alpha is None else f"{label}, alpha = {alpha}"


def newest_per_config(results: Sequence[RunResult]) -> list[RunResult]:
    """Drop repeated executions of the same (algorithm, config, seed); the newest wins.

    `results` are oldest first. Re-running a deterministic config must not count as a
    new repeat, or the averages in the sweep would silently weight it twice.
    """
    kept: dict[tuple[str, str, int | None], RunResult] = {}
    for result in results:
        key = (
            result.algorithm,
            json.dumps(dict(result.config), sort_keys=True),
            result.seed,
        )
        kept.pop(key, None)
        kept[key] = result
    return list(kept.values())


def representatives(results: Sequence[RunResult]) -> list[RunResult]:
    """One run per algorithm: the deepest circuit, and among equals the newest.

    `results` are oldest first; the output keeps the order in which algorithms appeared.
    Runs without a depth (classical baselines) count as depth 0.
    """
    chosen: dict[str, RunResult] = {}
    for result in results:
        current = chosen.get(result.algorithm)
        if current is None or (depth_of(result) or 0) >= (depth_of(current) or 0):
            chosen[result.algorithm] = result
    return list(chosen.values())


def landscape_of(
    results: Sequence[RunResult],
) -> tuple[RunResult, Landscape, tuple[float, float] | None] | None:
    """The newest run that carries a landscape:
    (run, landscape, optimizer's angles or None).

    Raises ValueError if a run has some but not all of the landscape fields,
    or invalid arrays.
    """
    for result in reversed(results):
        present = [key for key in LANDSCAPE_KEYS if key in result.extras]
        if not present:
            continue
        if len(present) != len(LANDSCAPE_KEYS):
            missing = [key for key in LANDSCAPE_KEYS if key not in result.extras]
            raise ValueError(
                f"{label_of(result)}: landscape is missing extras {missing}"
            )
        gammas, betas, values = (
            np.asarray(result.extras[key]) for key in LANDSCAPE_KEYS
        )
        landscape = Landscape(gammas, betas, values)
        return result, landscape, _found(result, landscape)
    return None


def _found(result: RunResult, landscape: Landscape) -> tuple[float, float] | None:
    raw = result.extras.get(FOUND_KEY)
    if raw is None:
        return None
    angles = np.asarray(raw, dtype=float)
    if angles.shape != (2,):
        raise ValueError(
            f"{label_of(result)}: extras[{FOUND_KEY!r}] must be [gamma, beta]"
        )
    gamma, beta = float(angles[0]), float(angles[1])
    inside = (
        landscape.gammas[0] <= gamma <= landscape.gammas[-1]
        and landscape.betas[0] <= beta <= landscape.betas[-1]
    )
    return (gamma, beta) if inside else None
