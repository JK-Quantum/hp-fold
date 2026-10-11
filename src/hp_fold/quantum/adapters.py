"""Adapters that run the QAOA circuits under the standard runner signature.

Every adapter is f(problem, params, seed) -> RunResult (see experiments/algorithms.py).
"""

from collections.abc import Mapping
from typing import Any

import numpy as np

from hp_fold.analysis.result import RunResult
from hp_fold.folding.problem import HPProblem
from hp_fold.quantum.experiment import BACKENDS, QAOAExperiment
from hp_fold.quantum.objectives import CVaRObjective, ExpectationObjective
from hp_fold.quantum.optimize import minimize_with_restarts

_DEFAULTS: dict[str, Any] = {
    "restarts": 6,
    "maxiter": 300,
    "shots": 1024,
    "backend": "numpy",
}


def run_qaoa(problem: HPProblem, params: Mapping[str, Any], seed: int) -> RunResult:
    """QAOA of depth `p`, trained on <C>; probs are empirical from `shots` samples."""
    return _run("qaoa", problem, params, seed, cvar=False)


def run_qaoa_cvar(
    problem: HPProblem, params: Mapping[str, Any], seed: int
) -> RunResult:
    """QAOA of depth `p`, trained on CVaR at level `cvar_alpha`."""
    return _run("qaoa_cvar", problem, params, seed, cvar=True)


def _run(
    name: str,
    problem: HPProblem,
    params: Mapping[str, Any],
    seed: int,
    cvar: bool,
) -> RunResult:
    options = _validated(name, params, cvar)
    circuit = BACKENDS[options["backend"]](problem, options["p"])
    objective = (
        CVaRObjective(circuit, options["cvar_alpha"])
        if cvar
        else ExpectationObjective(circuit)
    )
    rng = np.random.default_rng(seed)
    best = minimize_with_restarts(
        objective,
        restarts=options["restarts"],
        rng=rng,
        maxiter=options["maxiter"],
    )
    xs = circuit.sample(best.params, options["shots"], rng)
    gammas, betas = circuit.split_params(best.params)
    extras = {
        "training_evaluations": best.n_evaluations,
        "objective_history": best.history,
        "gammas": gammas.tolist(),
        "betas": betas.tolist(),
        "objective": best.value,
    }
    if "landscape_grid" in options:
        extras |= _landscape_extras(problem, options["landscape_grid"], gammas, betas)
    return RunResult.from_samples(
        name,
        problem,
        xs,
        evaluations=best.n_evaluations + options["shots"],
        seed=seed,
        config=dict(params),
        extras=extras,
    )


def _landscape_extras(
    problem: HPProblem, grid: int, gammas: np.ndarray, betas: np.ndarray
) -> dict[str, Any]:
    """<C> over the (gamma, beta) grid of a depth-1 circuit; not counted as budget."""
    gamma_axis, beta_axis, values = QAOAExperiment(problem).landscape(grid)
    return {
        "landscape_gammas": gamma_axis.tolist(),
        "landscape_betas": beta_axis.tolist(),
        "landscape_values": values.tolist(),
        "landscape_found": [float(gammas[0]), float(betas[0])],
    }


def _validated(name: str, params: Mapping[str, Any], cvar: bool) -> dict[str, Any]:
    """The params with defaults filled in; unknown or invalid ones raise ValueError."""
    required = ("p", "cvar_alpha") if cvar else ("p",)
    allowed = {*required, *_DEFAULTS, "landscape_grid"}
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(f"{name}: unknown param(s) {', '.join(unknown)}")
    for key in required:
        if key not in params:
            raise ValueError(f"{name}: missing param {key!r}")
    options = {**_DEFAULTS, **params}
    if cvar:
        alpha = options["cvar_alpha"]
        if isinstance(alpha, bool) or not isinstance(alpha, int | float):
            raise ValueError(f"{name}: cvar_alpha must be a number, got {alpha!r}")
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"{name}: cvar_alpha must be in (0, 1], got {alpha!r}")
    for key in ("p", "restarts", "maxiter", "shots"):
        value = options[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name}: {key} must be a positive integer, got {value!r}")
    if "landscape_grid" in options:
        grid = options["landscape_grid"]
        if isinstance(grid, bool) or not isinstance(grid, int) or grid < 2:
            raise ValueError(f"{name}: landscape_grid must be an integer >= 2")
        if options["p"] != 1:
            raise ValueError(f"{name}: landscape_grid needs p = 1")
    if options["backend"] not in BACKENDS:
        raise ValueError(
            f"{name}: backend must be one of {sorted(BACKENDS)}, "
            f"got {options['backend']!r}"
        )
    return options
