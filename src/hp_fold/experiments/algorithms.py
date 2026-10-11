"""Turn the `algorithm` string of a config into a function that produces a RunResult.

Every runner has the same signature:  run(problem, params, seed) -> RunResult
An algorithm is named either by a built-in ("random", "annealing") or by a dotted path
"package.module:function", which lets other code plug in without editing this file.
"""

import importlib
from collections.abc import Callable, Mapping
from typing import Any, cast

import numpy as np

from hp_fold.analysis.result import RunResult
from hp_fold.baselines.annealing import SimulatedAnnealing
from hp_fold.baselines.base import Sampler
from hp_fold.baselines.random_sampling import RandomSampler
from hp_fold.folding.problem import HPProblem

Runner = Callable[[HPProblem, Mapping[str, Any], int], RunResult]

_ANNEALING_OPTIONS = ("n_steps", "t_start", "t_end")


def run_random(problem: HPProblem, params: Mapping[str, Any], seed: int) -> RunResult:
    _check_params("random", params, required=("n_samples",))
    return _sample("random", RandomSampler(), problem, params, seed)


def run_annealing(
    problem: HPProblem,
    params: Mapping[str, Any],
    seed: int,
) -> RunResult:
    _check_params(
        "annealing",
        params,
        required=("n_samples",),
        optional=_ANNEALING_OPTIONS,
    )
    options = {k: params[k] for k in _ANNEALING_OPTIONS if k in params}
    return _sample("annealing", SimulatedAnnealing(**options), problem, params, seed)


BUILTINS: dict[str, Runner] = {"random": run_random, "annealing": run_annealing}


def resolve(name: str) -> Runner:
    """Return the runner for a built-in name or a "module:function" path."""
    if ":" not in name:
        if name not in BUILTINS:
            raise ValueError(
                f"unknown algorithm {name!r}; "
                f"built-ins are {', '.join(sorted(BUILTINS))}, "
                "or use 'package.module:function'"
            )
        return BUILTINS[name]

    module_name, _, attribute = name.partition(":")
    if not module_name or not attribute or ":" in attribute:
        raise ValueError(f"algorithm {name!r} must look like 'package.module:function'")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(
            f"cannot import module {module_name!r} for algorithm {name!r}: {exc}"
        ) from exc
    try:
        runner: Any = getattr(module, attribute)
    except AttributeError as exc:
        raise ValueError(
            f"module {module_name!r} has no attribute {attribute!r}"
        ) from exc
    if not callable(runner):
        raise ValueError(f"{name!r} is not callable")
    return cast(Runner, runner)


def _sample(
    name: str,
    sampler: Sampler,
    problem: HPProblem,
    params: Mapping[str, Any],
    seed: int,
) -> RunResult:
    n_samples = params["n_samples"]
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError(
            f"{name}: n_samples must be a positive integer, got {n_samples!r}"
        )
    xs = sampler.sample(problem, n_samples, np.random.default_rng(seed))
    return RunResult.from_samples(
        name,
        problem,
        xs,
        evaluations=n_samples * sampler.evaluations_per_sample,
        seed=seed,
        config=dict(params),
    )


def _check_params(
    name: str,
    params: Mapping[str, Any],
    required: tuple[str, ...],
    optional: tuple[str, ...] = (),
) -> None:
    unknown = sorted(set(params) - set(required) - set(optional))
    if unknown:
        raise ValueError(f"{name}: unknown param(s) {', '.join(unknown)}")
    for key in required:
        if key not in params:
            raise ValueError(f"{name}: missing param {key!r}")
