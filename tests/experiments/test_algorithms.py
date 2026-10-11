import os

import numpy as np
import pytest

from hp_fold import HPProblem
from hp_fold.analysis.metrics import compute_metrics
from hp_fold.analysis.result import RunResult
from hp_fold.experiments.algorithms import resolve, run_annealing, run_random


@pytest.fixture(scope="module")
def problem():
    return HPProblem.from_sequence("PHPPH")


def test_builtin_names_resolve_to_their_runners():
    assert resolve("random") is run_random
    assert resolve("annealing") is run_annealing


def test_dotted_path_resolves_to_the_named_function():
    assert resolve("os.path:join") is os.path.join


@pytest.mark.parametrize(
    "name, message",
    [
        ("quantum-magic", "unknown algorithm"),
        ("no_such_module_xyz:run", "cannot import"),
        ("os:no_such_function", "no attribute"),
        ("os:sep", "not callable"),
        (":run", "package.module:function"),
        ("os.path:", "package.module:function"),
        ("a:b:c", "package.module:function"),
    ],
)
def test_resolve_rejects_bad_names_with_a_clear_message(name, message):
    with pytest.raises(ValueError, match=message):
        resolve(name)


def test_random_runner_returns_a_labelled_result(problem):
    result = run_random(problem, {"n_samples": 40}, seed=3)
    assert isinstance(result, RunResult)
    assert (result.algorithm, result.seq, result.seed) == ("random", "PHPPH", 3)
    assert result.shots == 40
    assert result.evaluations == 40
    assert dict(result.config) == {"n_samples": 40}


def test_annealing_runner_counts_every_cost_evaluation(problem):
    result = run_annealing(problem, {"n_samples": 3, "n_steps": 50}, seed=0)
    assert result.algorithm == "annealing"
    assert result.shots == 3
    assert result.evaluations == 3 * 51
    assert dict(result.config) == {"n_samples": 3, "n_steps": 50}


@pytest.mark.parametrize("runner", [run_random, run_annealing])
def test_same_seed_gives_same_result_and_other_seed_differs(problem, runner):
    params = {"n_samples": 100}
    a, b, c = (runner(problem, params, seed=s) for s in (5, 5, 6))
    assert np.array_equal(a.probs, b.probs)
    assert not np.array_equal(a.probs, c.probs)


def test_annealing_beats_random_guessing_per_sample(problem):
    params = {"n_samples": 100}
    random = compute_metrics(problem, run_random(problem, params, seed=0))
    annealing = compute_metrics(problem, run_annealing(problem, params, seed=0))
    assert annealing.p_ground > 0.5 > random.p_ground


@pytest.mark.parametrize("runner", [run_random, run_annealing])
def test_missing_n_samples_is_rejected(problem, runner):
    with pytest.raises(ValueError, match="n_samples"):
        runner(problem, {}, seed=0)


@pytest.mark.parametrize("bad", [0, -5, 2.5, "10", True])
def test_invalid_n_samples_is_rejected(problem, bad):
    with pytest.raises(ValueError, match="n_samples"):
        run_random(problem, {"n_samples": bad}, seed=0)


@pytest.mark.parametrize("runner", [run_random, run_annealing])
def test_unknown_params_are_rejected_not_ignored(problem, runner):
    with pytest.raises(ValueError, match="unknown.*n_sample"):
        runner(problem, {"n_samples": 10, "n_sample": 20}, seed=0)


def test_random_does_not_accept_annealing_options(problem):
    with pytest.raises(ValueError, match="unknown.*n_steps"):
        run_random(problem, {"n_samples": 10, "n_steps": 5}, seed=0)
