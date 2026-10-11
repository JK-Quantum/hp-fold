import numpy as np
import pytest

from hp_fold import HPProblem
from hp_fold.analysis.store import load_result, save_result
from hp_fold.experiments.algorithms import resolve
from hp_fold.quantum.adapters import run_qaoa, run_qaoa_cvar

FAST = {"p": 1, "restarts": 1, "maxiter": 15, "shots": 64}


@pytest.fixture(scope="module")
def problem():
    return HPProblem.from_sequence("PHPPH")


def test_qaoa_result_follows_the_standard_and_survives_saving(problem, tmp_path):
    result = run_qaoa(problem, FAST, seed=2)
    assert resolve("qaoa") is run_qaoa
    assert (result.algorithm, result.seq, result.seed) == ("qaoa", "PHPPH", 2)
    assert result.probs.shape == (2**problem.n_qubits,)
    assert result.shots == 64
    assert dict(result.config) == FAST
    training = result.extras["training_evaluations"]
    assert training > 0
    assert result.evaluations == training + result.shots
    assert len(result.extras["gammas"]) == len(result.extras["betas"]) == 1
    assert len(result.extras["objective_history"]) == training

    loaded = load_result(save_result(problem, result, tmp_path))
    assert np.array_equal(loaded.probs, result.probs)


def test_qaoa_is_deterministic_given_the_seed(problem):
    a, b, c = (run_qaoa(problem, FAST, seed=s) for s in (1, 1, 2))
    assert np.array_equal(a.probs, b.probs)
    assert not np.array_equal(a.probs, c.probs)


def test_cvar_result_has_its_own_name_and_records_alpha(problem):
    params = {**FAST, "cvar_alpha": 0.25}
    result = run_qaoa_cvar(problem, params, seed=2)
    assert resolve("qaoa_cvar") is run_qaoa_cvar
    assert result.algorithm == "qaoa_cvar"
    assert dict(result.config) == params
    assert result.evaluations == result.extras["training_evaluations"] + 64
    assert result.extras["objective"] >= problem.ground_energy


@pytest.mark.parametrize(
    "params, message",
    [
        (FAST, "missing param 'cvar_alpha'"),
        ({**FAST, "cvar_alpha": 0}, "cvar_alpha must be in"),
        ({**FAST, "cvar_alpha": 1.5}, "cvar_alpha must be in"),
        ({**FAST, "cvar_alpha": "0.1"}, "cvar_alpha must be a number"),
    ],
)
def test_cvar_rejects_bad_alpha(problem, params, message):
    with pytest.raises(ValueError, match=message):
        run_qaoa_cvar(problem, params, seed=0)


def test_plain_qaoa_does_not_accept_alpha(problem):
    with pytest.raises(ValueError, match="unknown.*cvar_alpha"):
        run_qaoa(problem, {**FAST, "cvar_alpha": 0.1}, seed=0)


@pytest.mark.parametrize(
    "params, message",
    [
        ({}, "missing param 'p'"),
        ({"p": 1, "depth": 2}, "unknown param.*depth"),
        ({"p": 0}, "p must be a positive integer"),
        ({"p": True}, "p must be a positive integer"),
        ({"p": 1, "shots": 0}, "shots must be a positive integer"),
        ({"p": 1, "backend": "gpu"}, "backend must be one of"),
    ],
)
def test_qaoa_rejects_bad_params(problem, params, message):
    with pytest.raises(ValueError, match=message):
        run_qaoa(problem, params, seed=0)
