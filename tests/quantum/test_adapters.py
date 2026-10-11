import numpy as np
import pytest

from hp_fold import HPProblem
from hp_fold.analysis.store import load_result, save_result
from hp_fold.experiments.algorithms import resolve
from hp_fold.quantum.adapters import run_qaoa

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
    assert result.extras["objective_history"]

    loaded = load_result(save_result(problem, result, tmp_path))
    assert np.array_equal(loaded.probs, result.probs)


def test_qaoa_is_deterministic_given_the_seed(problem):
    a, b, c = (run_qaoa(problem, FAST, seed=s) for s in (1, 1, 2))
    assert np.array_equal(a.probs, b.probs)
    assert not np.array_equal(a.probs, c.probs)


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
