import numpy as np
import pytest

from hp_fold import HPProblem
from hp_fold.analysis.result import RunResult


@pytest.fixture(scope="module")
def problem():
    return HPProblem.from_sequence("PHPPH")


def uniform(n_qubits):
    return np.full(2**n_qubits, 1 / 2**n_qubits)


def make(probs, **overrides):
    fields = dict(algorithm="test", seq="PHPPH", shots=None, evaluations=1)
    return RunResult(probs=probs, **{**fields, **overrides})


def test_n_qubits_is_log2_of_probs_length():
    assert make(uniform(6)).n_qubits == 6


def test_rejects_unnormalised_probs():
    with pytest.raises(ValueError, match="sum"):
        make(2 * uniform(6))


def test_rejects_negative_probs():
    p = np.zeros(8)
    p[0], p[1] = -0.5, 1.5
    with pytest.raises(ValueError, match="negative"):
        make(p)


def test_rejects_length_that_is_not_a_power_of_two():
    with pytest.raises(ValueError, match="power of two"):
        make(np.full(6, 1 / 6))


def test_rejects_non_positive_shots():
    with pytest.raises(ValueError, match="shots"):
        make(uniform(6), shots=0)


def test_probs_are_copied_and_read_only():
    p = uniform(6)
    result = make(p)
    p[0] = 1.0
    assert result.probs[0] == pytest.approx(1 / 64)
    with pytest.raises(ValueError):
        result.probs[0] = 0.0


def test_from_samples_builds_empirical_distribution(problem):
    result = RunResult.from_samples("random", problem, [3, 3, 5, 0], evaluations=4)
    assert result.probs[3] == pytest.approx(0.5)
    assert result.probs[5] == pytest.approx(0.25)
    assert result.probs[0] == pytest.approx(0.25)
    assert result.probs.sum() == pytest.approx(1.0)
    assert (result.algorithm, result.seq, result.shots) == ("random", "PHPPH", 4)
    assert result.evaluations == 4


def test_from_samples_rejects_out_of_range_x(problem):
    with pytest.raises(ValueError, match="out of range"):
        RunResult.from_samples("random", problem, [64], evaluations=1)


def test_from_samples_rejects_empty(problem):
    with pytest.raises(ValueError, match="empty"):
        RunResult.from_samples("random", problem, [], evaluations=0)
