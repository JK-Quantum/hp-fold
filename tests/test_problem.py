import itertools

import pytest

from hp_fold.lattice import x_of
from hp_fold.problem import HPProblem, default_penalty


def test_contract_attributes_of_the_worked_example():
    p = HPProblem.from_sequence("PHPPH")
    assert p.seq == "PHPPH"
    assert p.n_residues == 5
    assert p.n_qubits == 6
    assert p.cost.shape == (64,)
    assert p.cost.dtype == float
    assert p.valid.shape == (64,)
    assert p.valid.dtype == bool


def test_ground_states_of_the_worked_example():
    p = HPProblem.from_sequence("PHPPH")
    assert p.ground_energy == -1
    assert p.ground_states.tolist() == [19, 36, 44, 49]
    assert [p.fold(x) for x in p.ground_states] == ["RDRU", "RRUL", "RRDL", "RURD"]


def test_number_of_valid_folds():
    assert HPProblem.from_sequence("PHPPH").valid.sum() == 25


def test_costs_of_specific_folds():
    p = HPProblem.from_sequence("PHPPH", penalty=10)
    assert p.cost[x_of("RURD")] == -1
    assert p.cost[x_of("RRRR")] == 0
    assert p.cost[x_of("RULD")] == 9


def test_decode_returns_lattice_positions():
    p = HPProblem.from_sequence("PHPPH")
    assert p.decode(49).tolist() == [[0, 0], [1, 0], [1, 1], [2, 1], [2, 0]]


def test_default_penalty_is_used_unless_overridden():
    assert HPProblem.from_sequence("PHPPH").penalty == default_penalty(5)
    assert HPProblem.from_sequence("PHPPH", penalty=10).penalty == 10


def test_too_small_a_penalty_is_rejected():
    with pytest.raises(ValueError, match="penalty"):
        HPProblem.from_sequence("PHPPH", penalty=0)


@pytest.mark.parametrize("n_residues", [4, 5, 6])
def test_default_penalty_keeps_every_optimum_valid(n_residues):
    for bits in itertools.product("HP", repeat=n_residues):
        HPProblem.from_sequence("".join(bits))


def test_arrays_are_read_only():
    p = HPProblem.from_sequence("PHPPH")
    with pytest.raises(ValueError):
        p.cost[0] = 1.0
    with pytest.raises(ValueError):
        p.valid[0] = True


@pytest.mark.parametrize("seq", ["PXPPH", "HH"])
def test_rejects_bad_sequences(seq):
    with pytest.raises(ValueError):
        HPProblem.from_sequence(seq)
