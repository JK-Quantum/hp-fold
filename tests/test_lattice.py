import pytest

from hp_fold.lattice import fold_of, n_qubits, positions, x_of


@pytest.mark.parametrize(("n_residues", "expected"), [(3, 2), (4, 4), (5, 6), (10, 16)])
def test_n_qubits(n_residues, expected):
    assert n_qubits(n_residues) == expected


@pytest.mark.parametrize(
    ("fold", "x"),
    [
        ("RRRR", 0),
        ("RUR", 1),
        ("RRU", 4),
        ("RDRU", 19),
        ("RURD", 49),
    ],
)
def test_x_of_pins_the_bit_layout(fold, x):
    assert x_of(fold) == x
    assert fold_of(x, len(fold) + 1) == fold


@pytest.mark.parametrize("n_residues", [4, 5, 6])
def test_every_integer_round_trips(n_residues):
    for x in range(2 ** n_qubits(n_residues)):
        fold = fold_of(x, n_residues)
        assert fold[0] == "R"
        assert len(fold) == n_residues - 1
        assert x_of(fold) == x


def test_positions_of_a_square_hairpin():
    assert positions("RURD").tolist() == [[0, 0], [1, 0], [1, 1], [2, 1], [2, 0]]


def test_positions_of_a_straight_chain():
    assert positions("RRR").tolist() == [[0, 0], [1, 0], [2, 0], [3, 0]]


@pytest.mark.parametrize("fold", ["", "R", "URD", "RXR"])
def test_rejects_malformed_folds(fold):
    with pytest.raises(ValueError):
        x_of(fold)
    with pytest.raises(ValueError):
        positions(fold)


@pytest.mark.parametrize(("x", "n_residues"), [(-1, 5), (64, 5), (0, 2)])
def test_fold_of_rejects_out_of_range(x, n_residues):
    with pytest.raises(ValueError):
        fold_of(x, n_residues)
