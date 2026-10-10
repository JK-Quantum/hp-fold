import pytest

from hp_fold.energy import count_contacts, count_overlaps, energy


def test_square_hairpin_has_one_contact():
    assert count_contacts("PHPPH", "RURD") == 1


def test_straight_chain_has_no_contacts():
    assert count_contacts("PHPPH", "RRRR") == 0


def test_chain_neighbours_are_not_contacts():
    assert count_contacts("HHHH", "RRR") == 0


def test_smallest_possible_contact_is_a_2x2_square():
    assert count_contacts("HHHH", "RUL") == 1


def test_polar_residues_never_make_contacts():
    assert count_contacts("PPPP", "RUL") == 0


@pytest.mark.parametrize(("fold", "expected"), [("RRRR", 0), ("RLR", 2), ("RULD", 1)])
def test_count_overlaps(fold, expected):
    assert count_overlaps(fold) == expected


def test_energy_of_known_folds():
    assert energy("PHPPH", "RURD", penalty=10) == -1
    assert energy("PHPPH", "RRRR", penalty=10) == 0


def test_penalty_makes_an_overlapping_fold_worse_than_a_straight_one():
    assert energy("PHPPH", "RULD", penalty=10) == 9
    assert energy("PHPPH", "RULD", penalty=10) > energy("PHPPH", "RRRR", penalty=10)


@pytest.mark.parametrize(("hp", "fold"), [("PHP", "RURD"), ("PXPPH", "RURD")])
def test_count_contacts_rejects_inconsistent_input(hp, fold):
    with pytest.raises(ValueError):
        count_contacts(hp, fold)
