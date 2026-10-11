import pytest

from hp_fold.chem.records import Fragment
from hp_fold.chem.selection import select_fragments


def frag(structure, start, hp):
    return Fragment(structure, "A", start, "A" * len(hp), hp)


def test_drops_duplicate_hp_strings():
    fragments = [frag("X", 1, "HPPHHPPH"), frag("X", 9, "HPPHHPPH")]
    assert [f.start for f in select_fragments(fragments, 5)] == [1]


def test_most_hydrophobic_first_within_a_structure():
    fragments = [frag("X", 1, "HPPHPPPP"), frag("X", 2, "HHHHPPPP")]
    assert [f.start for f in select_fragments(fragments, 2)] == [2, 1]


def test_interleaves_structures():
    fragments = [
        frag("X", 1, "HHHHHHHH"),
        frag("X", 2, "HHHHHHHP"),
        frag("Y", 1, "HPPHPPPH"),
    ]
    assert [f.structure_id for f in select_fragments(fragments, 2)] == ["X", "Y"]


def test_rejects_non_positive_count():
    with pytest.raises(ValueError):
        select_fragments([], 0)
