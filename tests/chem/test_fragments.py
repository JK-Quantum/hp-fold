from pathlib import Path

import pytest
from Bio.Data.PDBData import protein_letters_1to3

from hp_fold.chem.alphabet import to_sequence
from hp_fold.chem.fragments import FragmentExtractor, can_form_contact
from hp_fold.chem.hydrophobicity import FixedSetClassifier, to_hp
from hp_fold.chem.loader import load_chains
from hp_fold.chem.records import ChainRecord, ResidueRecord

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


def make_chain(aa, seq_ids=None):
    seq_ids = seq_ids or range(1, len(aa) + 1)
    residues = tuple(
        ResidueRecord(protein_letters_1to3[a].upper(), i, (0.0, 0.0, 0.0))
        for a, i in zip(aa, seq_ids, strict=True)
    )
    return ChainRecord("T", "A", residues)


@pytest.mark.parametrize(
    ("hp", "expected"),
    [
        ("HPPH", True),
        ("HPH", False),
        ("HPPPH", False),
        ("HPPPPH", True),
        ("PPPPPP", False),
        ("HHH", False),
        ("HHHH", True),
    ],
)
def test_can_form_contact(hp, expected):
    assert can_form_contact(hp) is expected


def test_extracts_windows_that_can_contact():
    extractor = FragmentExtractor(FixedSetClassifier(), length=4, min_hydrophobic=2)
    fragments = extractor.extract(make_chain("AGGAGGAA"))
    assert [(f.start, f.aa, f.hp) for f in fragments] == [
        (1, "AGGA", "HPPH"),
        (4, "AGGA", "HPPH"),
    ]
    assert fragments[0].label == "T:A:1-4"


def test_windows_do_not_cross_gaps():
    chain = make_chain("GGAGGAGG", seq_ids=[1, 2, 3, 4, 6, 7, 8, 9])
    extractor = FragmentExtractor(FixedSetClassifier(), length=4, min_hydrophobic=2)
    assert extractor.extract(chain) == []


def test_min_hydrophobic_defaults_to_three():
    extractor = FragmentExtractor(FixedSetClassifier(), length=4)
    assert extractor.extract(make_chain("AGGA")) == []


def test_rejects_windows_too_short_to_make_a_contact():
    with pytest.raises(ValueError, match="length"):
        FragmentExtractor(FixedSetClassifier(), length=3)


def test_real_fragments_are_consistent_with_the_chain():
    (chain,) = load_chains(RAW / "1UBQ.cif")
    classifier = FixedSetClassifier()
    full = to_sequence(chain)
    fragments = FragmentExtractor(classifier, length=8).extract(chain)

    assert fragments
    for f in fragments:
        assert full[f.start - 1 : f.end] == f.aa
        assert f.hp == to_hp(f.aa, classifier)
        assert can_form_contact(f.hp)
        assert f.end - f.start == 7
