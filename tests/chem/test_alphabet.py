from pathlib import Path

import pytest

from hp_fold.chem.alphabet import UnknownResidueError, one_letter, to_sequence
from hp_fold.chem.loader import load_chains
from hp_fold.chem.records import ChainRecord, ResidueRecord

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


def make_chain(*names):
    residues = tuple(
        ResidueRecord(name, i, (0.0, 0.0, 0.0)) for i, name in enumerate(names, start=1)
    )
    return ChainRecord("T", "A", residues)


def test_one_letter_standard_residues():
    assert one_letter("ALA") == "A"
    assert one_letter("TRP") == "W"


def test_one_letter_error_names_the_residue():
    with pytest.raises(UnknownResidueError, match="XYZ"):
        one_letter("XYZ")


def test_to_sequence():
    assert to_sequence(make_chain("THR", "CYS", "GLY")) == "TCG"


def test_to_sequence_error_names_residue_and_location():
    with pytest.raises(UnknownResidueError, match=r"XYZ.*T:A:2"):
        to_sequence(make_chain("THR", "XYZ", "GLY"))


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("1CRN.cif", "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"),
        (
            "1UBQ.cif",
            "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
        ),
    ],
)
def test_real_structures_match_known_sequences(filename, expected):
    (chain,) = load_chains(RAW / filename)
    assert to_sequence(chain) == expected
