from pathlib import Path

import pytest

from hp_fold.chem.alphabet import AMINO_ACIDS, to_sequence
from hp_fold.chem.hydrophobicity import (
    KYTE_DOOLITTLE,
    FixedSetClassifier,
    KyteDoolittleClassifier,
    to_hp,
)
from hp_fold.chem.loader import load_chains

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

CRAMBIN_HP = "PPHHPPHHHPPPHPHHPHPPPPPHHHHPPPPHHHHPPHPHPPPPHP"


def test_fixed_set_splits_all_twenty_residues():
    classifier = FixedSetClassifier()
    assert to_hp("AVLIMFWC", classifier) == "H" * 8
    assert to_hp("GPSTNQYHKRDE", classifier) == "P" * 12


def test_kyte_doolittle_table_covers_exactly_the_twenty_residues():
    assert set(KYTE_DOOLITTLE) == AMINO_ACIDS


def test_scales_disagree_only_on_tryptophan():
    fixed, kd = FixedSetClassifier(), KyteDoolittleClassifier()
    differing = {
        aa for aa in AMINO_ACIDS if fixed.is_hydrophobic(aa) != kd.is_hydrophobic(aa)
    }
    assert differing == {"W"}


def test_kyte_doolittle_threshold_moves_the_split():
    assert not KyteDoolittleClassifier().is_hydrophobic("W")
    assert KyteDoolittleClassifier(threshold=-1.0).is_hydrophobic("W")


def test_to_hp_rejects_non_standard_letters():
    with pytest.raises(ValueError, match=r"'X'.*position 1"):
        to_hp("AXA", FixedSetClassifier())


def test_fixed_set_rejects_unknown_letters_at_construction():
    with pytest.raises(ValueError, match="Z"):
        FixedSetClassifier("AVZ")


def test_crambin_hp_string():
    (chain,) = load_chains(RAW / "1CRN.cif")
    assert to_hp(to_sequence(chain), FixedSetClassifier()) == CRAMBIN_HP
