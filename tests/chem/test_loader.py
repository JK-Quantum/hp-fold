from pathlib import Path

import pytest

from hp_fold.chem.loader import load_chains

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


@pytest.mark.parametrize(
    ("filename", "n_residues", "first_two", "last"),
    [
        ("1CRN.cif", 46, ("THR", "THR"), "ASN"),
        ("1UBQ.cif", 76, ("MET", "GLN"), "GLY"),
    ],
)
def test_loads_real_structure(filename, n_residues, first_two, last):
    (chain,) = load_chains(RAW / filename)
    names = [r.name for r in chain.residues]

    assert chain.structure_id == Path(filename).stem
    assert chain.chain_id == "A"
    assert len(chain) == n_residues
    assert [r.seq_id for r in chain.residues] == list(range(1, n_residues + 1))
    assert tuple(names[:2]) == first_two
    assert names[-1] == last


def test_structure_id_can_be_overridden():
    (chain,) = load_chains(RAW / "1CRN.cif", structure_id="crambin")
    assert chain.structure_id == "crambin"
