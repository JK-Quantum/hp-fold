import pytest

from hp_fold.chem.cif import parse_chains
from hp_fold.chem.records import ChainRecord, ResidueRecord

COLS = [
    "group_PDB",
    "label_atom_id",
    "label_alt_id",
    "label_comp_id",
    "label_asym_id",
    "label_seq_id",
    "Cartn_x",
    "Cartn_y",
    "Cartn_z",
    "pdbx_PDB_model_num",
]


def row(
    comp,
    seq,
    *,
    atom="CA",
    group="ATOM",
    alt=".",
    chain="A",
    model="1",
    xyz=(0.0, 0.0, 0.0),
):
    return (group, atom, alt, comp, chain, seq, *xyz, model)


def atom_site(*rows) -> dict[str, list[str]]:
    """Column-oriented dict-like MMCIF2Dict returns.

    Every value is a string.
    """
    columns = zip(*rows, strict=True)
    return {
        f"_atom_site.{name}": [str(v) for v in col]
        for name, col in zip(COLS, columns, strict=True)
    }


def test_returns_ca_residues_in_order():
    cif = atom_site(
        row("THR", 1, atom="N"),
        row("THR", 1, xyz=(1.5, 2.5, 3.5)),
        row("THR", 1, atom="C"),
        row("CYS", 2, xyz=(4.0, 5.0, 6.0)),
    )
    (chain,) = parse_chains(cif, "TEST")
    assert chain == ChainRecord(
        "TEST",
        "A",
        (
            ResidueRecord("THR", 1, (1.5, 2.5, 3.5)),
            ResidueRecord("CYS", 2, (4.0, 5.0, 6.0)),
        ),
    )


def test_ignores_hetatm_calcium_named_ca():
    cif = atom_site(
        row("THR", 1),
        row("CA", ".", group="HETATM"),
    )
    (chain,) = parse_chains(cif, "TEST")
    assert [r.name for r in chain.residues] == ["THR"]


def test_keeps_only_the_first_model():
    cif = atom_site(
        row("THR", 1, xyz=(1.0, 1.0, 1.0), model="1"),
        row("THR", 1, xyz=(9.0, 9.0, 9.0), model="2"),
    )
    (chain,) = parse_chains(cif, "TEST")
    assert [r.ca for r in chain.residues] == [(1.0, 1.0, 1.0)]


def test_first_altloc_wins():
    cif = atom_site(
        row("ILE", 25, alt="A"),
        row("LEU", 25, alt="B", xyz=(1.0, 1.0, 1.0)),
    )
    (chain,) = parse_chains(cif, "TEST")
    assert [r.name for r in chain.residues] == ["ILE"]


def test_groups_by_chain_and_sorts_by_seq_id():
    cif = atom_site(
        row("CYS", 2, chain="A"),
        row("GLY", 1, chain="B"),
        row("THR", 1, chain="A"),
    )
    chains = parse_chains(cif, "TEST")
    assert [(c.chain_id, [r.seq_id for r in c.residues]) for c in chains] == [
        ("A", [1, 2]),
        ("B", [1]),
    ]


def test_raises_when_no_polymer_residues():
    cif = atom_site(row("HOH", ".", atom="O", group="HETATM"))
    with pytest.raises(ValueError, match="polymer"):
        parse_chains(cif, "TEST")
