from collections.abc import Mapping, Sequence

from hp_fold.chem.records import ChainRecord, ResidueRecord

_COLUMNS = (
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
)


def parse_chains(
    cif: Mapping[str, Sequence[str]],
    structure_id: str,
) -> list[ChainRecord]:
    """Extract one CA-only record per polymer residue from an mmCIF dict.

    `cif` is the column-oriented mapping returned by Biopython's MMCIF2Dict.

    Rules: ATOM rows only, atom name CA only, lowest model number only, the
    first altloc listed wins, residues sorted by seq_id, chains sorted by id.
    """
    columns = [cif[f"_atom_site.{name}"] for name in _COLUMNS]
    rows = [
        dict(zip(_COLUMNS, values, strict=True))
        for values in zip(*columns, strict=True)
    ]

    ca_rows = [
        r for r in rows if r["group_PDB"] == "ATOM" and r["label_atom_id"] == "CA"
    ]
    if not ca_rows:
        raise ValueError(f"No polymer CA atoms found in {structure_id!r}")

    first_model = min(int(r["pdbx_PDB_model_num"]) for r in ca_rows)

    residues: dict[tuple[str, int], ResidueRecord] = {}
    for r in ca_rows:
        if int(r["pdbx_PDB_model_num"]) != first_model:
            continue
        key = (r["label_asym_id"], int(r["label_seq_id"]))

        if key in residues:
            continue

        residues[key] = ResidueRecord(
            name=r["label_comp_id"],
            seq_id=key[1],
            ca=(float(r["Cartn_x"]), float(r["Cartn_y"]), float(r["Cartn_z"])),
        )

    by_chain: dict[str, list[ResidueRecord]] = {}
    for (chain_id, _), residue in sorted(residues.items()):
        by_chain.setdefault(chain_id, []).append(residue)

    return [
        ChainRecord(structure_id, chain_id, tuple(chain_residues))
        for chain_id, chain_residues in by_chain.items()
    ]
