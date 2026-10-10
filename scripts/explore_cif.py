"""Thrownaway script: EDA on what's inside an mmCIF file; residues per chain"""

import sys

from Bio.Data import PDBData
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

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

cif = MMCIF2Dict(sys.argv[1])
rows = [
    dict(zip(COLS, vals, strict=True))
    for vals in zip(*(cif[f"_atom_site.{c}"] for c in COLS), strict=True)
]

print("models:", sorted({r["pdbx_PDB_model_num"] for r in rows}))
print("groups:", sorted({r["group_PDB"] for r in rows}))
print("alt ids:", sorted({r["label_alt_id"] for r in rows}))
print("CA rows with an altloc:")
for r in rows:
    if r["label_atom_id"] == "CA" and r["label_alt_id"] != ".":
        print(
            "  ",
            r["label_asym_id"],
            r["label_seq_id"],
            r["label_comp_id"],
            r["label_alt_id"],
        )

ca_rows = [
    r
    for r in rows
    if r["group_PDB"] == "ATOM"
    and r["label_atom_id"] == "CA"
    and r["pdbx_PDB_model_num"] == "1"
]

residues = {}
for r in ca_rows:
    residues.setdefault((r["label_asym_id"], int(r["label_seq_id"])), r)

aa = PDBData.protein_letters_3to1
print("\n3to1 sample:", list(aa.items())[:3])

chains = {}
for (chain, seq_id), r in sorted(residues.items()):
    chains.setdefault(chain, []).append((seq_id, r["label_comp_id"]))

canonical = "".join(cif["_entity_poly.pdbx_seq_one_letter_code_can"][0].split())
for chain, items in chains.items():
    ids = [i for i, _ in items]
    seq = "".join(aa[name] for _, name in items)
    print(f"\nchain {chain}: {len(seq)} residues\n{seq}")
    print("  contiguous:", ids == list(range(ids[0], ids[0] + len(ids))))
    print("  == entity_poly:", seq == canonical)
