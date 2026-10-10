"""Thrownaway script: EDA on what's inside an mmCIF file"""

import sys
from collections import Counter

from Bio.PDB.MMCIF2Dict import MMCIF2Dict

cif = MMCIF2Dict(sys.argv[1])

print(f"{len(cif)} tags\n")

categories = Counter(tag.split(".")[0] for tag in cif)
for name, n_tags in sorted(categories.items()):
    print(f"{name:40s} {n_tags} tags")

atom = {tag: col for tag, col in cif.items() if tag.startswith("_atom_site.")}
n_rows = len(next(iter(atom.values())))
print(f"\n_atom_site: {len(atom)} columns x {n_rows} rows")
print(list(atom))

for i in range(3):
    print({tag.split(".")[1]: col[i] for tag, col in atom.items()})

print("\nentity_poly:", cif.get("_entity_poly.pdbx_seq_one_letter_code_can"))
