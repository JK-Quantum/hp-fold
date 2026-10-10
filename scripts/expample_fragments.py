"""Example: mmCIF file -> chains -> H/P strings -> fragments that can form contacts.

```bash
$ uv run scripts/example_fragments.py data/raw/1UBQ.cif 8
```
"""

import sys

from hp_fold.chem.alphabet import to_sequence
from hp_fold.chem.fragments import FragmentExtractor
from hp_fold.chem.hydrophobicity import FixedSetClassifier, to_hp
from hp_fold.chem.loader import load_chains

path, length = sys.argv[1], int(sys.argv[2])
classifier = FixedSetClassifier()
extractor = FragmentExtractor(classifier, length=length)

for chain in load_chains(path):
    aa = to_sequence(chain)
    fragments = extractor.extract(chain)
    print(f"{chain.structure_id}:{chain.chain_id}  {len(chain)} residues")
    print(f"  {aa}")
    print(f"  {to_hp(aa, classifier)}")
    print(f"  {len(fragments)} fragments of length {length}; first five:")
    for f in fragments[:5]:
        print(f"    {f.label}  {f.aa}  {f.hp}")
