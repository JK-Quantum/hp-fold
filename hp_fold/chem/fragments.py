from collections.abc import Iterator

from hp_fold.chem.alphabet import to_sequence
from hp_fold.chem.hydrophobicity import HPClassifier, to_hp
from hp_fold.chem.records import ChainRecord, Fragment


def can_form_contact(hp: str) -> bool:
    """True if two H residues are an odd distance >= 3 apart.

    On a square lattice, each step flips the parity of x + y, so two residues can be
    lattice neighbours only if their chain distance is odd.

    Distance 1 is a chain bond, so contacts need distance 3, 5, 7, ...
    """
    h = [i for i, c in enumerate(hp) if c == "H"]
    return any(j - i >= 3 and (j - i) % 2 == 1 for i in h for j in h)


def _contiguous_runs(chain: ChainRecord) -> Iterator[ChainRecord]:
    run = []
    for residue in chain.residues:
        if run and residue.seq_id != run[-1].seq_id + 1:
            yield ChainRecord(chain.structure_id, chain.chain_id, tuple(run))
            run = []
        run.append(residue)
    if run:
        yield ChainRecord(chain.structure_id, chain.chain_id, tuple(run))


class FragmentExtractor:
    def __init__(
        self,
        classifier: HPClassifier,
        length: int,
        min_hydrophobic: int = 3,
    ) -> None:
        if length < 4:
            raise ValueError(
                "length must be >= 4: shorter windows cannot form a contact"
            )
        self._classifier = classifier
        self._length = length
        self._min_hydrophobic = min_hydrophobic

    def extract(self, chain: ChainRecord) -> list[Fragment]:
        fragments = []
        for run in _contiguous_runs(chain):
            aa = to_sequence(run)
            hp = to_hp(aa, self._classifier)
            for i in range(len(aa) - self._length + 1):
                window = hp[i : i + self._length]
                if window.count("H") >= self._min_hydrophobic and can_form_contact(
                    window
                ):
                    fragments.append(
                        Fragment(
                            chain.structure_id,
                            chain.chain_id,
                            run.residues[i].seq_id,
                            aa[i : i + self._length],
                            window,
                        )
                    )
        return fragments
