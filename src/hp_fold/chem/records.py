from dataclasses import dataclass

Coord = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class ResidueRecord:
    """A polymer residue, represented by its alpha-Carbon."""

    name: str
    seq_id: int
    ca: Coord


@dataclass(frozen=True, slots=True)
class ChainRecord:
    """A polymer chain.

    Residues are ordered by `ResidueRecord.seq_id`.
    """

    structures_id: str
    chain_id: str
    residues: tuple[ResidueRecord, ...]

    def __len__(self) -> int:
        return len(self.residues)
