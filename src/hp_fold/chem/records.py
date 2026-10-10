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

    structure_id: str
    chain_id: str
    residues: tuple[ResidueRecord, ...]

    def __len__(self) -> int:
        return len(self.residues)


@dataclass(frozen=True, slots=True)
class Fragment:
    """A contiguous window of a chain, as an amino-acid string and its H/P string.

    `start`: seq_id of the first residue
    `aa`: one-letter aminoacid sequence
    `hp`: H/P string, must have same length as `aa`
    """

    structure_id: str
    chain_id: str
    start: int
    aa: str
    hp: str

    @property
    def end(self) -> int:
        return self.start + len(self.aa) - 1

    @property
    def label(self) -> str:
        return f"{self.structure_id}:{self.chain_id}:{self.start}-{self.end}"
