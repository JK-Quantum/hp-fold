from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from hp_fold.chem.cif import parse_chains
from hp_fold.chem.records import ChainRecord

CifReader = Callable[[str], Mapping[str, Sequence[str]]]


def load_chains(
    path: str | Path,
    structure_id: str | None = None,
    reader: CifReader = MMCIF2Dict,
) -> list[ChainRecord]:
    """Read an mmCIF file and return its polymer chains as CA-only records."""
    path = Path(path)
    return parse_chains(reader(str(path)), structure_id or path.stem)
