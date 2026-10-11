from typing import Protocol

from hp_fold.chem.alphabet import AMINO_ACIDS

# Hydropathy index of Kyte & Doolittle (1982), J. Mol. Biol.
KYTE_DOOLITTLE: dict[str, float] = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}

DEFAULT_HYDROPHOBIC = "AVLIMFWC"


class HPClassifier(Protocol):
    def is_hydrophobic(self, aa: str) -> bool: ...


class FixedSetClassifier:
    def __init__(self, hydrophobic: str = DEFAULT_HYDROPHOBIC) -> None:
        unknown = set(hydrophobic) - AMINO_ACIDS
        if unknown:
            raise ValueError(f"Not standard amino-acid letters: {sorted(unknown)}")
        self._hydrophobic = frozenset(hydrophobic)

    def is_hydrophobic(self, aa: str) -> bool:
        return aa in self._hydrophobic


class KyteDoolittleClassifier:
    def __init__(self, threshold: float = 0.0) -> None:
        self._threshold = threshold

    def is_hydrophobic(self, aa: str) -> bool:
        return KYTE_DOOLITTLE[aa] > self._threshold


def to_hp(sequence: str, classifier: HPClassifier) -> str:
    """Map a one-letter amino-acid string to an H/P string."""
    out = []
    for i, aa in enumerate(sequence):
        if aa not in AMINO_ACIDS:
            raise ValueError(f"Non-standard amino-acid letter {aa!r} at position {i}")
        out.append("H" if classifier.is_hydrophobic(aa) else "P")
    return "".join(out)
