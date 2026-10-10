"""HP-model energy of a fold on the square lattice.

E = -(number of H-H contacts) + penalty * (number of overlaps)

A contact is a pair of H residues that are lattice neighbours but not neighbours
along the chain (Dill, 1985).
"""

import numpy as np

from hp_fold.lattice import positions


def _check(hp: str, fold: str) -> None:
    if len(hp) != len(fold) + 1:
        raise ValueError(f"hp has {len(hp)} residues but fold has {len(fold)} moves")
    invalid = set(hp) - {"H", "P"}
    if invalid:
        raise ValueError(f"hp may contain only 'H' and 'P', got {sorted(invalid)}")


def count_overlaps(fold: str) -> int:
    sites = positions(fold).tolist()
    return len(sites) - len({tuple(site) for site in sites})


def count_contacts(hp: str, fold: str) -> int:
    _check(hp, fold)
    pos = positions(fold)
    is_h = np.array([residue == "H" for residue in hp])
    distance = np.abs(pos[:, None, :] - pos[None, :, :]).sum(axis=-1)
    non_bonded = np.triu(
        np.ones_like(distance, dtype=bool),
        k=3,
    )
    touching = non_bonded & (distance == 1) & is_h[:, None] & is_h[None, :]
    return int(touching.sum())


def energy(hp: str, fold: str, penalty: float) -> float:
    return -count_contacts(hp, fold) + penalty * count_overlaps(fold)
