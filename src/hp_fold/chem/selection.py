from collections.abc import Sequence
from itertools import zip_longest

from hp_fold.chem.records import Fragment


def select_fragments(fragments: Sequence[Fragment], count: int) -> list[Fragment]:
    """Pick up to `count` fragments with distinct H/P strings.

    Fragments sharing an H/P string are the same QAOA problem, so only the first is
    kept. Within each structure the most hydrophobic ones come first, and the
    structures are interleaved so one entry can't fill the whole selection.
    """
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    unique: dict[str, Fragment] = {}
    for fragment in fragments:
        unique.setdefault(fragment.hp, fragment)

    by_structure: dict[str, list[Fragment]] = {}
    for fragment in unique.values():
        by_structure.setdefault(fragment.structure_id, []).append(fragment)
    ranked = [
        sorted(group, key=lambda f: -f.hp.count("H")) for group in by_structure.values()
    ]
    interleaved = [f for row in zip_longest(*ranked) for f in row if f is not None]
    return interleaved[:count]
