"""Decide which HP sequence an experiment runs on: given or picked from a CIF."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from hp_fold.chem.fragments import FragmentExtractor
from hp_fold.chem.hydrophobicity import FixedSetClassifier
from hp_fold.chem.loader import load_chains
from hp_fold.chem.records import ChainRecord
from hp_fold.experiments.config import ExperimentConfig

ChainLoader = Callable[[Path], Sequence[ChainRecord]]


@dataclass(frozen=True, slots=True)
class ResolvedSequence:
    sequence: str
    origin: str


def resolve_sequence(
    config: ExperimentConfig,
    loader: ChainLoader = load_chains,
) -> ResolvedSequence:
    if config.sequence is not None:
        return ResolvedSequence(config.sequence, "given in config")
    if config.cif is None:
        raise ValueError("config has neither a sequence nor a cif")

    extractor = FragmentExtractor(FixedSetClassifier(), length=config.fragment_length)
    fragments = [f for chain in loader(config.cif) for f in extractor.extract(chain)]
    if config.fragment_pick >= len(fragments):
        raise ValueError(
            f"problem.pick = {config.fragment_pick} is out of range: "
            f"{config.cif.name} has "
            f"{len(fragments)} candidate fragment(s) of length {config.fragment_length}"
        )
    fragment = fragments[config.fragment_pick]
    return ResolvedSequence(fragment.hp, f"{fragment.label} ({fragment.aa})")
