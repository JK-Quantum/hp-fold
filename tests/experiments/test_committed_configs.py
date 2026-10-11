from pathlib import Path

import pytest

from hp_fold.experiments.algorithms import resolve
from hp_fold.experiments.config import load_config
from hp_fold.experiments.problem_source import resolve_sequence

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = sorted((ROOT / "configs").glob("*.toml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: path.stem)
def test_committed_config_parses_and_its_algorithms_resolve(path):
    config = load_config(path, base_dir=ROOT)
    assert resolve_sequence(config).sequence
    for spec in config.runs:
        resolve(spec.algorithm)
