"""Experiment configuration.

A TOML file, validated into frozen dataclasses.
"""

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT = "data/results/runs"

_TOP_KEYS = {"seed", "output", "problem", "runs"}
_PROBLEM_KEYS = {"sequence", "penalty"}
_RUN_KEYS = {"algorithm", "repeats", "params"}


@dataclass(frozen=True, slots=True)
class RunSpec:
    algorithm: str
    params: Mapping[str, Any]
    repeats: int = 1


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    sequence: str
    penalty: float | None
    seed: int
    output: Path
    runs: tuple[RunSpec, ...]


def load_config(
    path: str | Path, base_dir: str | Path | None = None
) -> ExperimentConfig:
    """Read a TOML file.

    Relative paths inside resolve against base_dir (default: its folder).
    """
    path = Path(path)
    with path.open("rb") as f:
        data = tomllib.load(f)
    return parse_config(data, Path(base_dir) if base_dir is not None else path.parent)


def parse_config(data: Mapping[str, Any], base_dir: Path) -> ExperimentConfig:
    _reject_unknown("top level", data, _TOP_KEYS)

    problem = data.get("problem")
    if not isinstance(problem, Mapping):
        raise ValueError("missing [problem] section")
    _reject_unknown("[problem]", problem, _PROBLEM_KEYS)

    runs = data.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("config needs at least one [[runs]] entry")

    output = Path(data.get("output", DEFAULT_OUTPUT))
    return ExperimentConfig(
        sequence=_sequence(problem.get("sequence")),
        penalty=_penalty(problem.get("penalty")),
        seed=_integer("seed", data.get("seed", 0), minimum=None),
        output=output if output.is_absolute() else base_dir / output,
        runs=tuple(_run(i, r) for i, r in enumerate(runs)),
    )


def _reject_unknown(section: str, data: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"unknown key(s) in {section}: {', '.join(unknown)}")


def _integer(name: str, value: Any, minimum: int | None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def _sequence(value: Any) -> str:
    if not isinstance(value, str) or not value or set(value) - {"H", "P"}:
        raise ValueError(f"problem.sequence must be a string of H and P, got {value!r}")
    return value


def _penalty(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int | float)) or value <= 0:
        raise ValueError(f"problem.penalty must be a positive number, got {value!r}")
    return float(value)


def _run(index: int, raw: Any) -> RunSpec:
    where = f"[[runs]] #{index + 1}"
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where} must be a table")
    _reject_unknown(where, raw, _RUN_KEYS)
    algorithm = raw.get("algorithm")
    if not isinstance(algorithm, str) or not algorithm:
        raise ValueError(f"{where}: algorithm must be a non-empty string")
    params = raw.get("params", {})
    if not isinstance(params, Mapping):
        raise ValueError(f"{where}: params must be a table")
    return RunSpec(
        algorithm=algorithm,
        params=dict(params),
        repeats=_integer(f"{where}: repeats", raw.get("repeats", 1), minimum=1),
    )
