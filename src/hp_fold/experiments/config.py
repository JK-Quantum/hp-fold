"""Experiment configuration: a TOML file, validated into frozen dataclasses."""

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT = "data/results/runs"

_TOP_KEYS = {"seed", "output", "problem", "runs"}
_PROBLEM_KEYS = {"sequence", "cif", "length", "pick", "penalty"}
DEFAULT_FRAGMENT_LENGTH = 6
_RUN_KEYS = {"algorithm", "repeats", "params"}


@dataclass(frozen=True, slots=True)
class RunSpec:
    algorithm: str
    params: Mapping[str, Any]
    repeats: int = 1


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """The problem is either a given HP `sequence` or a fragment picked from a file."""

    sequence: str | None
    penalty: float | None
    seed: int
    output: Path
    runs: tuple[RunSpec, ...]
    cif: Path | None = None
    fragment_length: int = DEFAULT_FRAGMENT_LENGTH
    fragment_pick: int = 0


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

    has_sequence, has_cif = "sequence" in problem, "cif" in problem
    if has_sequence == has_cif:
        raise ValueError(
            "[problem] needs either 'sequence' or 'cif' (exactly one of them)"
        )
    if has_sequence and ({"length", "pick"} & set(problem)):
        raise ValueError(
            "[problem] 'length' and 'pick' are only valid together with 'cif'"
        )

    output = Path(data.get("output", DEFAULT_OUTPUT))
    return ExperimentConfig(
        sequence=_sequence(problem["sequence"]) if has_sequence else None,
        penalty=_penalty(problem.get("penalty")),
        seed=_integer("seed", data.get("seed", 0), minimum=None),
        output=_resolve(output, base_dir),
        runs=tuple(_run(i, r) for i, r in enumerate(runs)),
        cif=_cif(problem["cif"], base_dir) if has_cif else None,
        fragment_length=_integer(
            "problem.length",
            problem.get("length", DEFAULT_FRAGMENT_LENGTH),
            minimum=4,
        ),
        fragment_pick=_integer("problem.pick", problem.get("pick", 0), minimum=0),
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


def _resolve(path: Path, base_dir: Path) -> Path:
    return path if path.is_absolute() else base_dir / path


def _cif(value: Any, base_dir: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"problem.cif must be a non-empty path string, got {value!r}")
    return _resolve(Path(value), base_dir)


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
