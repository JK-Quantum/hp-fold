"""Persist a RunResult: metadata and scalars as JSON, arrays in a sibling .npz."""

import json
import math
import re
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from hp_fold.analysis.metrics import compute_metrics
from hp_fold.analysis.result import RunResult
from hp_fold.folding.problem import HPProblem

SCHEMA_VERSION = 1
_PACKAGES = ("numpy", "qiskit", "qiskit-aer", "qiskit-ionq")


def save_result(problem: HPProblem, result: RunResult, directory: str | Path) -> Path:
    """Write <run_id>.json and <run_id>.npz into `directory`; return the JSON path."""
    metrics = compute_metrics(problem, result)
    config = _jsonable(dict(result.config), "config")
    arrays: dict[str, Any] = {"probs": result.probs}
    extras: dict[str, Any] = {}
    for key, value in result.extras.items():
        if isinstance(value, np.ndarray):
            arrays[f"extras.{key}"] = value
        else:
            extras[key] = _jsonable(value, f"extras[{key!r}]")

    created = datetime.now(UTC)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    run_id = _unique_run_id(directory, created, result)
    record = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": created.isoformat(),
        "provenance": _provenance(),
        "problem": {
            "seq": problem.seq,
            "penalty": problem.penalty,
            "n_qubits": problem.n_qubits,
            "ground_energy": problem.ground_energy,
        },
        "result": {
            "algorithm": result.algorithm,
            "seq": result.seq,
            "shots": result.shots,
            "evaluations": result.evaluations,
            "seed": result.seed,
            "config": config,
            "extras": extras,
        },
        "metrics": {
            k: None if math.isnan(v) else v for k, v in asdict(metrics).items()
        },
        "arrays_file": f"{run_id}.npz",
    }
    np.savez_compressed(directory / f"{run_id}.npz", **arrays)
    json_path = directory / f"{run_id}.json"
    json_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return json_path


def load_result(path: str | Path) -> RunResult:
    path = Path(path)
    record = json.loads(path.read_text())
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {record.get('schema_version')!r} in "
            f"{path.name}"
        )
    saved = record["result"]
    with np.load(path.with_name(record["arrays_file"])) as arrays:
        extras = dict(saved["extras"])
        for name in arrays.files:
            if name.startswith("extras."):
                extras[name.removeprefix("extras.")] = arrays[name]
        return RunResult(
            algorithm=saved["algorithm"],
            seq=saved["seq"],
            probs=arrays["probs"],
            shots=saved["shots"],
            evaluations=saved["evaluations"],
            seed=saved["seed"],
            config=saved["config"],
            extras=extras,
        )


def _jsonable(value: Any, where: str) -> Any:
    """Convert to plain JSON types (numpy scalars unwrapped, tuples become lists)."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool | int | str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{where}: non-finite float {value}")
        return value
    if isinstance(value, (list | tuple)):
        return [_jsonable(v, f"{where}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise TypeError(f"{where}: dict keys must be strings")
        return {k: _jsonable(v, f"{where}[{k!r}]") for k, v in value.items()}
    raise TypeError(f"{where}: {type(value).__name__} is not JSON-serializable")


def _unique_run_id(directory: Path, created: datetime, result: RunResult) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", f"{result.algorithm}-{result.seq}".lower()).strip(
        "-"
    )
    base = f"{created:%Y%m%dT%H%M%SZ}_{slug}"
    run_id, n = base, 1
    while (directory / f"{run_id}.json").exists():
        run_id, n = f"{base}-{n}", n + 1
    return run_id


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
            cwd=Path(__file__).parent,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _provenance() -> dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    dirty = None
    if commit:
        dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    return {
        "git_commit": commit or None,
        "git_dirty": dirty,
        "python": sys.version.split()[0],
        "packages": {name: _version(name) for name in _PACKAGES},
    }
