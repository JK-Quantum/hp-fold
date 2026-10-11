"""Run QAOA on HP fragments taken from the mmCIF files in data/raw/."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from hp_fold.chem.fragments import FragmentExtractor
from hp_fold.chem.hydrophobicity import (
    FixedSetClassifier,
    HPClassifier,
    KyteDoolittleClassifier,
)
from hp_fold.chem.loader import load_chains
from hp_fold.chem.records import Fragment
from hp_fold.chem.selection import select_fragments

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
DEFAULT_RAW = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUT = PROJECT_ROOT / "data" / "results" / "cif"

CLASSIFIERS: dict[str, tuple[str, HPClassifier]] = {
    "kd": ("Kyte-Doolittle hydropathy > 0", KyteDoolittleClassifier()),
    "fixed": ("fixed set AVLIMFWC", FixedSetClassifier()),
}
METHODS = ("numpy", "qiskit", "cvar", "ionq")


def pick_fragments(
    raw: Path, classifier: HPClassifier, length: int, count: int
) -> list[Fragment]:
    extractor = FragmentExtractor(classifier, length)
    fragments = []
    for path in sorted(raw.glob("*.cif")):
        for chain in load_chains(path):
            fragments.extend(extractor.extract(chain))
    return select_fragments(fragments, count)


def command_for(
    args: argparse.Namespace, hp: str, out: Path, extra: list[str]
) -> list[str]:
    if args.method == "ionq":
        return [
            sys.executable,
            str(SCRIPTS / "run_ionq.py"),
            hp,
            *extra,
            "--out",
            str(out),
        ]
    backend = args.cvar_backend if args.method == "cvar" else args.method
    cmd = [sys.executable, str(SCRIPTS / "run_qaoa.py"), hp, "--backend", backend]
    if args.method == "cvar":
        cmd += ["--cvar-alpha", str(args.cvar_alpha)]
    return [*cmd, *extra, "--out", str(out)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run QAOA on fragments from data/raw/*.cif."
    )
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--length", type=int, default=8, help="fragment length N")
    parser.add_argument("--count", type=int, default=4, help="fragments to run")
    parser.add_argument("--classifier", choices=sorted(CLASSIFIERS), default="kd")
    parser.add_argument("--cvar-alpha", type=float, default=0.25)
    parser.add_argument("--cvar-backend", choices=["numpy", "qiskit"], default="numpy")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--list", action="store_true", help="print the selected fragments and exit"
    )
    args, extra = parser.parse_known_args()

    mapping, classifier = CLASSIFIERS[args.classifier]
    fragments = pick_fragments(args.raw, classifier, args.length, args.count)
    if not fragments:
        raise SystemExit(f"no usable length-{args.length} fragments in {args.raw}")
    for f in fragments:
        print(f"{f.label:<16} {f.aa}  {f.hp}")
    if args.list:
        return

    name = f"cvar_{args.cvar_alpha}" if args.method == "cvar" else args.method
    out = args.out / name
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": args.method,
        "hydrophobicity": mapping,
        "length": args.length,
        "fragments": [
            {"label": f.label, "aa": f.aa, "hp": f.hp, "start": f.start}
            for f in fragments
        ],
        "extra_args": extra,
    }
    (out / "fragments.json").write_text(json.dumps(manifest, indent=2))

    for f in fragments:
        print(f"\n=== {f.label} ({f.hp}) via {name} ===", flush=True)
        subprocess.run(command_for(args, f.hp, out, extra), check=True)


if __name__ == "__main__":
    main()
