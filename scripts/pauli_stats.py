import argparse
from pathlib import Path

from hp_fold.analysis.pauli_sweep import PauliSweep

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "results" / "pauli"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pauli term counts of the HP cost.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[5, 6, 7, 8])
    parser.add_argument("--samples", type=int, default=10, help="random sequences/N")
    parser.add_argument("--tol", type=float, default=1e-9, help="pruning threshold")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    sweep = PauliSweep(samples=args.samples, tol=args.tol, seed=args.seed)
    report = sweep.run(args.sizes)
    for n, entry in report.items():
        for row in entry["sequences"]:
            print(
                f"N={n} {row['sequence']}: {row['n_terms']} terms, max weight "
                f"{row['max_weight']}, weights {row['weight_histogram']}"
            )
    print(f"\nSaved to {sweep.save(report, args.out)}")


if __name__ == "__main__":
    main()
