import argparse
import json
from pathlib import Path

from hp_fold.analysis.qaoa_report import QAOAReport
from hp_fold.folding.problem import HPProblem
from hp_fold.quantum.experiment import BACKENDS, QAOAExperiment

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "results" / "qaoa"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QAOA on an HP sequence.")
    parser.add_argument("sequence", nargs="?", default="PHPPH", help="HP string")
    parser.add_argument("--depth", type=int, default=3, help="max QAOA depth p")
    parser.add_argument("--restarts", type=int, default=6, help="random starts per p")
    parser.add_argument("--maxiter", type=int, default=300, help="COBYLA iterations")
    parser.add_argument("--shots", type=int, default=1024, help="shots for sampling")
    parser.add_argument("--penalty", type=float, default=None, help="overlap penalty")
    parser.add_argument(
        "--cvar-alpha",
        type=float,
        default=None,
        help="minimize CVaR_alpha instead of <C>",
    )
    parser.add_argument("--backend", choices=sorted(BACKENDS), default="numpy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.depth < 1:
        parser.error("--depth must be >= 1")

    seq = args.sequence.upper()
    problem = HPProblem.from_sequence(seq, args.penalty)
    experiment = QAOAExperiment(
        problem,
        backend=args.backend,
        cvar_alpha=args.cvar_alpha,
        restarts=args.restarts,
        maxiter=args.maxiter,
        shots=args.shots,
        seed=args.seed,
    )
    result = experiment.run(args.depth)

    out = args.out / seq
    QAOAReport(experiment, result).save_all(out)
    metrics = {
        **result.to_dict(),
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    for r in result.depths:
        print(
            f"p={r.p}: objective={r.objective_value:.4f}  <C>={r.expectation:.4f}  "
            f"P(ground)={r.p_ground:.4f}  P(valid)={r.p_valid:.4f}  "
            f"improvement={r.improvement_over_random:.3f}"
        )
    print(f"\nProblem: {json.dumps(metrics['problem'], indent=2)}")
    print(f"\nSaved metrics and figures to {out}")


if __name__ == "__main__":
    main()
