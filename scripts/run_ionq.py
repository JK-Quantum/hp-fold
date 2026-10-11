import argparse
import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from hp_fold.analysis.backend_report import BackendReport
from hp_fold.folding.problem import HPProblem
from hp_fold.quantum.backend_runner import BackendRunner
from hp_fold.quantum.experiment import QAOAExperiment
from hp_fold.quantum.qaoa import QAOACircuit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "data" / "results" / "ionq"


def find_params(args: argparse.Namespace, problem: HPProblem) -> np.ndarray:
    if args.params is not None:
        params = np.array(args.params, dtype=float)
        if params.size != 2 * args.p:
            raise SystemExit(f"--params needs {2 * args.p} values for p={args.p}")
        return params
    experiment = QAOAExperiment(
        problem,
        cvar_alpha=args.cvar_alpha,
        restarts=args.restarts,
        seed=args.seed,
    )
    return np.array(experiment.run(args.p).depths[-1].params)


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description="Run QAOA on an IonQ backend.")
    parser.add_argument("sequence", nargs="?", default="PHPPH", help="HP string")
    parser.add_argument("--p", type=int, default=1, help="QAOA depth")
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--backend", default="ionq_simulator", help="IonQ backend name")
    parser.add_argument("--gateset", choices=["qis", "native"], default="qis")
    parser.add_argument("--noise-model", default=None, help="IonQ simulator noise")
    parser.add_argument("--local", action="store_true", help="Qiskit BasicSimulator")
    parser.add_argument(
        "--transpile-only", action="store_true", help="print circuit size, send nothing"
    )
    parser.add_argument("--params", type=float, nargs="+", help="gammas then betas")
    parser.add_argument("--cvar-alpha", type=float, default=None)
    parser.add_argument("--restarts", type=int, default=10, help="NumPy optimizer")
    parser.add_argument("--penalty", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0, help="NumPy optimizer seed")
    parser.add_argument(
        "--sampler-seed",
        type=int,
        default=None,
        help="IonQ simulator sampling seed (positive integer); omit for unseeded",
    )
    parser.add_argument("--label", default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.p < 1:
        parser.error("--p must be >= 1")

    seq = args.sequence.upper()
    problem = HPProblem.from_sequence(seq, args.penalty)
    circuit = QAOACircuit(problem, args.p)
    params = find_params(args, problem)
    print(f"{seq}: {problem.n_qubits} qubits, p={args.p}, params={params.round(4)}")

    if args.local:
        from qiskit.providers.basic_provider import BasicSimulator

        runner = BackendRunner(
            circuit,
            BasicSimulator(),
            run_options={"seed_simulator": args.seed},
        )
    else:
        runner = BackendRunner.from_ionq(
            circuit,
            backend_name=args.backend,
            gateset=args.gateset,
            noise_model=args.noise_model,
            sampler_seed=args.sampler_seed,
        )

    stats = runner.stats_of(runner.transpile(params))
    print(
        f"transpiled for {runner.backend_name}: depth={stats.depth}, "
        f"two-qubit gates={stats.n_two_qubit_gates}, gates={stats.n_gates}"
    )
    if args.transpile_only:
        return

    result = runner.run(params, args.shots, seed=args.seed)
    label = args.label or "_".join(
        part
        for part in (
            "local" if args.local else args.backend,
            args.noise_model,
            f"p{args.p}",
        )
        if part
    )
    BackendReport(problem, result).save(args.out / seq, label)

    summary = {
        k: v
        for k, v in result.to_dict().items()
        if k not in ("counts", "params", "stats")
    }
    print(json.dumps(summary, indent=2))
    print(f"\nrandom P(ground) = {problem.ground_states.size / problem.cost.size:.4f}")
    print(f"Saved to {args.out / seq / label}.json / .png")


if __name__ == "__main__":
    main()
