"""End-to-end classical demo: (optional CIF) -> HP string -> HPProblem -> baselines.

```bash
$ uv run scripts/classical_demo.py --hp PHPPH
$ uv run scripts/classical_demo.py --cif data/raw/1UBQ.cif --length 6 --pick 0
```
"""

import argparse
from pathlib import Path

import numpy as np

from hp_fold import HPProblem
from hp_fold.analysis.metrics import compute_metrics, success_vs_shots
from hp_fold.analysis.result import RunResult
from hp_fold.baselines.annealing import SimulatedAnnealing
from hp_fold.baselines.base import Sampler
from hp_fold.baselines.random_sampling import RandomSampler
from hp_fold.chem.fragments import FragmentExtractor
from hp_fold.chem.hydrophobicity import FixedSetClassifier
from hp_fold.chem.loader import load_chains


def pick_sequence(args) -> str:
    if args.hp:
        return args.hp
    chains = load_chains(Path(args.cif))
    extractor = FragmentExtractor(FixedSetClassifier(), length=args.length)
    fragments = [f for c in chains for f in extractor.extract(c)]
    if not fragments:
        raise SystemExit("no fragment with enough hydrophobic residues")
    print(f"{len(fragments)} candidate fragments; picking #{args.pick}")
    frag = fragments[args.pick]
    print(f"fragment {frag.label}  aa={frag.aa}  hp={frag.hp}")
    return frag.hp


def draw(seq: str, positions: np.ndarray) -> str:
    """ASCII picture: residue letters, '-' '|' for bonds, ':' for H-H contacts."""
    pos = positions - positions.min(axis=0)
    w, h = pos.max(axis=0) * 2 + 1
    grid = [[" "] * w for _ in range(h)]
    for i, (x, y) in enumerate(pos):
        grid[(h - 1) - 2 * y][2 * x] = seq[i]
    for i in range(len(seq) - 1):
        (x0, y0), (x1, y1) = pos[i], pos[i + 1]
        grid[(h - 1) - (y0 + y1)][x0 + x1] = "-" if y0 == y1 else "|"
    for i in range(len(seq)):
        for j in range(i + 3, len(seq)):
            d = pos[j] - pos[i]
            if seq[i] == seq[j] == "H" and abs(d).sum() == 1:
                grid[(h - 1) - (pos[i][1] + pos[j][1])][pos[i][0] + pos[j][0]] = ":"
    return "\n".join("".join(r).rstrip() for r in grid)


def run_baselines(problem: HPProblem, n_samples: int, seed: int) -> list[RunResult]:
    """Every algorithm hands back the same record: a RunResult."""
    size = 2**problem.n_qubits
    results = [
        RunResult("random (exact)", problem.seq, np.full(size, 1 / size), None, 1)
    ]
    samplers: list[tuple[str, Sampler]] = [
        ("random", RandomSampler()),
        ("annealing", SimulatedAnnealing()),
    ]
    for name, sampler in samplers:
        xs = sampler.sample(problem, n_samples, np.random.default_rng(seed))
        results.append(
            RunResult.from_samples(
                name,
                problem,
                xs,
                evaluations=n_samples * sampler.evaluations_per_sample,
                seed=seed,
                config={"n_samples": n_samples},
            )
        )
    return results


def print_table(problem: HPProblem, results: list[RunResult]) -> None:
    print(
        f"\n{'algorithm':<16}{'shots':>6}{'evals':>7}{'P(ground)':>11}{'P(valid)':>10}"
        f"{'<E>':>8}{'<E>valid':>10}{'ratio':>7}"
    )
    for r in results:
        m = compute_metrics(problem, r)
        shots = "-" if r.shots is None else str(r.shots)
        print(
            f"{r.algorithm:<16}{shots:>6}{r.evaluations:>7}{m.p_ground:>11.3f}"
            f"{m.p_valid:>10.3f}{m.expected_energy:>8.2f}"
            f"{m.mean_valid_energy:>10.2f}{m.approx_ratio:>7.2f}"
        )


def main() -> None:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--hp", help="HP string, e.g. PHPPH")
    src.add_argument("--cif", help="mmCIF file")
    p.add_argument("--length", type=int, default=6)
    p.add_argument("--pick", type=int, default=0)
    p.add_argument("--penalty", type=float, default=None)
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    seq = pick_sequence(args)
    problem = HPProblem.from_sequence(seq, args.penalty)
    n_states = 2**problem.n_qubits

    print(
        f"\nsequence {seq}: {problem.n_residues} residues, {problem.n_qubits} qubits, "
        f"{n_states} folds, {int(problem.valid.sum())} valid, "
        f"penalty={problem.penalty:g}"
    )
    print(
        f"ground energy {problem.ground_energy:g}, "
        f"{len(problem.ground_states)} ground states"
    )
    for x in problem.ground_states:
        print(f"  x={int(x):>4}  {problem.fold(int(x))}")
    x0 = int(problem.ground_states[0])
    print("\nfold of first ground state ( : = H-H contact )")
    print(draw(seq, problem.decode(x0)))

    results = run_baselines(problem, args.samples, args.seed)
    print_table(problem, results)

    budget = SimulatedAnnealing().evaluations_per_sample
    p_hit = success_vs_shots(problem, results[0], [1, budget])
    print(
        f"\nrandom guessing finds a ground state with probability {p_hit[0]:.3f} in 1 "
        f"draw and {p_hit[1]:.3f} in {budget} draws (the cost of one annealing run)"
    )


if __name__ == "__main__":
    main()
