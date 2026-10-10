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
from hp_fold.baselines.annealing import SimulatedAnnealing
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

    n_gs = len(problem.ground_states)
    rng = np.random.default_rng
    print(f"\nbaselines, {args.samples} samples each (seed={args.seed})")
    print(
        f"{'sampler':<20}{'evals/sample':>13}{'P(ground)':>11}{'P(valid)':>10}"
        f"{'mean E':>9}"
    )
    for name, sampler in [
        ("random", RandomSampler()),
        ("annealing", SimulatedAnnealing()),
    ]:
        xs = sampler.sample(problem, args.samples, rng(args.seed))
        print(
            f"{name:<20}{sampler.evaluations_per_sample:>13}"
            f"{np.isin(xs, problem.ground_states).mean():>11.3f}"
            f"{problem.valid[xs].mean():>10.3f}{problem.cost[xs].mean():>9.2f}"
        )
    k = SimulatedAnnealing().evaluations_per_sample
    p0 = n_gs / n_states
    print(
        f"\nrandom reference: P(ground)={p0:.3f} per draw; "
        f"{1 - (1 - p0) ** k:.3f} with the {k} draws one annealing run costs"
    )


if __name__ == "__main__":
    main()
