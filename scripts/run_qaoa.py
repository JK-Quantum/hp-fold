import numpy as np
import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from hp_fold.folding.problem import HPProblem
from hp_fold.quantum.qaoa import QAOACircuit

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "results" / "qaoa"

@dataclass(frozen=True)
class DepthResult:
    p: int
    params: list[float]
    expectation: float
    improvement_over_random: float
    p_valid: float
    p_ground: float
    best_sampled_energy: float
    n_evaluations: int
    history: list[float]


def problem_metrics(problem: HPProblem) -> dict:
    n_valid = int(problem.valid.sum())
    return {
        "sequence": problem.seq,
        "n_residues": problem.n_residues,
        "n_qubits": problem.n_qubits,
        "n_folds": int(problem.cost.size),
        "penalty": problem.penalty,
        "ground_energy": problem.ground_energy,
        "n_ground_states": int(problem.ground_states.size),
        "ground_folds": [problem.fold(int(x)) for x in problem.ground_states],
        "n_valid_folds": n_valid,
        "valid_fraction": n_valid / problem.cost.size,
        "mean_cost": float(problem.cost.mean()),
        "random_p_ground": problem.ground_states.size / problem.cost.size,
    }


def optimize_depth(
    problem: HPProblem,
    p: int,
    restarts: int,
    maxiter: int,
    rng: np.random.Generator,
    warm_start: np.ndarray | None,
    shots: int,
) -> tuple[DepthResult, QAOACircuit, np.ndarray]:
    circuit = QAOACircuit(problem, p)
    best_x, best_f, best_history, n_evals = None, np.inf, [], 0

    starts = [rng.uniform(0, np.pi, circuit.n_params) for _ in range(restarts)]
    if warm_start is not None:
        g, b = np.split(warm_start, 2)
        starts.append(np.concatenate([g, [g[-1]], b, [b[-1]]]))

    for x0 in starts:
        history: list[float] = []

        def objective(x: np.ndarray, history: list[float] = history) -> float:
            value = circuit.expectation(x)
            history.append(min(value, history[-1]) if history else value)
            return value

        result = minimize(objective, x0, method="COBYLA", options={"maxiter": maxiter})
        n_evals += len(history)
        if result.fun < best_f:
            best_x, best_f, best_history = result.x, float(result.fun), history

    assert best_x is not None
    probs = circuit.probabilities(best_x)
    mean, ground = float(problem.cost.mean()), problem.ground_energy
    sampled = circuit.sample(best_x, shots, rng)
    result = DepthResult(
        p=p,
        params=best_x.tolist(),
        expectation=best_f,
        improvement_over_random=(mean - best_f) / (mean - ground),
        p_valid=float(probs[problem.valid].sum()),
        p_ground=float(probs[problem.ground_states].sum()),
        best_sampled_energy=float(problem.cost[sampled].min()),
        n_evaluations=n_evals,
        history=best_history,
    )
    return result, circuit, probs


def plot_distribution(problem: HPProblem, probs: np.ndarray, p: int, path: Path):
    order = np.argsort(problem.cost, kind="stable")
    ground = set(problem.ground_states.tolist())
    colors = [
        "tab:green" if x in ground else "tab:blue" if problem.valid[x] else "tab:red"
        for x in order.tolist()
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(np.arange(order.size), probs[order], color=colors, width=1.0)
    ax.axhline(1 / problem.cost.size, color="k", ls="--", lw=0.8, label="uniform")
    ax.set(
        xlabel="folds, sorted by cost (low to high)",
        ylabel="probability",
        title=f"{problem.seq}: output distribution, p={p} "
        "(green = ground, blue = valid, red = overlap)",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_convergence(results: list[DepthResult], ground: float, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for r in results:
        ax.plot(r.history, label=f"p={r.p}")
    ax.axhline(ground, color="k", ls="--", lw=0.8, label="ground energy")
    ax.set(xlabel="evaluation", ylabel="best <C> so far", title="Optimizer convergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_metrics_vs_depth(results: list[DepthResult], path: Path):
    ps = [r.p for r in results]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].plot(ps, [r.expectation for r in results], "o-")
    axes[0].set(xlabel="p", ylabel="<C>", title="Expected cost")
    axes[1].plot(ps, [r.p_ground for r in results], "o-", label="P(ground)")
    axes[1].plot(ps, [r.p_valid for r in results], "s-", label="P(valid)")
    axes[1].set(xlabel="p", ylabel="probability", title="Success probabilities")
    axes[1].legend()
    axes[2].plot(ps, [r.improvement_over_random for r in results], "o-")
    axes[2].set(xlabel="p", ylabel="(mean - <C>)/(mean - ground)", title="Improvement")
    for ax in axes:
        ax.set_xticks(ps)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_landscape(problem: HPProblem, path: Path, grid: int = 60):
    circuit = QAOACircuit(problem, 1)
    gammas = np.linspace(0, np.pi, grid)
    betas = np.linspace(0, np.pi / 2, grid)
    z = np.array(
        [[circuit.expectation(np.array([g, b])) for g in gammas] for b in betas]
    )
    fig, ax = plt.subplots(figsize=(6, 4.5))
    mesh = ax.pcolormesh(gammas, betas, z, shading="auto", cmap="viridis")
    fig.colorbar(mesh, label="<C>")
    ax.set(xlabel="gamma", ylabel="beta", title=f"{problem.seq}: p=1 landscape")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QAOA on an HP sequence.")
    parser.add_argument("sequence", nargs="?", default="PHPPH", help="HP string")
    parser.add_argument("--depth", type=int, default=3, help="max QAOA depth p")
    parser.add_argument("--restarts", type=int, default=6, help="random starts per p")
    parser.add_argument("--maxiter", type=int, default=300, help="COBYLA iterations")
    parser.add_argument("--shots", type=int, default=1024, help="shots for sampling")
    parser.add_argument("--penalty", type=float, default=None, help="overlap penalty")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.depth < 1:
        parser.error("--depth must be >= 1")

    seq = args.sequence.upper()
    problem = HPProblem.from_sequence(seq, args.penalty)
    out = args.out / seq
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    results: list[DepthResult] = []
    warm = None
    last_probs = None
    for p in range(1, args.depth + 1):
        result, _, probs = optimize_depth(
            problem, p, args.restarts, args.maxiter, rng, warm, args.shots
        )
        results.append(result)
        warm, last_probs = np.array(result.params), probs
        plot_distribution(problem, probs, p, out / f"distribution_p{p}.png")
        print(
            f"p={p}: <C>={result.expectation:.4f}  "
            f"P(ground)={result.p_ground:.4f}  P(valid)={result.p_valid:.4f}  "
            f"improvement={result.improvement_over_random:.3f}"
        )
    assert last_probs is not None

    plot_convergence(results, problem.ground_energy, out / "convergence.png")
    plot_metrics_vs_depth(results, out / "metrics_vs_depth.png")
    plot_landscape(problem, out / "landscape_p1.png")

    metrics = {
        "problem": problem_metrics(problem),
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "qaoa": [asdict(r) for r in results],
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nProblem: {json.dumps(metrics['problem'], indent=2)}")
    print(f"\nSaved metrics and figures to {out}")


if __name__ == "__main__":
    main()
