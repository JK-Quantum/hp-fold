from dataclasses import asdict, dataclass, field

import numpy as np

from hp_fold.folding.problem import HPProblem
from hp_fold.quantum.numpy_qaoa import NumpyQAOA
from hp_fold.quantum.objectives import CVaRObjective, ExpectationObjective
from hp_fold.quantum.optimize import minimize_with_restarts
from hp_fold.quantum.qaoa import QAOACircuit

BACKENDS: dict[str, type[NumpyQAOA] | type[QAOACircuit]] = {
    "numpy": NumpyQAOA,
    "qiskit": QAOACircuit,
}


@dataclass(frozen=True)
class DepthResult:
    p: int
    params: list[float]
    objective_value: float
    expectation: float
    improvement_over_random: float
    p_valid: float
    p_ground: float
    best_sampled_energy: float
    n_evaluations: int
    history: list[float]


def problem_metrics(problem: HPProblem) -> dict:
    """metrics about the problem itself."""
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


@dataclass(frozen=True)
class ExperimentResult:
    problem: HPProblem
    depths: list[DepthResult]
    probabilities: dict[int, np.ndarray] = field(repr=False)

    def to_dict(self) -> dict:
        return {
            "problem": problem_metrics(self.problem),
            "qaoa": [asdict(r) for r in self.depths],
        }


class QAOAExperiment:
    """QAOA (or CVaR-QAOA) over increasing depth for one problem."""

    def __init__(
        self,
        problem: HPProblem,
        *,
        backend: str = "numpy",
        cvar_alpha: float | None = None,
        restarts: int = 6,
        maxiter: int = 300,
        shots: int = 1024,
        seed: int = 0,
    ) -> None:
        if backend not in BACKENDS:
            raise ValueError(
                f"backend must be one of {sorted(BACKENDS)}, got {backend!r}"
            )
        if shots < 1:
            raise ValueError(f"shots must be >= 1, got {shots}")
        self._problem = problem
        self._backend = backend
        self._cvar_alpha = cvar_alpha
        self._restarts = restarts
        self._maxiter = maxiter
        self._shots = shots
        self._seed = seed

    @property
    def problem(self) -> HPProblem:
        return self._problem

    def circuit(self, p: int) -> NumpyQAOA | QAOACircuit:
        return BACKENDS[self._backend](self._problem, p)

    def objective(self, circuit: NumpyQAOA | QAOACircuit):
        if self._cvar_alpha is None:
            return ExpectationObjective(circuit)
        return CVaRObjective(circuit, self._cvar_alpha)

    def run(self, depth: int) -> ExperimentResult:
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        rng = np.random.default_rng(self._seed)
        results: list[DepthResult] = []
        probabilities: dict[int, np.ndarray] = {}
        warm_start = None
        for p in range(1, depth + 1):
            result, probs = self._optimize_depth(self.circuit(p), warm_start, rng)
            results.append(result)
            probabilities[p] = probs
            warm_start = np.array(result.params)
        return ExperimentResult(self._problem, results, probabilities)

    def landscape(self, grid: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        circuit = NumpyQAOA(self._problem, 1)
        gammas = np.linspace(0, np.pi, grid)
        betas = np.linspace(0, np.pi / 2, grid)
        z = np.array(
            [[circuit.expectation(np.array([g, b])) for g in gammas] for b in betas]
        )
        return gammas, betas, z

    def _optimize_depth(
        self,
        circuit: NumpyQAOA | QAOACircuit,
        warm_start: np.ndarray | None,
        rng: np.random.Generator,
    ) -> tuple[DepthResult, np.ndarray]:
        problem = self._problem
        extra_starts = []
        if warm_start is not None:
            g, b = np.split(warm_start, 2)
            extra_starts.append(np.concatenate([g, [g[-1]], b, [b[-1]]]))

        best = minimize_with_restarts(
            self.objective(circuit),
            restarts=self._restarts,
            rng=rng,
            maxiter=self._maxiter,
            extra_starts=extra_starts,
        )
        probs = circuit.probabilities(best.params)
        expectation = float(probs @ problem.cost)
        mean, ground = float(problem.cost.mean()), problem.ground_energy
        sampled = circuit.sample(best.params, self._shots, rng)
        result = DepthResult(
            p=circuit.n_params // 2,
            params=best.params.tolist(),
            objective_value=best.value,
            expectation=expectation,
            improvement_over_random=(mean - expectation) / (mean - ground),
            p_valid=float(probs[problem.valid].sum()),
            p_ground=float(probs[problem.ground_states].sum()),
            best_sampled_energy=float(problem.cost[sampled].min()),
            n_evaluations=best.n_evaluations,
            history=best.history,
        )
        return result, probs
