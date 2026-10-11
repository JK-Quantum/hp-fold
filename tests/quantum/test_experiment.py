import numpy as np
import pytest

from hp_fold.folding.problem import HPProblem
from hp_fold.quantum.experiment import QAOAExperiment, problem_metrics


@pytest.fixture(scope="module")
def problem() -> HPProblem:
    return HPProblem.from_sequence("PHPPH")


def test_problem_metrics_are_read_from_the_problem(problem):
    m = problem_metrics(problem)
    assert m["n_qubits"] == problem.n_qubits
    assert m["n_folds"] == problem.cost.size
    assert m["penalty"] == problem.penalty
    assert m["ground_energy"] == problem.ground_energy
    assert m["n_ground_states"] == problem.ground_states.size
    assert m["n_valid_folds"] == problem.valid.sum()
    assert m["ground_folds"] == [problem.fold(int(x)) for x in problem.ground_states]


def test_run_returns_one_result_per_depth(problem):
    result = QAOAExperiment(problem, restarts=2, maxiter=60).run(2)
    assert [r.p for r in result.depths] == [1, 2]
    assert set(result.probabilities) == {1, 2}
    for r in result.depths:
        probs = result.probabilities[r.p]
        assert probs.sum() == pytest.approx(1.0)
        assert r.p_valid == pytest.approx(probs[problem.valid].sum())
        assert r.p_ground == pytest.approx(probs[problem.ground_states].sum())
        assert r.expectation == pytest.approx(probs @ problem.cost)
        assert len(r.params) == 2 * r.p
        assert r.best_sampled_energy >= problem.ground_energy


def test_run_is_reproducible_for_a_seed(problem):
    a = QAOAExperiment(problem, restarts=2, maxiter=40, seed=7).run(2)
    b = QAOAExperiment(problem, restarts=2, maxiter=40, seed=7).run(2)
    assert [r.params for r in a.depths] == [r.params for r in b.depths]


def test_numpy_and_qiskit_backends_agree(problem):
    kwargs = dict(restarts=1, maxiter=25, seed=1)
    a = QAOAExperiment(problem, backend="numpy", **kwargs).run(1)
    b = QAOAExperiment(problem, backend="qiskit", **kwargs).run(1)
    # Exact agreement at fixed parameters is tested in test_numpy_qaoa; here COBYLA
    # amplifies rounding differences between the simulators, so the tolerance is loose.
    assert a.depths[0].expectation == pytest.approx(b.depths[0].expectation, abs=1e-3)


def test_cvar_objective_differs_from_expectation(problem):
    result = QAOAExperiment(problem, cvar_alpha=0.1, restarts=2, maxiter=80).run(1)
    r = result.depths[0]
    assert r.objective_value < r.expectation  # CVaR looks at the best tail only
    assert r.objective_value >= problem.ground_energy - 1e-9


def test_to_dict_is_json_ready(problem):
    import json

    result = QAOAExperiment(problem, restarts=1, maxiter=20).run(1)
    data = result.to_dict()
    assert set(data) == {"problem", "qaoa"}
    json.dumps(data)


def test_landscape_shape(problem):
    gammas, betas, z = QAOAExperiment(problem).landscape(grid=5)
    assert gammas.shape == betas.shape == (5,)
    assert z.shape == (5, 5)


@pytest.mark.parametrize(
    "kwargs, match",
    [({"backend": "nope"}, "backend"), ({"shots": 0}, "shots")],
)
def test_invalid_settings(problem, kwargs, match):
    with pytest.raises(ValueError, match=match):
        QAOAExperiment(problem, **kwargs)


def test_depth_must_be_positive(problem):
    with pytest.raises(ValueError, match="depth"):
        QAOAExperiment(problem).run(0)


def test_circuit_depth_and_backend(problem):
    circuit = QAOAExperiment(problem, backend="numpy").circuit(3)
    assert circuit.n_params == 6
    assert circuit.problem is problem
    assert np.isclose(circuit.probabilities(np.zeros(6)).sum(), 1.0)
