import json

import numpy as np
import pytest
from qiskit import transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library import (
    CXGate,
    HGate,
    Measure,
    PauliEvolutionGate,
    RXGate,
    RZGate,
)
from qiskit.providers import BackendV2, Options
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.quantum_info import Statevector
from qiskit.transpiler import Target

from hp_fold.analysis.backend_report import BackendReport
from hp_fold.folding.problem import HPProblem
from hp_fold.quantum.backend_runner import BackendRunner, expand_opaque_blocks
from hp_fold.quantum.numpy_qaoa import NumpyQAOA
from hp_fold.quantum.qaoa import QAOACircuit

PARAMS = np.array([0.45, 0.35])  # p = 1


@pytest.fixture(scope="module")
def problem() -> HPProblem:
    return HPProblem.from_sequence("PHPPH")


@pytest.fixture(scope="module")
def runner(problem) -> BackendRunner:
    return BackendRunner(
        QAOACircuit(problem, 1),
        BasicSimulator(),
        run_options={"seed_simulator": 7},
    )


@pytest.fixture(scope="module")
def result(runner):
    return runner.run(PARAMS, shots=4000, seed=1)


def test_counts_to_folds_is_little_endian_int():
    # "000011" = qubits 0 and 1 set = fold index 3 (rightmost char is qubit 0)
    assert BackendRunner.counts_to_folds({"000011": 5, "100000": 2}) == {3: 5, 32: 2}
    assert BackendRunner.counts_to_folds({"0001 0000": 4}) == {16: 4}


def test_counts_to_folds_merges_duplicate_keys():
    assert BackendRunner.counts_to_folds({"01": 1, "0 1": 2}) == {1: 3}


def test_measured_distribution_matches_exact_within_shot_noise(result):
    assert result.shots == 4000
    assert sum(result.counts.values()) == 4000
    assert result.probabilities.sum() == pytest.approx(1.0)
    # An ideal backend should land at the sampling noise floor, not far above it.
    assert result.tv_distance < 2.5 * result.tv_distance_shot_noise + 0.02


def test_bit_order_bug_would_be_caught(problem, result):
    # Reverse the bit order of every fold index: the distance must become large.
    n = problem.n_qubits
    reverse = [int(format(x, f"0{n}b")[::-1], 2) for x in range(2**n)]
    flipped = result.probabilities[reverse]
    tv = 0.5 * np.abs(flipped - result.exact_probabilities).sum()
    assert tv > 4 * result.tv_distance


def test_metrics_are_consistent_with_the_problem(problem, result):
    p = result.probabilities
    assert result.p_valid == pytest.approx(p[problem.valid].sum())
    assert result.p_ground == pytest.approx(p[problem.ground_states].sum())
    assert result.p_ground_given_valid == pytest.approx(
        result.p_ground / result.p_valid
    )
    assert result.mean_energy == pytest.approx(p @ problem.cost)
    assert result.best_sampled_energy >= problem.ground_energy
    assert result.best_sampled_energy == problem.cost[np.flatnonzero(p)].min()


def test_exact_distribution_is_the_numpy_reference(problem, result):
    assert np.allclose(
        result.exact_probabilities, NumpyQAOA(problem, 1).probabilities(PARAMS)
    )


def test_circuit_stats_are_reported_after_transpiling(result):
    stats = result.stats
    assert stats.depth > 0
    assert stats.n_two_qubit_gates > 0
    assert stats.n_gates >= stats.n_two_qubit_gates
    assert stats.ops["measure"] == 6


def test_run_is_reproducible_for_seeded_backend(runner):
    a = runner.run(PARAMS, shots=500, seed=3)
    b = runner.run(PARAMS, shots=500, seed=3)
    assert a.counts == b.counts


def test_invalid_shots(runner):
    with pytest.raises(ValueError, match="shots"):
        runner.run(PARAMS, shots=0)


def test_to_dict_is_json_ready(result):
    data = result.to_dict()
    assert "probabilities" not in data and "exact_probabilities" not in data
    assert all(isinstance(k, str) for k in data["counts"])
    json.dumps(data)


def test_ionq_factory_fails_clearly_without_a_key(problem, monkeypatch):
    monkeypatch.delenv("IONQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="IONQ_API_KEY"):
        BackendRunner.from_ionq(QAOACircuit(problem, 1))


def test_report_writes_json_and_figure(tmp_path, problem, result):
    paths = BackendReport(problem, result).save(tmp_path, "local_p1")
    assert {p.name for p in paths} == {"local_p1.json", "local_p1.png"}
    assert all(p.stat().st_size > 0 for p in paths)
    assert json.loads(paths[0].read_text())["shots"] == 4000


class OpaquePauliBackend(BackendV2):
    """Like IonQ's QIS target: PauliEvolution is a supported (opaque) instruction."""

    def __init__(self, n_qubits: int) -> None:
        super().__init__(name="opaque_pauli")
        theta = Parameter("theta")
        target = Target(num_qubits=n_qubits)
        target.add_instruction(HGate(), name="h")
        target.add_instruction(RXGate(theta), name="rx")
        target.add_instruction(RZGate(theta), name="rz")
        target.add_instruction(CXGate(), name="cx")
        target.add_instruction(Measure(), name="measure")
        target.add_instruction(PauliEvolutionGate, name="PauliEvolution")
        self._target = target

    @property
    def target(self) -> Target:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._target

    @property
    def max_circuits(self) -> int | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        return None

    @classmethod
    def _default_options(cls) -> Options:  # pyright: ignore[reportIncompatibleMethodOverride]
        return Options()

    def run(self, run_input, **options):
        raise NotImplementedError


def test_plain_transpile_keeps_pauli_evolution_opaque_on_such_targets(problem):
    # The trap: stats from this circuit would claim a handful of two-qubit gates.
    circuit = QAOACircuit(problem, 1)
    naive = transpile(
        circuit.measured_circuit(PARAMS),
        OpaquePauliBackend(problem.n_qubits),
        optimization_level=1,
    )
    assert naive.num_nonlocal_gates() <= 3


def test_runner_expands_opaque_blocks_so_stats_are_real(problem):
    runner = BackendRunner(
        QAOACircuit(problem, 1), OpaquePauliBackend(problem.n_qubits)
    )
    stats = runner.stats_of(runner.transpile(PARAMS))
    assert "PauliEvolution" not in stats.ops and "QAOA" not in stats.ops
    assert stats.n_two_qubit_gates > 50  # N=5, p=1: ~134 CX after optimization
    assert stats.depth > 50


def test_expansion_does_not_change_the_state(problem):
    circuit = QAOACircuit(problem, 2)
    params = np.array([0.37, -0.81, 0.22, 0.54])
    expanded = expand_opaque_blocks(circuit.build_qaoa_circuit(params))
    assert not {"QAOA", "PauliEvolution"} & set(expanded.count_ops())
    assert np.allclose(
        np.abs(Statevector(expanded).data) ** 2, circuit.probabilities(params)
    )


def test_expansion_leaves_depth_zero_circuit_alone(problem):
    circuit = QAOACircuit(problem, 0).build_qaoa_circuit(np.array([]))
    assert expand_opaque_blocks(circuit).count_ops() == circuit.count_ops()


@pytest.mark.parametrize("seed", [0, -3])
def test_ionq_sampler_seed_must_be_positive(problem, seed):
    # The IonQ API answers HTTP 400 '"noise.seed" must be a positive number'.
    # Validated before any key lookup or network access.
    with pytest.raises(ValueError, match="sampler_seed"):
        BackendRunner.from_ionq(QAOACircuit(problem, 1), sampler_seed=seed)
