import numpy as np
import pytest

from hp_fold import HPProblem
from hp_fold.baselines.annealing import SimulatedAnnealing, metropolis_accept


def hamming_landscape(target: int, n_qubits: int = 8) -> HPProblem:
    """Toy problem: cost is the Hamming distance to `target`.

    Single-bit flips either fix a wrong bit or break a right one, so there is exactly
    one minimum and no local traps. 8 qubits correspond to 6 residues ("HPHPHP").
    """
    cost = np.array([(v ^ target).bit_count() for v in range(2**n_qubits)], dtype=float)
    return HPProblem(
        seq="HPHPHP",
        penalty=0.0,
        cost=cost,
        valid=np.ones(len(cost), dtype=bool),
        ground_energy=0.0,
        ground_states=np.array([target]),
    )


@pytest.mark.parametrize(
    ("delta", "temperature", "u", "expected"),
    [
        (-1.0, 0.1, 0.999, True),
        (0.0, 0.1, 0.999, True),
        (1.0, 1.0, 0.1, True),
        (1.0, 1.0, 0.5, False),
    ],
)
def test_metropolis_accept(delta, temperature, u, expected):
    assert metropolis_accept(delta, temperature, u) is expected


def test_finds_the_minimum_of_a_trap_free_landscape():
    problem = hamming_landscape(target=0b10110010)
    samples = SimulatedAnnealing(n_steps=200).sample(
        problem, 20, np.random.default_rng(0)
    )
    assert samples.tolist() == [0b10110010] * 20


def test_finds_ground_states_of_the_worked_example_most_of_the_time():
    problem = HPProblem.from_sequence("PHPPH")
    samples = SimulatedAnnealing().sample(problem, 30, np.random.default_rng(0))
    assert np.isin(samples, problem.ground_states).mean() >= 0.9


def test_samples_stay_inside_the_search_space():
    problem = HPProblem.from_sequence("PHPPH")
    samples = SimulatedAnnealing(n_steps=50).sample(
        problem, 30, np.random.default_rng(0)
    )
    assert samples.min() >= 0 and samples.max() < 64


def test_same_seed_gives_same_samples():
    problem = hamming_landscape(target=3)
    a = SimulatedAnnealing(n_steps=20).sample(problem, 10, np.random.default_rng(1))
    b = SimulatedAnnealing(n_steps=20).sample(problem, 10, np.random.default_rng(1))
    assert a.tolist() == b.tolist()


def test_evaluations_per_sample_counts_the_start_and_every_step():
    assert SimulatedAnnealing(n_steps=100).evaluations_per_sample == 101


@pytest.mark.parametrize(
    "kwargs", [{"n_steps": 0}, {"t_start": 0.1, "t_end": 1.0}, {"t_end": 0.0}]
)
def test_rejects_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        SimulatedAnnealing(**kwargs)
