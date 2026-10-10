import numpy as np

from hp_fold import HPProblem
from hp_fold.baselines.random_sampling import RandomSampler


def test_samples_are_valid_integers_for_the_problem():
    problem = HPProblem.from_sequence("PHPPH")
    samples = RandomSampler().sample(problem, 1000, np.random.default_rng(0))
    assert samples.shape == (1000,)
    assert samples.min() >= 0 and samples.max() < 2**problem.n_qubits


def test_covers_the_whole_search_space():
    problem = HPProblem.from_sequence("PHPPH")
    samples = RandomSampler().sample(problem, 6400, np.random.default_rng(0))
    assert len(np.unique(samples)) == 64


def test_same_seed_gives_same_samples():
    problem = HPProblem.from_sequence("PHPPH")
    a = RandomSampler().sample(problem, 50, np.random.default_rng(7))
    b = RandomSampler().sample(problem, 50, np.random.default_rng(7))
    assert a.tolist() == b.tolist()


def test_costs_one_evaluation_per_sample():
    assert RandomSampler().evaluations_per_sample == 1
