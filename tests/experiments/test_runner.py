from pathlib import Path

import pytest

from hp_fold import HPProblem
from hp_fold.analysis.result import RunResult
from hp_fold.analysis.store import load_result
from hp_fold.chem.records import ChainRecord, ResidueRecord
from hp_fold.experiments.algorithms import run_random
from hp_fold.experiments.config import ExperimentConfig, RunSpec
from hp_fold.experiments.runner import run_experiment


def make_config(tmp_path, runs, seed=0, penalty=None, sequence="PHPPH"):
    return ExperimentConfig(
        sequence=sequence,
        penalty=penalty,
        seed=seed,
        output=tmp_path / "out",
        runs=tuple(runs),
    )


def spec(algorithm="random", repeats=1, **params):
    return RunSpec(algorithm, {"n_samples": 20, **params}, repeats)


def fake_resolver(**runners):
    def resolve(name):
        if name not in runners:
            raise ValueError(f"unknown algorithm {name!r}")
        return runners[name]

    return resolve


def test_every_repeat_is_run_saved_and_loadable(tmp_path):
    config = make_config(
        tmp_path, [spec("random", repeats=3), spec("annealing", repeats=2)]
    )
    report = run_experiment(config)
    assert report.ok and len(report.saved) == 5
    assert all(path.exists() for path in report.saved)
    assert [load_result(p).algorithm for p in report.saved] == ["random"] * 3 + [
        "annealing"
    ] * 2


def test_repeats_use_consecutive_seeds_starting_at_the_config_seed(tmp_path):
    config = make_config(tmp_path, [spec(repeats=3), spec(repeats=2)], seed=10)
    report = run_experiment(config)
    assert [load_result(p).seed for p in report.saved] == [10, 11, 12, 10, 11]


def test_runner_receives_the_configured_problem_params_and_seed(tmp_path):
    seen = []

    def spy(problem, params, seed):
        seen.append((problem.seq, problem.penalty, dict(params), seed))
        return run_random(problem, {"n_samples": params["n_samples"]}, seed)

    config = make_config(tmp_path, [spec("spy", extra=1)], seed=4, penalty=7)
    run_experiment(config, fake_resolver(spy=spy))
    assert seen == [("PHPPH", 7.0, {"n_samples": 20, "extra": 1}, 4)]


def test_default_penalty_is_used_when_the_config_has_none(tmp_path):
    seen = []

    def spy(problem, params, seed):
        seen.append(problem.penalty)
        return run_random(problem, params, seed)

    run_experiment(make_config(tmp_path, [spec("spy")]), fake_resolver(spy=spy))
    assert seen == [5.0]  # number of residues


def test_a_failing_run_is_reported_and_the_others_still_run(tmp_path):
    def boom(problem, params, seed):
        raise RuntimeError("backend unavailable")

    config = make_config(tmp_path, [spec("ok"), spec("boom", repeats=2), spec("ok")])
    report = run_experiment(config, fake_resolver(ok=run_random, boom=boom))
    assert len(report.saved) == 2 and not report.ok
    assert [(f.algorithm, f.repeat, f.seed) for f in report.failures] == [
        ("boom", 0, 0),
        ("boom", 1, 1),
    ]
    assert "RuntimeError: backend unavailable" in report.failures[0].error


def test_a_runner_returning_the_wrong_type_is_reported(tmp_path):
    config = make_config(tmp_path, [spec("bad")])
    report = run_experiment(
        config, fake_resolver(bad=lambda problem, params, seed: None)
    )
    assert not report.saved
    assert (
        "NoneType" in report.failures[0].error
        and "RunResult" in report.failures[0].error
    )


def test_a_result_for_the_wrong_sequence_is_reported_not_saved(tmp_path):
    def wrong(problem, params, seed):
        other = HPProblem.from_sequence("HHPHH")
        return RunResult.from_samples("wrong", other, [0], evaluations=1)

    report = run_experiment(
        make_config(tmp_path, [spec("wrong")]),
        fake_resolver(wrong=wrong),
    )
    assert not report.saved
    assert report.failures[0].error.startswith("ValueError")


def test_an_unknown_algorithm_is_one_failure_and_does_not_stop_the_rest(tmp_path):
    config = make_config(tmp_path, [spec("quantum-magic", repeats=3), spec("random")])
    report = run_experiment(config)
    assert len(report.saved) == 1
    (failure,) = report.failures
    assert (failure.algorithm, failure.repeat, failure.seed) == (
        "quantum-magic",
        None,
        None,
    )
    assert "unknown algorithm" in failure.error


def test_bad_params_in_a_builtin_are_reported_per_repeat(tmp_path):
    config = make_config(tmp_path, [RunSpec("random", {"n_sample": 5}, repeats=2)])
    report = run_experiment(config)
    assert [f.repeat for f in report.failures] == [0, 1]
    assert "n_sample" in report.failures[0].error


def test_an_unusable_problem_stops_the_experiment_immediately(tmp_path):
    config = make_config(tmp_path, [spec()], penalty=0.5, sequence="HHHHHH")
    with pytest.raises(ValueError, match="penalty"):
        run_experiment(config)


def test_report_carries_the_sequence_and_where_it_came_from(tmp_path):
    report = run_experiment(make_config(tmp_path, [spec()]))
    assert report.sequence == "PHPPH"
    assert report.origin


def test_a_cif_config_runs_on_the_selected_fragment(tmp_path):
    names = ["MET", "GLN", "ILE", "PHE", "VAL", "LYS"]
    chain = ChainRecord(
        "FAKE",
        "A",
        tuple(ResidueRecord(n, i, (3.8 * i, 0.0, 0.0)) for i, n in enumerate(names, 1)),
    )
    config = ExperimentConfig(
        sequence=None,
        penalty=None,
        seed=0,
        output=tmp_path / "out",
        runs=(spec(),),
        cif=Path("x.cif"),
    )
    report = run_experiment(config, loader=lambda path: [chain])
    assert report.sequence == "HPHHHP"
    assert [load_result(p).seq for p in report.saved] == ["HPHHHP"]


def test_a_pick_beyond_the_available_fragments_stops_the_experiment(tmp_path):
    config = ExperimentConfig(
        sequence=None,
        penalty=None,
        seed=0,
        output=tmp_path / "out",
        runs=(spec(),),
        cif=Path("x.cif"),
        fragment_pick=5,
    )
    with pytest.raises(ValueError, match="out of range"):
        run_experiment(config, loader=lambda path: [])
