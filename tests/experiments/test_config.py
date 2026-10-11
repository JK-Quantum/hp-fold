from pathlib import Path

import pytest

from hp_fold.experiments.config import load_config, parse_config

BASE = Path("/project")
CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def minimal(**overrides):
    data = {"problem": {"sequence": "PHPPH"}, "runs": [{"algorithm": "random"}]}
    data.update(overrides)
    return data


def test_minimal_config_gets_defaults():
    cfg = parse_config(minimal(), BASE)
    assert cfg.sequence == "PHPPH"
    assert cfg.penalty is None
    assert cfg.seed == 0
    assert cfg.output == BASE / "data" / "results" / "runs"
    (run,) = cfg.runs
    assert (run.algorithm, run.repeats, dict(run.params)) == ("random", 1, {})


def test_full_config_is_read():
    data = {
        "seed": 7,
        "output": "out/dir",
        "problem": {"sequence": "HPHPPH", "penalty": 3},
        "runs": [
            {"algorithm": "annealing", "repeats": 5, "params": {"n_samples": 200}},
            {
                "algorithm": "hp_fold.algos.qaoa:run",
                "params": {"p": 2, "backend": "aer"},
            },
        ],
    }
    cfg = parse_config(data, BASE)
    assert (cfg.seed, cfg.penalty, cfg.sequence) == (7, 3.0, "HPHPPH")
    assert [r.algorithm for r in cfg.runs] == ["annealing", "hp_fold.algos.qaoa:run"]
    assert cfg.runs[0].repeats == 5
    assert dict(cfg.runs[1].params) == {"p": 2, "backend": "aer"}


def test_relative_output_resolves_against_base_but_absolute_does_not():
    assert parse_config(minimal(output="a/b"), BASE).output == BASE / "a" / "b"
    assert parse_config(minimal(output="/elsewhere"), BASE).output == Path("/elsewhere")


@pytest.mark.parametrize(
    "data, key",
    [
        (minimal(sead=3), "sead"),
        (minimal(problem={"sequence": "PHPPH", "lenght": 5}), "lenght"),
        (minimal(runs=[{"algorithm": "random", "repeat": 2}]), "repeat"),
    ],
)
def test_unknown_keys_are_rejected(data, key):
    with pytest.raises(ValueError, match=f"unknown.*{key}"):
        parse_config(data, BASE)


@pytest.mark.parametrize(
    "data, message",
    [
        ({"runs": [{"algorithm": "random"}]}, "problem"),
        (minimal(runs=[]), "runs"),
        ({"problem": {"sequence": "PHPPH"}}, "runs"),
        (minimal(runs=[{"repeats": 2}]), "algorithm"),
        (minimal(runs=[{"algorithm": ""}]), "algorithm"),
        (minimal(runs=[{"algorithm": "random", "params": [1, 2]}]), "params"),
    ],
)
def test_missing_or_malformed_sections_are_rejected(data, message):
    with pytest.raises(ValueError, match=message):
        parse_config(data, BASE)


@pytest.mark.parametrize("sequence", ["", "PHX", "phpph", 123, None])
def test_bad_sequence_is_rejected(sequence):
    with pytest.raises(ValueError, match="sequence"):
        parse_config(minimal(problem={"sequence": sequence}), BASE)


@pytest.mark.parametrize("repeats", [0, -1, 1.5, True])
def test_bad_repeats_are_rejected(repeats):
    with pytest.raises(ValueError, match="repeats"):
        parse_config(minimal(runs=[{"algorithm": "random", "repeats": repeats}]), BASE)


@pytest.mark.parametrize("seed", [1.5, "3", True])
def test_bad_seed_is_rejected(seed):
    with pytest.raises(ValueError, match="seed"):
        parse_config(minimal(seed=seed), BASE)


@pytest.mark.parametrize("penalty", [0, -2.0, "big", True])
def test_bad_penalty_is_rejected(penalty):
    with pytest.raises(ValueError, match="penalty"):
        parse_config(minimal(problem={"sequence": "PHPPH", "penalty": penalty}), BASE)


def test_load_config_reads_toml_and_defaults_base_to_file_directory(tmp_path):
    path = tmp_path / "exp.toml"
    path.write_text(
        'seed = 4\n[problem]\nsequence = "PHPPH"\n'
        '[[runs]]\nalgorithm = "random"\n[runs.params]\nn_samples = 10\n'
    )
    cfg = load_config(path)
    assert cfg.seed == 4
    assert cfg.output == tmp_path / "data" / "results" / "runs"
    assert dict(cfg.runs[0].params) == {"n_samples": 10}


def test_every_committed_config_parses():
    paths = sorted(CONFIGS.glob("*.toml"))
    assert paths, f"no configs found in {CONFIGS}"
    for path in paths:
        load_config(path, base_dir=CONFIGS.parent)


def cif_problem(**extra):
    return minimal(problem={"cif": "data/raw/1UBQ.cif", **extra})


def test_cif_problem_is_read_with_defaults():
    cfg = parse_config(cif_problem(), BASE)
    assert cfg.sequence is None
    assert cfg.cif == BASE / "data" / "raw" / "1UBQ.cif"
    assert (cfg.fragment_length, cfg.fragment_pick) == (6, 0)


def test_cif_problem_reads_length_and_pick_and_keeps_absolute_paths():
    cfg = parse_config(
        minimal(problem={"cif": "/data/x.cif", "length": 8, "pick": 3}), BASE
    )
    assert cfg.cif == Path("/data/x.cif")
    assert (cfg.fragment_length, cfg.fragment_pick) == (8, 3)


def test_sequence_problem_has_no_cif():
    cfg = parse_config(minimal(), BASE)
    assert cfg.cif is None


@pytest.mark.parametrize(
    "problem",
    [
        {"sequence": "PHPPH", "cif": "x.cif"},
        {"penalty": 3},
    ],
)
def test_exactly_one_of_sequence_and_cif_is_required(problem):
    with pytest.raises(ValueError, match="either 'sequence' or 'cif'"):
        parse_config(minimal(problem=problem), BASE)


@pytest.mark.parametrize("key", ["length", "pick"])
def test_fragment_options_need_a_cif(key):
    with pytest.raises(ValueError, match="only valid together with 'cif'"):
        parse_config(minimal(problem={"sequence": "PHPPH", key: 4}), BASE)


@pytest.mark.parametrize("length", [3, 0, 1.5, True, "6"])
def test_bad_fragment_length_is_rejected(length):
    with pytest.raises(ValueError, match="length"):
        parse_config(cif_problem(length=length), BASE)


@pytest.mark.parametrize("pick", [-1, 1.5, True, "0"])
def test_bad_fragment_pick_is_rejected(pick):
    with pytest.raises(ValueError, match="pick"):
        parse_config(cif_problem(pick=pick), BASE)


@pytest.mark.parametrize("cif", ["", 5, None])
def test_bad_cif_path_is_rejected(cif):
    with pytest.raises(ValueError, match="cif"):
        parse_config(minimal(problem={"cif": cif}), BASE)
