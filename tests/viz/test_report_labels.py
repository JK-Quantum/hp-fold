import numpy as np

from hp_fold.analysis.result import RunResult
from hp_fold.viz.report import label_of


def run(algorithm: str, **config) -> RunResult:
    return RunResult(algorithm, "PHPPH", np.full(4, 0.25), None, 1, config=config)


def test_label_names_the_depth_and_the_cvar_level():
    assert label_of(run("random")) == "random"
    assert label_of(run("qaoa", p=2)) == "qaoa, p = 2"
    assert label_of(run("qaoa_cvar", p=2, cvar_alpha=0.1)) == (
        "qaoa_cvar, p = 2, alpha = 0.1"
    )
