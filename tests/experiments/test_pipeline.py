from pathlib import Path

from hp_fold.experiments.pipeline import format_table, run_pipeline

QAOA_RUN = """
[[runs]]
algorithm = "qaoa"
repeats = 2

[runs.params]
p = 1
restarts = 1
maxiter = 10
shots = 32
landscape_grid = 3
"""

HEADER = """
output = "out"

[problem]
sequence = "PHPPH"

[[runs]]
algorithm = "random"
repeats = 2

[runs.params]
n_samples = 20
"""


def write_config(tmp_path: Path, extra: str = "") -> Path:
    path = tmp_path / "tiny.toml"
    path.write_text(HEADER + extra)
    return path


def test_pipeline_runs_summarizes_and_draws_every_available_figure(tmp_path):
    report = run_pipeline(write_config(tmp_path, QAOA_RUN), tmp_path)
    assert report.ok
    assert report.saved == 4
    assert [row.algorithm for row in report.rows] == ["random", "qaoa"]
    assert "qaoa p=1" in format_table(report.rows)
    assert report.summary_path == tmp_path / "out" / "summary.csv"
    assert (tmp_path / "out" / "summary.csv").exists()
    assert not report.figures.skipped
    names = {path.name for path in report.figures.written if path.suffix == ".png"}
    for figure in ("energy", "budget", "convergence", "sweep", "landscape"):
        assert f"{figure}_PHPPH_light.png" in names
        assert f"{figure}_PHPPH_dark.png" in names


def test_figures_the_runs_cannot_support_are_skipped_not_failed(tmp_path):
    report = run_pipeline(write_config(tmp_path), tmp_path)
    assert report.ok
    assert set(report.figures.skipped) == {"convergence", "sweep", "landscape"}
    assert {p.name for p in report.figures.written if p.suffix == ".png"} == {
        f"{figure}_PHPPH_{theme}.png"
        for figure in ("energy", "budget")
        for theme in ("light", "dark")
    }
