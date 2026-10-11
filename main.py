"""Qollab entry point: QAOA and CVaR-QAOA, ideal and on the platform backend.

`backend` is a global provided by the Qollab runtime; it is only declared here. Running
`python main.py` locally sets it to a Qiskit simulator.

The angles are trained locally on the exact statevector, then every trained circuit is
evaluated by the backend in one job. A metrics table is printed and the figures are
displayed.

DATA lists every data file the code reads, so the platform loads them.
"""

from typing import Any

from hp_fold.experiments.qollab import run_on_backend
from hp_fold.experiments.table import format_table
from hp_fold.viz.display import show_figure

DATA = ["configs/qollab.toml", "data/raw/1UBQ.cif"]

backend: Any


def main(shots: int = 1024, config: str = "configs/qollab.toml"):
    report = run_on_backend(config, shots, backend)
    problem = report.problem
    print(f"backend  {report.backend_name}")
    print(f"origin   {report.origin}")
    print(
        f"sequence {problem.seq}: {problem.n_qubits} qubits, "
        f"{int(problem.valid.sum())} valid folds of {2**problem.n_qubits}, "
        f"ground energy {problem.ground_energy:g} "
        f"({len(problem.ground_states)} ground states), penalty {problem.penalty:g}"
    )
    print(f"\n{format_table(report.rows)}\n")
    for name, reason in report.figures.skipped.items():
        print(f"skipped  {name}: {reason}")
    for name, reason in report.figures.errors.items():
        print(f"FAILED   {name}: {reason}")
    for _, figure in report.figures.figures:
        show_figure(figure)


if __name__ == "__main__":
    from qiskit.providers.basic_provider import BasicSimulator

    backend = BasicSimulator()
    main()
