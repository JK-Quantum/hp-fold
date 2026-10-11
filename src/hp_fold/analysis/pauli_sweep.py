import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from hp_fold.folding.problem import HPProblem
from hp_fold.quantum.pauli import cost_to_pauli, pauli_stats


class PauliSweep:
    """Pauli statistics over several H/P sequences for each chain length N."""

    def __init__(self, *, samples: int = 10, tol: float = 1e-9, seed: int = 0) -> None:
        self._samples = samples
        self._tol = tol
        self._seed = seed

    def sequences_for(self, n: int, rng: np.random.Generator) -> list[str]:
        fixed = ["H" * n, ("HP" * n)[:n], ("PH" * n)[:n]]
        seen = set(fixed)
        target = min(self._samples, 2**n - 1 - len(seen))
        random_ones: list[str] = []
        while len(random_ones) < target:
            seq = "".join(rng.choice(list("HP"), size=n))
            if seq not in seen and "H" in seq:
                seen.add(seq)
                random_ones.append(seq)
        return fixed + random_ones

    def run(self, sizes: list[int]) -> dict[str, dict]:
        rng = np.random.default_rng(self._seed)
        report: dict[str, dict] = {}
        for n in sizes:
            rows = []
            for seq in self.sequences_for(n, rng):
                problem = HPProblem.from_sequence(seq)
                stats = pauli_stats(cost_to_pauli(problem.cost, self._tol))
                row = {"sequence": seq, **asdict(stats)}
                row["weight_histogram"] = {
                    str(w): c for w, c in stats.weight_histogram.items()
                }
                rows.append(row)
            report[str(n)] = {
                "n_qubits": rows[0]["n_qubits"],
                "summary": self.summarize(rows),
                "sequences": rows,
            }
        return report

    @staticmethod
    def summarize(rows: list[dict]) -> dict:
        def stat(key: str) -> dict:
            values = [r[key] for r in rows]
            return {
                "min": min(values),
                "mean": float(np.mean(values)),
                "max": max(values),
            }

        return {
            "n_sequences": len(rows),
            "n_terms": stat("n_terms"),
            "max_weight": stat("max_weight"),
            "mean_weight": stat("mean_weight"),
            "cnot_per_layer": stat("cnot_per_layer"),
        }

    @staticmethod
    def save(report: dict, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "terms.json"
        path.write_text(json.dumps(report, indent=2))
        return path
