# Sprint Plan

- **Team:** 2 developers.
- **Goal:** a working lattice-protein-folding pipeline with several quantum algorithms compared against classical baselines, then a noise benchmark on IonQ.

## Roles

|         | Dev A: Problem & Analysis                                                                     | Dev B: Quantum & Hardware                      |
| ------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| Owns    | `problem.py`, `classical.py`, `metrics.py`, `viz.py`, designability scan, real-protein inputs | `pauli.py`, `algos/`, `backends.py`, IonQ runs |
| Reviews | B's results against brute force                                                               | A's cost vector against Pauli expansion        |

Both write tests for their own modules. Merge to `main` at the end of each sprint.

## The contract between the two roles

Agree on this in Sprint 0 and don't change it without telling the other person.

```python
# problem.py (Dev A)
class HPProblem:
    seq: str                    # e.g. "HPPHPH"
    n_qubits: int               # 2*(N-2)
    cost: np.ndarray            # shape (2**n_qubits,), float
    valid: np.ndarray           # shape (2**n_qubits,), bool
    ground_energy: float
    ground_states: np.ndarray   # indices of optimal bitstrings
    def decode(self, x: int) -> np.ndarray   # (N, 2) lattice positions

# pauli.py (Dev B)
def cost_to_pauli(cost: np.ndarray, tol: float = 1e-9) -> SparsePauliOp
```

Bit convention: the bitstring integer $x$ encodes bond $k$ in bits $2k, 2k+1$.
Write down Qiskit's little-endian ordering in one comment and test it.

> Mismatched bit order is the most likely silent bug.

## Time budget

| Sprint               | Hours  | Dev A                               | Dev B                                            |
| -------------------- | ------ | ----------------------------------- | ------------------------------------------------ |
| 0. Setup & contract  | 1      | repo, env, `problem.py` skeleton    | Qiskit/IonQ access check, `backends.py` skeleton |
| 1. Core + baselines  | 2.5    | cost vector, brute force, SA, tests | NumPy QAOA statevector, Pauli expansion          |
| 2. Algorithms        | 3      | metrics, experiment runner, plots   | QAOA + CVaR, VQE, adiabatic in Qiskit            |
| 3. Experiments       | 2      | designability scan, real fragments  | scaling runs, algorithm comparison               |
| 4. IonQ benchmark    | 2.5    | analysis of noisy vs ideal          | noise-model and hardware runs                    |
| 5. Packaging & pitch | 1      | figures, README results             | slides, reproducibility check                    |
| **Total**            | **12** |                                     |                                                  |

Hours are effective hours for the team, not wall-clock.
Sprints 2 and 3 overlap on purpose; start experiments as soon as one algorithm works.

---

## Sprint 0: Setup & contract (1 h)

**Tasks**

- [x] A: create repo layout from the README, setup `uv`, `.gitignore`.
- [x] A: search and download molecule datasets.
- [ ] B: verify the IonQ API key works, list the available backends (simulator, noise models, QPU).
- [ ] B: find qubit limits, job queue behavior and the credit/shot budget.
- [ ] B: submit one trivial 2-qubit circuit to the IonQ simulator through Qiskit to prove the path end to end.
- [x] Both: agree on the interface above and the bit-ordering convention.

**Deliverable:** repo pushed, both devs can run `pytest` and one IonQ simulator job.

## Sprint 1: Core + baselines (2.5 h)

**Tasks**

- [x] A: load mmCIF into CA-only chain records.

- [x] A: convert chains to one-letter sequences and H/P strings, with a Kyte-Doolittle check.

- [x] A: extract contact capable windows from H/P strings.

- [x] A: `problem.py`: decoder, energy (contacts + overlap penalty), tabulated cost vector, `valid` mask, ground states. Assert the optimum is valid.

- [x] A: `classical.py`: brute force (trivial from the cost vector), simulated annealing on bitstrings, uniform random sampling baseline.

- [x] A: tests: a hand-checked `HPPH` fold has $E=-1$, a straight chain has $E=0$, symmetry degeneracy counted correctly. Add one or two published HP benchmark energies if you can find them quickly.

- [ ] B: NumPy QAOA statevector (diagonal phase layer + X mixer), expectation and CVaR objectives, COBYLA with restarts. This is the fast reference implementation and the debugging oracle for the Qiskit version.

- [ ] B: `pauli.py`: fast Walsh-Hadamard transform of the cost vector into Z-string coefficients, prune near-zero terms, return `SparsePauliOp`. Test: `SparsePauliOp.to_matrix().diagonal()` equals `cost`.

- [ ] B: record the **number of Pauli terms and their weights** for N=5, 6, 7, 8. This predicts circuit depth on IonQ and decides how large the hardware run can be.

**Deliverable:** `scripts/hello_world.py` for N=6 prints the optimal fold, P(GS) for QAOA p=1..3, and the random baseline.

**Go/no-go check:** if the Pauli expansion at N=6 has an unmanageable term count, switch plan for hardware to N=5 and note it. Don't spend time optimizing it yet.

## Sprint 2: Algorithms in Qiskit (3 h)

Priority order. Stop adding algorithms when the clock says so.

| Priority | Algorithm                      | Task                                                              | Done when                                       |
| -------- | ------------------------------ | ----------------------------------------------------------------- | ----------------------------------------------- |
| 1        | QAOA (Qiskit)                  | `QAOAAnsatz` from the `SparsePauliOp`, run on Aer, optimizer loop | matches the NumPy version within sampling noise |
| 2        | CVaR-QAOA                      | same circuit, CVaR objective, alpha in {0.05, 0.1, 0.25}          | P(GS) comparison vs standard QAOA               |
| 3        | VQE, hardware-efficient ansatz | `EfficientSU2` or `RealAmplitudes`, CVaR objective                | runs at N=6, reports P(GS)                      |

**Tasks**

- [ ] B: implement algorithms 1-3 first, behind a common interface: `run(problem, backend, **cfg) -> Result` where `Result` holds probabilities/counts, parameters, optimizer history and circuit stats (depth, two-qubit gate count after transpile).

- [x] A: `metrics.py`: P(ground state), P(valid fold), approximation ratio, best-sampled energy, success versus number of shots.

- [x] A: `scripts/run_experiment.py`: config-driven runner (algorithm, N, p, seed) writing JSON to `results/`. Deterministic seeds.

- [ ] A: `viz.py`: fold picture, P(GS) vs p, bitstring probability histogram colored by validity.

- [ ] B: record circuit depth and two-qubit gate count per algorithm and N. These numbers are the bridge to the IonQ sprint.

**Deliverable:** one comparison table (algorithm x p) at N=6 with random and simulated-annealing baselines.

## Sprint 3: Experiments (2 h)

Choose from this list in order. Do not start the next item before the previous has a saved figure.

1. [ ] **Scaling:** N = 6, 8 (and 10 if the simulator can take it) for the best two algorithms. P(GS) vs N and vs p.

2. [ ] **Designability scan (A):** all length-8 sequences, or a random sample of ~100. For each: ground-state degeneracy (after removing the reflection symmetry), plus QAOA success. Question: _are uniquely-folding sequences easier for the quantum algorithms?_

3. [ ] **Penalty sweep (A):** vary lambda, show validity vs ground-state probability.

4. [ ] **Real fragments (A, optional):** extract 8-mers from a PDB entry, map to H/P (document the mapping), include 3-5 of them in the scan. A burial-vs-hydrophobicity plot from the real structure goes on one slide as justification for the model.

5. [ ] **Parameter transfer (B, optional):** initialize layer p+1 from layer p, report optimizer evaluations saved.

**Deliverable:** 3-4 figures with baselines, saved as PNG/PDF in `results/figures/`.

## Sprint 4: IonQ noise benchmark (stretch, 2.5 h)

**Entry condition (decide at the end of Sprint 3):** at least one algorithm works in Qiskit at N=5 or 6 with a saved result. If not, skip this sprint and spend the time hardening Sprints 2 and 3 instead.

**Tasks (in order)**

- [ ] B: pick the hardware configuration. Constraints: qubit count ($2(N-2)$), transpiled two-qubit gate count, shot budget. Start at N=5 (6 qubits) and p=1. Fixed parameters from the simulator's best run, so hardware only does evaluation (no optimizer loop on QPU).

- [ ] B: run on the **IonQ simulator without noise**; confirm it matches Aer.

- [ ] B: run on the **IonQ simulator with the hardware noise model** (if offered).

- [ ] B: run on the **QPU** with the shot budget agreed in Sprint 0. Submit early; queues can be slow. Keep job IDs in `results/ionq_jobs.json`.

- [ ] A: analysis: P(GS), P(valid), energy distribution for ideal vs noise-model vs hardware; how P(GS) decays with circuit depth or number of Pauli terms.

- [ ] Both: error mitigation, such as readout correction or post-selecting valid folds. Post-selection is cheap here because validity is checkable classically.

- [ ] Both: a "resource usage" slide: qubits, circuits, shots, jobs, queue time. Show deliberate choices (why this size, why fixed parameters, why these shots).

**Deliverable:** one figure: ideal vs noisy simulator vs hardware for the same circuit.

**Likely failure modes**

- Too many Pauli terms makes the circuit too deep. Reduce N, drop small terms and report the approximation error, or use p=1.
- Queue time. Submit hardware jobs by the end of Sprint 3 and work on other things while they run.
- Budget overrun. Run the simulator first; use the QPU only for the final configuration.

## Sprint 5: Packaging & pitch (1 h)

- [ ] A: fill the **Status** table and a results section in `README.md`; make sure `pytest` and `scripts/hello_world.py` pass from a clean environment.

- [ ] A: check dependencies, remove any key from history, add a "results" section in the README listing which script produces which figure.

- [ ] Both: proofread the repository, check links, confirm resources and references.

- [ ] B: slide presentation: why proteins fold -> HP model in one picture -> encoding and qubit count -> algorithm comparison with baselines -> IonQ results and resource use -> limits and scaling outlook.

- [ ] Both: rehearse and record a 3-minute pitch. Prepare an answer for "where is the quantum advantage?" (honest answer: none claimed; this is a method and noise study at toy scale).

## Cut lines

If time runs short, cut from the bottom:

| Priority               | Item                                                                                      |
| ---------------------- | ----------------------------------------------------------------------------------------- |
| Must                   | N=6 pipeline, brute-force baseline, QAOA in Qiskit, one comparison figure, README results |
| Should                 | CVaR-QAOA, VQE, scaling to N=8, one IonQ simulator run                                    |
| Could                  | designability scan, IonQ QPU run with noise comparison, adiabatic                         |
| Won't (this hackathon) | Real-protein validation, 3D lattices, position/one-hot encodings                          |

## Definition of done

- `pytest` passes; every quantum result has a brute-force baseline next to it.
- Every figure is regenerated by a script, not by hand.
- The README states the toy-scale caveats and the IonQ resources used.
- No API keys in the repository.

## Risk register

| Risk                                 | Mitigation                                                                              |
| ------------------------------------ | --------------------------------------------------------------------------------------- |
| Qiskit bit-ordering bug              | Test: Pauli expansion diagonal equals tabulated cost; compare Qiskit QAOA to NumPy QAOA |
| IonQ access/quota issues             | Verified in Sprint 0                                                                    |
| Optimizer stuck in poor local minima | Multiple restarts, CVaR, parameter transfer                                             |
| Simulator too slow at N=10           | Cap at N=8 for Qiskit; use NumPy statevector for larger scans                           |
| Scope creep                          | Cut lines above; check the clock at the end of each sprint                              |
