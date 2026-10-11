# HP-Fold: Quantum Algorithms for Lattice Protein Folding

Proteins fold into the lowest-energy shape their chain allows. The **HP model** reduces this to a toy problem: a chain of hydrophobic (H) and polar (P) residues is placed on a 2D square lattice without self-overlap, and every non-bonded H-H lattice contact lowers the energy by 1. Finding the minimum-energy fold is NP-hard.

We encode folds as bitstrings, define the folding energy as a diagonal cost over those bitstrings, and compare two quantum optimizers, **QAOA** and **CVaR-QAOA**, against classical baselines (random sampling, simulated annealing and an exact brute-force ground truth). The instances are 6-residue fragments of real proteins (8 qubits), small enough to simulate exactly, so every result can be checked against the true optimum. We also measure how gate noise degrades the result, and evaluate the trained circuits on a backend.

> **Questions we answer:**
>
> _Which QAOA variants find the native fold of a toy protein built from a real fragment, how does success change with circuit depth and with the objective, and how much of it survives noise?_

## Background

- **Hydrophobic collapse:** oily residues cluster in a core, polar ones stay on the surface. This is a main driver of folding.

- **HP model:** residues are `H` or `P`; the chain is a self-avoiding walk on a lattice; `energy = -(number of H-H contacts between residues that are lattice neighbors but not chain neighbors)`.

- **Parity rule:** on a square lattice, residues `i` and `j` can only touch if `|i-j|` is odd, so contacts need $|i-j| \geq 3$.

- **Designability:** a protein-like sequence has a _unique_ ground-state fold. The two fragments used here differ in this respect: one has 12 equivalent ground states, the other only 2, and the second is much harder to find.

- **Real fragments:** a window of 6 consecutive residues is cut from a PDB structure (1UBQ ubiquitin, 1CRN crambin) and mapped to H/P with a fixed hydrophobic set (A, V, L, I, M, F, W, C are `H`; everything else is `P`).

- This is a minimal model. It does **not** predict real protein structures.

## Encoding

- Each bond direction is 2 bits: right, up, left, down = `00`, `01`, `10`, `11` (codes 0 to 3). The first bond is fixed to right, which removes rotational symmetry.

- N residues need $2(N-2)$ qubits. A fold is an integer $x$ with bond $k$ in bits $2k$ and $2k+1$, so qubit $q$ is bit $q$ of $x$ (Qiskit's little-endian order, no reordering).

- Cost: $E(x) = -(HH \text{ contacts}) + \lambda \cdot (overlaps)$, where an overlap is a residue placed on an already occupied site. The penalty is $\lambda = N$, and the problem refuses to build if an overlapping fold would be among the optima (the smallest sufficient value measured was about $N/4$).

- We tabulate the cost for all $2^n$ bitstrings, together with a validity mask. The ground energy and ground states come from this table, so brute force is exact.

- For the circuits, the (mean-centred) cost is expanded into Pauli-Z strings with a fast Walsh-Hadamard transform and turned into a `SparsePauliOp` for Qiskit's `QAOAAnsatz`. For the two fragments below this gives 111 and 255 terms, up to weight 8, so a depth-1 circuit already has hundreds of two-qubit gates.

## Algorithms under comparison

| Family              | Variants                                                              | Notes                                                                    |
| ------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| QAOA                | expectation objective $\langle C \rangle$, depth $p = 1, 2, 3$         | Natural fit for a diagonal cost                                          |
| CVaR-QAOA           | CVaR objective with $\alpha = 0.05, 0.1, 0.25$, depth $p = 1, 2, 3$    | Optimizes the best $\alpha$ fraction of the distribution                 |
| Noisy QAOA          | trained exactly, measured on a depolarizing-noise simulator            | One- and two-qubit gate errors plus readout error                        |
| Classical baselines | brute force (ground truth), simulated annealing, random sampling       | Brute force gives the optimum; the other two are run for every problem   |

VQE and adiabatic algorithms are not implemented.

Every algorithm returns the full probability distribution over the $2^n$ folds. Quantum angles are trained with COBYLA and random restarts on the exact statevector; the distribution is then estimated from 1024 shots. The cost of a run is its evaluation count: objective evaluations during training plus shots (annealing counts every cost lookup).

## Repository layout

```
.
├── README.md
├── docs/
│   ├── sprints.md          # plan, tasks, deliverables, roles
│   └── (design notes, results, pitch outline as we go)
├── src/hp_fold/
│   ├── problem.py          # sequences, decoder, energy, cost vector, validity mask
│   ├── pauli.py            # cost vector -> Pauli-Z expansion (SparsePauliOp)
│   ├── classical.py        # brute force, simulated annealing, random baseline
│   ├── algos/              # qaoa.py, vqe.py, ...
│   ├── backends.py         # local simulator, noisy simulator, IonQ provider
│   ├── metrics.py          # P(ground state), P(valid), approx. ratio, CVaR
│   └── viz.py              # fold plots, result plots
├── scripts/                # experiment runners that write JSON/CSV to results/
├── notebooks/              # exploration and reports
├── data/                   # cached cost vectors, real-protein fragments
├── results/                # raw outputs
└── tests/                  # small sanity tests (brute force vs known energies)
```

## Quickstart

Needs Python 3.12 or newer.

Create and activate a virtual environment, then install `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run it from the project folder; a local Qiskit simulator is used as the backend:

```bash
python main.py
```

`main` trains QAOA and CVaR-QAOA on the problem in `configs/qollab.toml` (a fragment of ubiquitin), evaluates the trained circuits on the backend in a single job, prints a metrics table (P(ground), P(valid), evaluations) and shows the figures. Set `MPLBACKEND=Agg` to skip the figure windows.

## Experiments

**Problems.** Two 6-residue fragments, each with 8 qubits, 256 folds and 71 valid folds (28%). Penalty $\lambda = 6$.

| Fragment                      | H/P string | Ground energy | Ground states | Random P(ground) |
| ----------------------------- | ---------- | ------------- | ------------- | ---------------- |
| 1UBQ residues 1-6 (`MQIFVK`)  | `HPHHHP`   | -1            | 12            | 0.045            |
| 1CRN residues 3-8 (`CCPSIV`)  | `HHPPHH`   | -2            | 2             | 0.009            |

**Setup.** Every row is the mean over 5 seeds. QAOA runs use 5 restarts, COBYLA with at most 300 iterations per restart, and 1024 shots. $P(\text{ground})$ is the probability mass on ground-state folds, $P(\text{valid})$ the mass on overlap-free folds (random sampling gives 0.274). Each run also writes the figures: ground-state fold, energy distribution, success vs evaluation budget, optimizer convergence, success vs depth, and the $p = 1$ energy landscape. The configurations and saved results of experiments 1 to 3 are in the repository history on the `main` branch.

### 1. Baselines and QAOA depth

| Algorithm                | Evaluations | 1UBQ P(ground) | 1UBQ P(valid) | 1CRN P(ground) | 1CRN P(valid) |
| ------------------------ | ----------- | -------------- | ------------- | -------------- | ------------- |
| Random sampling          | 1,000       | 0.045          | 0.274         | 0.009          | 0.274         |
| Simulated annealing      | 20,100      | 0.830 ± 0.062  | 1.000         | 0.196 ± 0.041  | 1.000         |
| QAOA, $p = 1$            | 1,223-1,260 | 0.097 ± 0.055  | 0.542         | 0.009 ± 0.012  | 0.572         |
| QAOA, $p = 2$            | 1,887-1,931 | 0.082 ± 0.034  | 0.674         | 0.004 ± 0.006  | 0.637         |
| QAOA, $p = 3$            | 2,387-2,510 | 0.084 ± 0.058  | 0.668         | 0.016 ± 0.012  | 0.703         |

Plain QAOA learns to avoid overlaps (valid mass 0.27 to 0.54-0.70) but barely concentrates on the ground state: about twice random on 1UBQ and no better than random on 1CRN. More depth does not help. Minimizing the expected energy pulls the whole distribution down without putting weight on the minimum. Annealing is far better per run but spends 8 to 16 times more evaluations.

### 2. CVaR objective

$P(\text{ground})$ for CVaR-QAOA, by level $\alpha$ and depth:

| $\alpha$ | 1UBQ $p=1$ | 1UBQ $p=2$ | 1UBQ $p=3$ | 1CRN $p=1$ | 1CRN $p=2$ | 1CRN $p=3$ |
| -------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| 0.05     | 0.085      | 0.072      | 0.067      | 0.028      | 0.048      | 0.054      |
| 0.10     | 0.106      | 0.111      | 0.111      | 0.029      | 0.045      | 0.050      |
| 0.25     | 0.166      | 0.204      | **0.247**  | 0.026      | 0.028      | 0.030      |

CVaR is the best quantum variant. On 1UBQ, $\alpha = 0.25$ at $p = 3$ reaches 0.247 ± 0.038, about 5 times random and 3 times plain QAOA, and it improves with depth. On 1CRN it is 3 to 6 times random (0.054 at $\alpha = 0.05$, $p = 3$, against 0.009 random and 0.016 for plain QAOA), but still far below annealing (0.196). The price is validity: CVaR puts 0.28-0.49 of its mass on valid folds, less than plain QAOA (0.54-0.70), because it optimizes only the low-energy tail.

### 3. Noise

Angles are trained on the exact statevector, then the circuit is measured on a depolarizing-noise simulator. The $p = 1$ circuit for 1UBQ has 328 two-qubit gates and depth 426 on its native gate set. Default noise: 0.02% on one-qubit gates, 0.2% on two-qubit gates, 1% readout error.

| Run                  | 1UBQ P(ground) | 1UBQ P(valid) | 1CRN P(ground) | 1CRN P(valid) |
| -------------------- | -------------- | ------------- | -------------- | ------------- |
| Ideal, $p = 1$       | 0.097          | 0.542         | 0.009          | 0.572         |
| Noisy, $p = 1$       | 0.076          | 0.433         | 0.009          | 0.391         |
| Ideal, $p = 2$       | 0.082          | 0.674         | 0.004          | 0.637         |
| Noisy, $p = 2$       | 0.067          | 0.437         | 0.009          | 0.353         |
| Ideal, $p = 3$       | 0.084          | 0.668         | 0.016          | 0.703         |
| Noisy, $p = 3$       | 0.061          | 0.379         | 0.006          | 0.321         |

Noise sweep on 1UBQ at $p = 1$, by two-qubit error rate (the one-qubit error is 0.01%, 0.02% and 0.1% for the three noisy rows, readout error 1% throughout):

| Two-qubit error | P(ground) | P(valid) |
| --------------- | --------- | -------- |
| ideal           | 0.097     | 0.542    |
| 0.05%           | 0.084     | 0.495    |
| 0.2%            | 0.076     | 0.433    |
| 1%              | 0.052     | 0.306    |
| random sampling | 0.045     | 0.274    |

Both metrics fall steadily as the error rate grows, and noise cancels the benefit of depth: on 1UBQ the ideal validity rises from $p = 1$ to $p = 3$ (0.54 to 0.67) while the noisy validity falls (0.43 to 0.38). At a 1% two-qubit error the output is almost indistinguishable from random sampling.

### 4. Evaluation on a backend

`main.py` runs the deployable version of the experiment on the 1UBQ fragment. It trains QAOA ($p = 1, 2$) and CVaR-QAOA ($p = 2$, $\alpha = 0.05, 0.1, 0.25$) on the exact statevector, then sends all trained circuits to the backend together in a single job. The backend results are reported next to the ideal ones as `qaoa_backend` and `qaoa_cvar_backend`, with the same metrics table and figures. On a noiseless local simulator the backend results agree with the ideal ones within shot noise (differences of a few hundredths in $P(\text{ground})$ at 1024 shots). No real-hardware result is reported.

**Summary.** Quantum optimizers raise the ground-state probability well above random sampling on a fragment with many ground states, and CVaR-QAOA is clearly better than plain QAOA. None of them matches simulated annealing, and we claim no quantum advantage.

## Status

| Milestone                         | Status      |
| --------------------------------- | ----------- |
| Classical core + brute force      | not started |
| QAOA statevector baseline         | not started |
| Qiskit circuits + Pauli expansion | not started |
| Algorithm comparison              | not started |
| IonQ noise benchmark              | not started |
| Final figures + pitch             | not started |

## Limitations

- Toy scale. The problems have 8 qubits and 256 folds, which brute force enumerates instantly, and classical HP solvers handle chains of hundreds of residues.

- Everything is simulated. The quantum angles are trained on the exact statevector, so the evaluation counts do not include the shot cost a real device would charge during training, and no real-hardware result is reported. The backend path was only tested on a noiseless simulator.

- The noise model is a generic depolarizing model (gate and readout errors), not calibrated to any device.

- Two fragments and 5 seeds per row. The spread is large (for example 0.097 ± 0.055 for plain QAOA on 1UBQ), so small differences between rows are not significant.

- Valid folds are a shrinking fraction of the search space as $N$ grows (28% at $N = 6$), so constraint handling matters more at larger sizes. We did not study scaling in chain length, a designability scan, or a penalty sweep.

- Real circuits are deep: the cost has Z terms up to weight 8, so even $p = 1$ needs hundreds of two-qubit gates, which limits what hardware can do.

- VQE and adiabatic algorithms were not implemented, and no run on real trapped-ion hardware was done.

## References

> TODO: Verify titles, authors and provide URLs before the final write-up.

- Dill, K. A. (1985). Theory for the folding and stability of globular proteins. _Biochemistry_ 24(6):1501-1509.

- Perdomo-Ortiz et al. (2012). Finding low-energy conformations of lattice protein models by quantum annealing. _Scientific Reports_.

- Fingerhuth, Babej, Wittek (2018). Open source software in quantum computing / QAOA for lattice protein folding.

- Robert et al. (2021). Resource-efficient quantum algorithm for protein folding. _npj Quantum Information_.

- Barkoutsos et al. (2020). Improving variational quantum optimization using CVaR. _Quantum_.

- Farhi, Goldstone, Gutmann (2014). A quantum approximate optimization algorithm.

- Qiskit documentation (`SparsePauliOp`, `QAOAAnsatz`, `PauliEvolutionGate`) and the qiskit-ionq provider docs.

- Kyte, J. & Doolittle, R. F. (1982). A simple method for displaying the hydropathic character of a protein. _J. Mol. Biol._ 157(1):105-132. PMID 7108955.

## Team

- Kauê Miziara
- Júlia Santos

> TODO: add names and links
