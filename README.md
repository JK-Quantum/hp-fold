# HP-Fold: Quantum Algorithms for Lattice Protein Folding

Proteins fold into the lowest-energy shape their chain allows. The **HP model** reduces this to a toy problem: a chain of hydrophobic (H) and polar (P) residues is placed on a 2D square lattice without self-overlap, and every non-bonded H-H lattice contact lowers the energy by 1. Finding the minimum-energy fold is NP-hard.

We encode folds as bitstrings, define the folding energy as a diagonal cost over those bitstrings, and compare quantum algorithms against classical baselines.

If time permits, we'll run the best candidate on **IonQ** and benchmark how noise degrades the result.

> **Questions we want to answer:**
>
> _Which quantum optimization strategies find the native fold of a toy protein, how does success scale with chain length, and how much of it survives on real trapped-ion hardware?_

## Background

- **Hydrophobic collapse:** oily residues cluster in a core, polar ones stay on the surface. This is a main driver of folding.

- **HP model:** residues are `H` or `P`; the chain is a self-avoiding walk on a lattice; `energy = -(number of H-H contacts between residues that are lattice neighbors but not chain neighbors)`.

- **Parity rule:** on a square lattice, residues `i` and `j` can only touch if `|i-j|` is odd, so contacts need $|i-j| \geq 3$.

- **Designability:** a protein-like sequence has a _unique_ ground-state fold. Scanning sequences for uniqueness is cheap and scientifically meaningful.

- This is a minimal model. It does **not** predict real protein structures.

## Encoding

- Each bond direction in ${+x, +y, -x, -y}$ is 2 bits. The first bond is fixed to $+x$ (removes rotational symmetry).

- N residues need $2(N-2)$ qubits.

- Cost: $E(x) = -(HH \text{ contacts}) + \lambda \cdot (overlaps)$, with $\lambda$ large enough that the optimum is always a valid fold.

- For simulation, we tabulate the cost for all $2^n$ bitstrings. For hardware, we'll expand it into Pauli-Z terms (Walsh-Hadamard transform?, to be pondered) and build the Qiskit operator from that.

## Algorithms under comparison

| Family              | Variants                                                             | Notes                                               |
| ------------------- | -------------------------------------------------------------------- | --------------------------------------------------- |
| QAOA                | standard (expectation), CVaR-QAOA, warm-started / parameter-transfer | Natural fit for a diagonal cost                     |
| VQE                 | hardware-efficient ansatz on the cost Hamiltonian, CVaR objective    | Fewer two-qubit gates, no problem-structured layers |
| Classical baselines | brute force (ground truth), simulated annealing, random sampling     | Required for every plot                             |

It's likely that not every algorithm will be finished. See `docs/sprints.md` for the priority order.

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

```bash
uv sync
uv run pytest -q
uv run scripts/hello_world.py  # N=6 end-to-end
```

## Planned experiments

1. **Correctness:** brute-force optimum for every test sequence, cross-checked against published HP benchmark values where available.

2. **Algorithm comparison:** P(ground state) vs depth and chain length for each algorithm, with random sampling as the floor.

3. **Designability scan:** all $2^8$ sequences of length 8 (or a sample): ground-state degeneracy vs algorithm success.

4. **Real-protein inputs:** 8-10 residue windows from PDB structures mapped to H/P, plus a check that H residues are more buried in the real structure.

5. **IonQ:** run the best configuration on the IonQ simulator with a noise model, then on hardware at the smallest useful size. Report ideal vs noisy vs hardware.

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

- Toy scale. Classical HP solvers handle chains of hundreds of residues.

- Small instances are simulated, so we show method and scaling trends, not quantum advantage.

- Valid folds are a shrinking fraction of the search space as $N$ grows, so constraint handling is part of the research question.

- Hardware results are limited by qubit count, circuit depth (high-order Z terms) and shot budget.

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
