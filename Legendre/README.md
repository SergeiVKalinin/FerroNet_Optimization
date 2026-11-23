# Legendre control: polynomial processing trajectories

> **FerroNet Optimization — Legendre branch.** Read this before running `FerroSim_Legendre_v3.ipynb`. This README documents the research objective, implementation, figures, and open validation requirements; it does not report simulation results that have not been measured.

## The common FerroSim scientific experiment

**Purpose.** Discover electric-field processing histories that produce physical states with large terminal polarization curl under a fixed phase-field/Ginzburg–Landau-type dynamics model. A control waveform `E(t)` is applied to a `20 × 20` lattice with quenched defects; the simulator evolves `P_x(t,x,y)` and `P_y(t,x,y)`, and the optimizer is rewarded for the **simulator-measured** terminal value

```text
R(E, d) = sum_{x,y} | dP_y/dx - dP_x/dy |  at the terminal time,
```

where `d` identifies the disorder realization. The solver uses RK4 by default in v3. The surrogate DKL model is used to **guide the selection of new simulations**, not to replace measured rewards. This is a closed-model novelty-discovery benchmark: the physical laws and allowed operations are specified, but the high-performing trajectories and polarization outcomes are unknown.

The core scientific comparisons have **three axes**:

1. **Functional quality:** how much physical reward is obtained per attempted oracle evaluation?
2. **Structural novelty/complexity:** how complex is the *rendered physical waveform*, and where is the empirical reward–complexity Pareto frontier?
3. **Robustness/transfer:** does a waveform discovered for one disorder map remain effective when the defect locations change?

A candidate is called "best" only relative to the **finite set of valid controls actually evaluated**. Neither the observed Pareto front nor a high-reward waveform certifies global optimality, experimental feasibility, or a genuinely new physical phenomenon.

## Files in this directory

- **[FerroSim_Legendre_v3.ipynb](FerroSim_Legendre_v3.ipynb) — recommended v3 entry point** for the standardized experiments and figures.
- `FerroSim_Legendre_20Disorders_Standalone_FullAnalysis.ipynb` — prior or historical notebook, preserved for traceability.
- `FerroSim_Legendre_20_disorders_selfcontained.ipynb` — prior or historical notebook, preserved for traceability.
- [Repository-wide v3 guide](../FerroSim_v3_README.md) — consolidated Anaconda environment and F01–F20 descriptions.
- [Shared figure source](../FerroSim_v3_figures.py) — embedded into every v3 notebook for standalone execution.

## Running the v3 notebook with Anaconda

1. Obtain this repository (clone it or download its ZIP). The notebook itself does not need network access when running.
2. Create a clean environment. In **Anaconda Prompt**:

   ```bash
   conda create -n ferrosim-v3 python=3.11 numpy pandas scipy matplotlib numba jupyterlab ipython pip
   conda activate ferrosim-v3
   # Install a suitable PyTorch build for your platform using https://pytorch.org/get-started/locally/
   jupyter lab
   ```

3. In JupyterLab, open `Legendre/FerroSim_Legendre_v3.ipynb`, select the `ferrosim-v3` kernel, and edit the **first configuration code cell**.
4. Use **Kernel → Restart Kernel and Run All Cells**. Keep the notebook in its own fresh kernel, because different representations write Python modules with the same import names.
5. Look for generated `FerroSim_Legendre_v3/` runtime and results directories under Jupyter's **current working directory**. They contain the embedded modules, disorder maps, measured-evaluation histories, CSV summaries, and `figures/` outputs. The results path includes a configuration/source hash.

Configuration in the first code cell:

| Setting | Meaning |
|---|---|
| `MODE="QUICK"` / `"FULL"` | Small test search vs full training/search budget; the oracle remains RK4 |
| `N_DISORDERS` | Number of physical disorder maps; `1` or `20` or another positive integer |
| `N_SEEDS` | Independent optimizer stochastic seeds **per disorder** |
| `GENERATIONS` | Number of search generations, independent of `N_DISORDERS` |
| `INITIAL_CONTROLS` | Initial physical controls (v3 modal implementations require at least 32) |
| `RUN_TRANSFER` | Optional O(N²)-like extra simulations for source-to-target transfer, meaningful for N > 1 |
| `RUN_REFINEMENT` | Optional extra RK4 temporal-fidelity evaluations |
| `RUN_PHYSICAL_DIAGNOSTICS` | Calculate spatial and time-dependent polarization/curl diagnostics |
| `PLOT_ALL_DISORDERS` | Generate detailed per-disorder F19 and all-disorder F20 figures |

For **N = 1**, the notebook is intended to produce the individual-run/core figures without pretending to estimate disorder variability. For **N > 1**, additional ensemble comparisons become meaningful. The first disorder is the **historical 19-defect reference**, and additional maps are *spatial rearrangements of the same 19 defect vectors*. They are not independent defect-strength distributions or new physics models.

**Run accounting.** The modal FULL protocol uses 64 initial controls and 32 evaluations per generation for each method; the variable-topology FULL notebooks use the same default counts. When initial evaluations are shared/archived across multiple modal methods, the total number of method-history rows is **not** the number of unique physical oracle calls. QUICK and FULL searches should not be mixed in the same performance comparison.

## Standardized figures and the requested last cell

Every v3 notebook has the **same figure identifiers and scientific meanings**, even though the optimized controls are different:

| Figures | Meaning |
|---|---|
| **F01–F03** | Disorder maps, measured reward vs evaluation, reward–complexity Pareto front for the reference disorder |
| **F04–F06** | Per-disorder incumbent waveform evolution, representative best/simple controls, incumbent reward/complexity/peak/RMS field |
| **F07** | DKL predictions vs measured rewards and predictive-interval coverage; these are *selected proposals*, not an IID test set |
| **F08–F10** | Final polarization and curl maps; curl/polarization histories; integration and waveform-sampling diagnostics |
| **F11–F14** | Multi-disorder rewards, paired method differences, winning complexity, reward retained with simple controls (N > 1) |
| **F15** | Optional additional measured cross-disorder transfer (N > 1 and `RUN_TRANSFER=True`) |
| **F16–F18** | Variance components, target-attainment curves, and disorder descriptors (N > 1) |
| **F19** | **For each disorder:** the best-so-far *measured* waveform history for every algorithm and its own reward–complexity Pareto front |
| **F20** | **Across all disorders:** an aligned grid of method-specific and pooled empirical Pareto fronts |

For the **last notebook cell**, set `ONLY_DISORDERS=None` to generate all maps, or provide a list of map IDs, for example `["reference","d00"]`. Adjust `HISTORY_SNAPSHOTS` for how many best-so-far snapshots appear per method. Once the evaluation history is loaded, this cell can be rerun independently of the optimization. F19 and F20 use actual simulator results, omitting invalid rewards.

**Common complexity measure:** the number of resolved extrema of `E(t)`, with a 2%-of-peak-to-peak prominence threshold, minimized against maximized physical reward. This is useful for cross-representation plots, but **not sufficient** to equate implementation cost; amplitude, slew, duration, and control grammar should also be reported.

## Interpretation, known limitations, and verification status

- The simulator uses **open-edge polarization coupling**, but the reward is computed with **periodic finite differences** for curl. Separate interior and boundary contributions before attributing reward to genuine vortices or textures.
- **Integration fidelity and waveform sampling fidelity are different**. Increasing RK4 subdivisions does not restore operations or pulses shorter than the 900-sample field resolution. F10 diagnoses these separately, but it is *not* a complete variable-time-grid convergence test.
- Large applied fields can drive numerical artifacts or exploit simulator approximations; compare optimized controls at matched peak/RMS field, overall exposure, and complexity, and independently revalidate winners.
- The code includes measured-evaluation histories, candidate validity flags, and saved configurations. Results and visualizations are produced **when the notebook is executed**; they are **not precomputed claims of algorithm superiority**.
- **Current validation status:** the v3 notebook has been uploaded and structurally/source-checked. A complete clean-environment Anaconda execution and the full N=20 scientific campaign **have not been independently verified here**. Run QUICK first, inspect outputs, then validate a FULL N=1 case before trusting ensemble-level conclusions.

## Research-specific focus

We ask whether **global polynomial expansions** provide an effective family of time-dependent electric-field trajectories for discovering high-reward polarization configurations, and how that family compares with Fourier, spline, and variable-topology controls at matched physical cost.

The control uses Legendre polynomials,

```text
E(t)/Ec = sum_{n=0}^{35} a_n P_n(2*t/T - 1)
```

with **36 physical polynomial coefficients** subject to four homogeneous endpoint field/slope constraints; the allowed subspace therefore has **32 independent optimization coordinates**. The coordinate basis is normalized in a Legendre L2 metric.

### Optimization methods

The current modal benchmark compares **GA–DKL**, **QD–DKL**, **trained GFlowNet–DKL**, and **guided diffusion–DKL**. All use a common oracle, per-run evaluation budget and DKL-based candidate selection. The proposal methods are genuinely different search algorithms but share part of the exploration policy; track which proposer actually generated winning candidates.

### What to inspect beyond the shared figures

Legendre-specific diagnostics should include polynomial-order coefficients, the correctly normalized Legendre modal-energy spectrum, control waveform smoothness, resolved extrema, peak/RMS drive, and the difference between low-order-looking and genuinely simple *physical* waveforms. Compare results with Fourier and spline under matched rendered control constraints, rather than coefficient norms alone.

### Relation to earlier notebooks

The folder also contains the earlier `FerroSim_Legendre_20Disorders_Standalone_FullAnalysis.ipynb` and `FerroSim_Legendre_20_disorders_selfcontained.ipynb`. The revised Legendre notebook in a separate `Legendre Try` folder is another related implementation, **not** the v3 file documented here. The v3 notebook is the intended standardized entry point.
