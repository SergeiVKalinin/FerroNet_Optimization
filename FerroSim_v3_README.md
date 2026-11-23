# FerroSim v3 — unified Anaconda notebook suite

## Five new notebooks
- [Fourier/FerroSim_Fourier_v3.ipynb](Fourier/FerroSim_Fourier_v3.ipynb)
- [Legendre/FerroSim_Legendre_v3.ipynb](Legendre/FerroSim_Legendre_v3.ipynb)
- [Spline/FerroSim_Spline_v3.ipynb](Spline/FerroSim_Spline_v3.ipynb)
- [Segments/FerroSim_Segments_v3.ipynb](Segments/FerroSim_Segments_v3.ipynb)
- [Grammar/FerroSim_Grammar_v3.ipynb](Grammar/FerroSim_Grammar_v3.ipynb)

The original notebooks are preserved. **These v3 notebooks have undergone a source-level and structural audit, but complete physical experiments have not been executed or benchmarked as part of this GitHub update.** Run the QUICK configuration on a clean Anaconda Jupyter kernel before scheduling expensive FULL campaigns. The notebooks contain no precomputed physical optimization results.

## Anaconda / JupyterLab

```bash
conda create -n ferrosim-v3 python=3.11 numpy pandas scipy matplotlib numba jupyterlab ipython pip
conda activate ferrosim-v3
# Install a compatible CPU or CUDA build of PyTorch following the PyTorch installation page.
jupyter lab
```

Open each notebook in JupyterLab, select the **ferrosim-v3** environment's kernel, and choose **Kernel > Restart Kernel and Run All Cells**. Do not mix method notebooks in the same running kernel: each creates modules called `representation`, `oracle`, etc. Each writes its Python modules, result archives, tables, and figures to a dedicated `FerroSim_<METHOD>_v3/` directory under the current working directory. No GitHub access is required for the notebooks' own execution.

## Configuration

Edit the first code cell:

- `MODE="QUICK"` tests a small search with the full RK4 physics; use `"FULL"` for scientific training budgets.
- `N_DISORDERS=1` works; `20` is the nominal ensemble; any positive integer is accepted. The first map is the historical reference; remaining maps are randomized permutations of the same 19 defect vectors.
- `N_SEEDS` is an independent number of optimizer seeds.
- `GENERATIONS` sets the number of generations without changing the number of disorder realizations.
- `RUN_TRANSFER=True` enables measured cross-map champion re-evaluation (scaling approximately quadratically with N).
- `RUN_REFINEMENT=True` enables RK4 higher-integration-fidelity comparison.
- `PLOT_ALL_DISORDERS=True` appends F19–F20.

The FULL budgets are identical in the five notebooks in numbers of attempted physics evaluations: 64 shared initial controls, 32 attempts per generation, and 32 generations. Methods within each representation use the same budget, but algorithm architectures differ between the modal and variable-topology notebooks. QUICK budgets are intentionally smaller in Segments/Grammar than in the modal methods, and QUICK results should **not** be treated as a fair cross-representation benchmark.

## Standard figures

| ID | Core scientific output |
|---|---|
| F01 | Disorder maps |
| F02 | Best reward versus physics evaluations |
| F03 | Reference-map measured Pareto fronts |
| F04 | Per-disorder incumbent waveform histories |
| F05 | Highest-reward and simple controls |
| F06 | Incumbent reward, complexity, peak/RMS field versus evaluation |
| F07 | Surrogate prediction calibration on selected proposals |
| F08 | Terminal polarization and spatial curl |
| F09 | Curl and mean-polarization temporal traces |
| F10 | Integration and waveform-sampling diagnostics |
| F11 | Between-disorder reward distributions (N>1) |
| F12 | Paired method differences on the same disorder (N>1) |
| F13 | Winner complexity versus disorder (N>1) |
| F14 | Reward retained by simple controls (N>1) |
| F15 | Cross-disorder transfer, optional (N>1) |
| F16 | Disorder vs optimizer-seed variance (N>1) |
| F17 | Evaluation cost to reach a common target (N>1) |
| F18 | Disorder geometry and best reward (N>1) |
| F19 | Per-map measured champion waveform histories and algorithm Pareto fronts |
| F20 | All-map grid of empirical complexity–reward Pareto fronts |

F19 and F20 are in the **last code cell** of each notebook and can be rerun without repeating physical optimization. Change `ONLY_DISORDERS` and `HISTORY_SNAPSHOTS` there. Each map is analyzed separately; the pooled Pareto curve is an empirical nondominated envelope of actually evaluated controls.

## Important distinctions

- `Spline_v3` is a **new** self-contained cubic B-spline representation (36 physical coefficients, 32 interior controls and zero field/slope endpoints), not a reproduction of the historical nonstandalone `Spline_descriptors.ipynb`.
- The Segments implementation's sequential-GFlowNet-inspired and denoising-diffusion-inspired proposers are **heuristics**. They are not trained GFlowNet/diffusion models.
- The reward is the terminal sum of absolute periodic spatial curl, while the polarization evolution uses open-edge finite interactions; boundary contributions must be reported.
- The RK4 integration test in F10 changes solver fidelity but cannot establish full physical-waveform discretization convergence without modifying the oracle time grid.
- No method's output is claimed to be globally optimal or a validated physical novelty based on this code alone.

## Shared source

The source in [FerroSim_v3_figures.py](FerroSim_v3_figures.py) is embedded inside each notebook as `shared_figures.py`, so no external helper module is needed at runtime.
