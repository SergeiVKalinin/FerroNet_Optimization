# FerroSim v4 — Legendre Try analytics and texture evolution

The five new v4 notebooks are based on the latest Legendre Try statistical-analysis workflow and add the same physical texture history to each control representation.

| Representation | Standalone notebook | Control description |
|---|---|---|
| Legendre | [FerroSim_Legendre_v4.ipynb](Legendre/FerroSim_Legendre_v4.ipynb) | 32 independent Legendre modal coordinates |
| Fourier | [FerroSim_Fourier_v4.ipynb](Fourier/FerroSim_Fourier_v4.ipynb) | 32 independent sine-Fourier coordinates |
| Spline | [FerroSim_Spline_v4.ipynb](Spline/FerroSim_Spline_v4.ipynb) | 32 interior cubic B-spline coordinates |
| Segments | [FerroSim_Segments_v4.ipynb](Segments/FerroSim_Segments_v4.ipynb) | Variable-length piecewise-linear field segments |
| Grammar | [FerroSim_Grammar_v4.ipynb](Grammar/FerroSim_Grammar_v4.ipynb) | Variable-length programs made of physical operations |

The original v3 notebooks remain available. Legendre Try itself was not overwritten.

## Scientific goal

Find time-dependent electric-field controls that maximize the terminal sum of absolute curl on the 20×20 FerroSim polarization lattice. The optimization reward is unchanged in v4. Fourier, Legendre and Spline use GA–DKL, trained GFlowNet–DKL, guided diffusion–DKL and QD–DKL. Segments uses GA/QD and two *heuristic* sequential/denoising-inspired methods; Grammar compares simple/full operation vocabularies with GA/QD. Statistical figures and physical outputs are standardized, but these methods are not algorithmically identical.

The disorder ensemble consists of the original 19-site reference defect map plus reproducible *spatial rearrangements* of those same defect vectors; it does not sample new defect-strength distributions.

## Anaconda use

Create a clean Python 3.11 conda environment including NumPy, pandas, SciPy, Matplotlib, Numba, JupyterLab/IPython, and a platform-compatible PyTorch installation. Anaconda Navigator can launch JupyterLab using this environment.

Open one v4 notebook and choose **Restart Kernel → Run All**. It is standalone: embedded source cells write Python modules to a local runtime directory. The first Python cell controls all experiments:

- MODE: QUICK or FULL, both with RK4 physics and the same numerical resource limit.
- N_DISORDERS: any positive integer, including 1 or 20.
- SEEDS: a distinct list of nonnegative optimizer seeds, such as [0] or [0,1,2].
- GENERATIONS: 0 for initialization, 1 for small search, 32 for the full campaign.
- INITIAL_CONTROLS: minimum 2 for a regression test; 64 recommended for research.
- ANALYZE_EACH_DISORDER, RUN_CROSS_DISORDER_TRANSFER, RUN_HIGHER_FIDELITY_VALIDATION and VALIDATE_ALL_DISORDERS: optional analyses from Legendre Try.
- RUN_TEXTURE_ANALYSIS, TEXTURE_PLOT_EACH_DISORDER, TEXTURE_MAX_SNAPSHOTS, TEXTURE_PCA_REFERENCE_ONLY: new texture output settings.

The default is MODE=QUICK, N_DISORDERS=20, SEEDS=[0], GENERATIONS=1, INITIAL_CONTROLS=64. **This is still a 20-map physics run.** A recommended first test is N_DISORDERS=1 and GENERATIONS=0, followed by GENERATIONS=1.

All control representations receive 32 new attempted physical queries per generation per method, independently of N. Modal methods reuse one physically evaluated initialization across methods; Segments and Grammar use native runners that evaluate their initial populations separately. The budgets and cache provenance are audited in exported tables.

## Statistical analysis inherited from Legendre Try

All five notebooks preserve the same general section order and produce:

1. Experiment configuration, reference defects, representation invariants and a disorder manifest.
2. Recorded oracle histories, valid/invalid trials, source fingerprints and physical query budget.
3. Best measured reward versus physical evaluations; normalized gain from the best valid initial control.
4. Exact 95%/99% observed target-attainment evaluation indices.
5. Per-algorithm and per-disorder measured reward–waveform-complexity Pareto fronts.
6. Best/simple waveforms, evolution of descriptor coefficients/encoded controls and physical waveforms.
7. Reward retained with 0, 1 or 3 resolved turns and minimum complexity retaining 95% or 99% gain.
8. Physical waveform total variation, peak field and curl boundary fraction.
9. Disorder/seed variability and six paired method contrasts, with disorder-block bootstrap intervals.
10. Candidate proposal origin, acquisition/selection provenance and DKL calibration on measured trials.
11. Optional all-to-all source–target transfer and higher-RK4-fidelity polarization/curl snapshots.
12. CSV, JSON and NPZ archives of the experiments, figures, checkpoints and best controls.

For N=1 or only one seed, undefined between-disorder/within-seed confidence intervals are explicitly left undefined.

## NEW: Physical texture evolution (section 18)

Each optimizer's **best-so-far measured control is tracked at every generation**. Whenever the incumbent reward increases, v4 reruns the winning waveform in the physical oracle and saves its final polarization state. Later generations with the same winner reference that state instead of storing a duplicate.

Per unique improved control, the NPZ archive stores the full P_x and P_y arrays, signed and absolute curl, actual applied E(t)/Ec, terminal measured reward, and interior curl. A separate CSV records the map, optimizer, seed, generation, original evaluation index, physical-state path, and texture metrics. If a replay reward disagrees with the original archived reward, the analysis raises an error instead of silently accepting the state.

The five principal quantities are total curl, curl-localization participation fraction, high-curl area above a fixed initial 90th-percentile threshold (shared across methods on that map/seed), relative change of the polarization state between improving winners, and spatial polarization correlation length. Spatial variance/RMS and mean change in absolute curl are supplementary descriptors. None of these replaces the terminal-curl reward.

The texture figures are:

- **T1: Physical-state progression** — winning waveform, P_x, P_y, absolute curl and changes in absolute curl at selected optimization improvements.
- **T2: Texture metrics vs generation** — reward, localization, high-curl area, state difference, correlation length and curl difference.
- **T3: Reward vs texture descriptors** — only distinct improved states are plotted.
- **T4: PCA of polarization states** — states labeled by algorithm and reward, with connected best-so-far trajectories. Uses the reference disorder by default.

This is the progression of *outcomes across optimization*, not polarization evolution within a single waveform.

Outputs are under FerroSim_REPRESENTATION_v4/experiments/SOURCE_AND_CONFIG_HASH/ with maps, histories, tables, figures and a texture directory containing state NPZ files, a generation index, descriptor tables and T1–T4 images.

## Numerical cautions and status

- The physical evolution has open-edge couplings while the reported curl uses periodic spatial differences. Examine boundary contributions before interpreting high curl as a physically new texture.
- Finer RK4 integration does not refine the 900-sample driving waveform discretization.
- The new statistics are aligned, but different control representations and proposer families do not become physically or algorithmically equivalent merely through common plotting.
- Empirical measured Pareto fronts do not certify global optima.
- **The files have been uploaded and structurally checked; complete clean-environment Anaconda execution has not been independently validated in this update.** Use the small initialization and one-generation tests before running N=20 at full budget.

## Editable shared source

The following repository files are embedded verbatim into all five notebooks:

- [FerroSim_v4_common.py](FerroSim_v4_common.py): common history loaders, measured plotting and selected-control exports
- [FerroSim_v4_advanced_analysis.py](FerroSim_v4_advanced_analysis.py): harmonized Legendre Try statistics
- [FerroSim_v4_texture.py](FerroSim_v4_texture.py): physical-state archive and texture figures
