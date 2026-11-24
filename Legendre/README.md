# Legendre — FerroSim optimization

[FerroSim_Legendre_v4.ipynb](FerroSim_Legendre_v4.ipynb) uses a 36-term Legendre polynomial expansion with four endpoint constraints, leaving 32 independent electric-field control variables. GA–DKL, quality-diversity DKL, trained GFlowNet–DKL, and guided diffusion–DKL optimize the same measured terminal polarization-curl reward.

Following the latest **Legendre Try** notebook, it reports convergence, exact target-attainment evaluations, per-disorder waveforms and reward–complexity Pareto fronts, complexity retention, method attribution, uncertainty diagnostics, and optional transfer and higher-fidelity validation. New texture analysis saves and compares the polarization states obtained as the best reward improves.

Use the first configuration cell and run the v4 notebook in a fresh Anaconda/Jupyter kernel. All previous versions are in [iterations/](iterations/), and the [v4 guide](../FerroSim_v4_README.md) describes the common outputs. Full end-to-end numerical execution has not yet been independently verified.
