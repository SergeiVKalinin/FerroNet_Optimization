# Segments — FerroSim optimization

[FerroSim_Segments_v4.ipynb](FerroSim_Segments_v4.ipynb) discovers electric-field histories using variable-length, piecewise-linear segments, evolving their durations, knot positions, and amplitudes. It compares GA–DKL, quality-diversity DKL, and sequential/denoising-inspired DKL search heuristics; the latter are not trained GFlowNets or diffusion models.

The notebook uses the same **Legendre Try** statistics and reporting sequence: measured reward convergence, best and simple trajectories, per-disorder reward–complexity Pareto fronts, optimizer comparisons, and optional transfer and validation. Texture analysis saves the resulting polarization states as best-so-far reward improves and compares their curl localization and spatial organization.

Change the first configuration cell and run the v4 notebook in a fresh Anaconda/Jupyter kernel. Previous notebooks are in [iterations/](iterations/), with setup and output details in the [v4 guide](../FerroSim_v4_README.md). Full end-to-end execution remains unverified.
