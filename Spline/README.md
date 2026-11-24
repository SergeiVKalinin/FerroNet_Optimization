# Spline — FerroSim optimization

[FerroSim_Spline_v4.ipynb](FerroSim_Spline_v4.ipynb) represents the electric-field history using smooth, locally supported cubic B-splines with 32 independent interior control coordinates and constrained endpoints. GA–DKL, quality-diversity DKL, trained GFlowNet–DKL, and guided diffusion–DKL search for high terminal polarization curl. This is a new standalone spline benchmark, not an exact reproduction of the historical spline analysis.

Its reporting follows **Legendre Try**: measured optimization histories, per-disorder waveforms and Pareto fronts, complexity and optimizer statistics, optional transfer and numerical validation, and polarization-texture evolution as reward improves. The physical texture archives include polarization arrays, curl fields, and the corresponding winning waveforms.

Edit the v4 notebook’s first configuration cell and run it in a fresh Anaconda/Jupyter kernel. Earlier notebooks, including the historical spline descriptors, are in [iterations/](iterations/). See the [v4 guide](../FerroSim_v4_README.md). Full end-to-end execution remains unverified.
