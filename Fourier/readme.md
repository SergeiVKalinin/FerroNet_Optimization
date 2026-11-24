# Fourier — FerroSim optimization

[FerroSim_Fourier_v4.ipynb](FerroSim_Fourier_v4.ipynb) optimizes a time-dependent electric field represented by a constrained sine-Fourier expansion (34 coefficients, 32 independent controls). GA–DKL, quality-diversity DKL, trained GFlowNet–DKL, and guided diffusion–DKL search for waveforms that maximize the simulator-measured terminal polarization curl.

The notebook follows the **Legendre Try** analysis protocol: configurable disorder realizations and optimizer seeds, convergence curves, reward–complexity Pareto fronts, best and simple waveforms, statistical comparisons, and optional transfer and numerical validation. It also archives polarization configurations at successive reward improvements and analyzes their spatial textures.

Open the v4 notebook in a fresh Anaconda/Jupyter kernel, edit its first configuration cell, and run all cells. Older notebooks are preserved in [iterations/](iterations/); the [v4 guide](../FerroSim_v4_README.md) covers the shared workflow. Full end-to-end numerical execution has not yet been independently verified.
