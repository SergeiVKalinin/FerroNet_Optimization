# Grammar — FerroSim optimization

[FerroSim_Grammar_v4.ipynb](FerroSim_Grammar_v4.ipynb) optimizes experimental programs made from 2–12 operations, including ramps, holds, reversals, pulses, ringdowns, and bursts. It compares a simple versus full operation vocabulary using GA–DKL and quality-diversity DKL, seeking higher measured terminal polarization curl.

Using the common **Legendre Try** analysis structure, it reports reward convergence, per-disorder champion waveforms, empirical reward–complexity Pareto fronts, method statistics, and optional transfer and numerical validation. The added texture section archives the physical polarization configurations reached by successive reward-improving programs.

Edit the first configuration cell and run the v4 notebook in a fresh Anaconda/Jupyter kernel. Older versions are in [iterations/](iterations/), while the [v4 guide](../FerroSim_v4_README.md) explains the common outputs. Full end-to-end execution has not yet been independently verified.
