# Legendre Try — Audited FerroSim control optimization

The notebook `FerroSim_Legendre_Clean_Standalone_Audited.ipynb` is the maintained **v2 standalone full-analysis** version of the Legendre benchmark. It embeds the FerroSim physics and all four optimizers; no separate repository source files or Google Drive folders are required.

## One configuration cell

Edit the **first Python cell**:

| Setting | Default | Supported |
| --- | --- | --- |
| `N_DISORDERS` | `20` | Any integer `>= 1` (`1` is supported) |
| `INITIAL_CONTROLS` | `64` | Any integer `>= 2`; `2` is a low-data regression test |
| `SEEDS` | `[0]` | Any nonempty list of distinct nonnegative integers |
| `GENERATIONS` | `1` | Any integer `>= 0`; `0` runs physical initialization only |
| `MODE` | `'QUICK'` | `'QUICK'` or `'FULL'`: identical RK4 physical oracle, different proposer-training steps |
| `ANALYZE_EACH_DISORDER` | `True` | Generate best-waveform/Pareto figures for every map |
| `RUN_CROSS_DISORDER_TRANSFER` | `True` | Disable for a fast test; full transfer costs roughly `2*N_DISORDERS^2` RK4 evaluations |
| `RUN_HIGHER_FIDELITY_VALIDATION` | `True` | Independently reevaluate selected champions at higher RK4 fidelity |
| `VALIDATE_ALL_DISORDERS` | `False` | When True, repeat high-fidelity champion checks on every map |

The default benchmark has **20 disorder maps**, not one. To test one map with the absolute minimum initial design, set `N_DISORDERS=1`, `INITIAL_CONTROLS=2`, `GENERATIONS=1`. The initial design size is *independent* of the map count. DKL requires at least **two valid** simulator observations; if fewer survive numerical guards, the optimization reports an explicit error. The 2-control setup is *not* scientifically adequate for robust model comparisons; retain 64 controls for serious experiments.

## Four optimizers and comparable budgets

- GA–DKL
- GFlowNet–DKL
- Guided diffusion–DKL
- Quality-diversity–DKL

Each optimizer receives the same initial observations within each (disorder, seed) pair. Then every generation evaluates **32 new physical controls per method**. With `I` initial controls, `G` generations, `D` maps and `S` seeds:

- **Method-history rows:** `D*S*4*(I+32*G)`.
- **Distinct physical calls:** `D*S*(I+4*32*G)`.

For `D=20`, `S=3`, `I=64`, `G=32`, these totals are **261,120 recorded method observations** and **249,600 distinct physical calls**, because shared initialization should not be charged four times.

## Analysis and visualization inventory

The notebook saves reproducible source histories and includes:

1. Explicit map manifest and the reference 19-site defect map.
2. Numerical validity, true physical-query accounting, runtime/proposer origin.
3. Best actual reward versus evaluations and observed normalized improvement.
4. Exact evaluation index at 95% and 99% of observed reward gain.
5. Historical reward–complexity Pareto fronts on the reference and **every** map.
6. Best and simplest waveforms for each map and algorithm, with Legendre coefficients.
7. Per-generation `imshow` coefficient and waveform evolution.
8. Modal coefficient energies, reversals, total variation, peak field, and boundary-versus-interior curl.
9. Minimum complexity for 95% and 99% reward-gain retention, and 0/1/3-turn baselines.
10. Seed versus disorder variance and matched-pair algorithm contrasts (CI omitted when only one pair exists).
11. Candidate provenance / selection mechanism and DKL calibration against subsequently measured physical rewards.
12. Optional raw and normalized cross-disorder transfer heatmaps.
13. Optional higher-fidelity RK4 validation and spatial polarization/curl maps.
14. CSV, JSON, NPZ, checkpoint, PNG, and output-index exports.

With one map or seed the notebook omits meaningless multi-map error estimates. Missing optimization results are never synthesized or replaced with simulated proxy rewards.

## Validation

The revised simulator/proposer runner was physically tested for `N_DISORDERS=1`, `INITIAL_CONTROLS=2`, `GENERATIONS=1`: all four optimizers successfully ran with two valid initial observations and 32 additional RK4 queries per optimizer. The notebook was also executed end-to-end with two maps and an initialization-only budget, and subjected to a 20-map initialization test. These are software regression tests, **not** completion of the full 20-map/three-seed/32-generation study.

The original notebook under `Legendre/` has not been modified.
