# Gustafson's Law & Weak Scaling — Practice Repository

Parallel & Distributed Computing Architecture · Instructor: Sufyan bin Uzayr · Ungraded practice

An executive technical guide to scaled speedup: theory (Tasks 01–05, 11–15), worked
calculations (06–10, 16–17), algorithmic case studies (18–20) and an empirical
weak-scaling benchmark on an Apple M4 laptop (21–24).

## Table of contents
1. [Core rules](#1-core-rules)
2. [When to design for weak scaling](#2-when-to-design-for-weak-scaling)
3. [Key results](#3-key-results)
4. [Reproduce](#4-reproduce)
5. [Artifact index](#5-artifact-index)

## 1. Core rules

| Quantity | Formula | Notes |
|---|---|---|
| Amdahl (fixed size) | $S_A = 1/(s + (1-s)/P)$ | $S_A \to 1/s$, $E_A \to 0$ |
| Gustafson (fixed time) | $S_G = P - \alpha(P-1)$ | $E_G \to 1-\alpha$ |
| Frame conversion | $s = \alpha/(\alpha + (1-\alpha)P)$ | same run, same speedup (Task 03) |
| With communication | $S_C = (P - \alpha(P-1))/(1 + C(P))$ | $C = c\log P$, $c\sqrt P$, … (Task 12) |
| Sun–Ni | $S = \dfrac{f + (1-f)G(P)}{f + (1-f)G(P)/P}$ | $G=1$: Amdahl, $G=P$: Gustafson (Task 11) |
| Isoefficiency | $W = K\,T_o(W,P)$ | Gustafson-compliant ⇒ $\Theta(P)$ (Task 14) |
| Max tolerable serial | $\alpha_{max} = (P - S_{req})/(P-1)$ | Task 07 |

**Five rules for engineers**
1. Always state the frame and problem size of a serial fraction.
2. Weak scaling holds only if per-rank work, per-rank communication and absolute serial time are all constant.
3. Scale the **work** with $P$, not the dimension: an $O(N^3)$ code grows $N$ only as $P^{1/3}$.
4. A growing Karp–Flatt $e$ or a growing $\alpha_{eff}(P)$ means overhead that grows with $P$, so Gustafson's line is only an upper bound.
5. Reductions and collectives must be trees ($O(\log P)$); any $O(P)$ step at rank 0 brings back Amdahl's ceiling.

## 2. When to design for weak scaling

**Good fit:** explicit PDE/stencil solvers, Monte Carlo, ensemble simulations, parameter sweeps, data-parallel training with fixed per-device batch size, workloads whose users want *higher resolution in the same time*.

**Poor fit:** latency-critical fixed-size jobs (use strong scaling), iterative elliptic solvers without multigrid, all-to-all algorithms (sort, FFT transposes), irregular graph analytics, direct $O(N^2)$ N-body, small workloads where process start-up dominates.

## 3. Key results

| Task | Result |
|---|---|
| 04 | $\lim E_G = 1-\alpha$; $\lim E_A = 0$ |
| 05 | nCUBE 1024: fixed-size 502–637×, scaled 1016–1021×; Gustafson with α = 0.4% predicts 1020× |
| 06 | α = 0.04: E falls only from 96.25% (P=16) to 96.0002% (P=16384) |
| 07 | P = 8192, S ≥ 7500 ⇒ α_max = 692/8191 = 0.08448 |
| 08 | s = 0.08, P = 128: Amdahl 11.47×, Gustafson 117.84× (10.27×, +927%) |
| 09 | α = 0.08: S(512) = 471.12, S(2048) = 1884.24 |
| 10 | 32 GiB/node, 8 B/point: N_max = 6501 (64 nodes), 13003 (512 nodes) |
| 16 | GEMM: N ∝ P^{1/3} at constant time |
| 17 | Stencil: comm/comp ratio constant under weak scaling, ∝ P^{1/3} under strong |
| 23–24 | Measured α on Apple M4: see `benchmarks/fit_results.json` and `analysis/task24_hardware_audit.md` |

## 4. Reproduce

```bash
python3 -m pip install numpy matplotlib

# Task 21: theoretical sweep -> benchmarks/simulation_results.csv + plots/simulator_overlay.png
python3 benchmarks/simulator.py
python3 benchmarks/simulator.py --alpha 0.04 --p 16 64 256 1024 4096 16384 --table

# Task 22-23: run on the real machine (about 2-5 minutes), then fit
python3 benchmarks/weak_scale_bench/bench.py          # -> benchmarks/results.csv, hardware.json
python3 benchmarks/fit_gustafson.py                   # -> summary.csv, fit_results.json

# Task 24: plots
python3 benchmarks/plot_scaling.py                    # -> plots/scaling_comparison.png/.pdf
```

## 5. Artifact index

| Task | Artifact |
|---|---|
| 01 | [theory/task01_scaling_models.md](theory/task01_scaling_models.md) |
| 02 | [theory/task02_derivation.md](theory/task02_derivation.md) |
| 03 | [theory/task03_serial_invariance.md](theory/task03_serial_invariance.md) |
| 04 | [theory/task04_asymptotic_limits.md](theory/task04_asymptotic_limits.md) |
| 05 | [theory/task05_historical_ncube.md](theory/task05_historical_ncube.md) |
| 06 | [calculations/task06_parametric_table.md](calculations/task06_parametric_table.md) |
| 07 | [calculations/task07_tolerable_serial.md](calculations/task07_tolerable_serial.md) |
| 08 | [calculations/task08_cross_divergence.md](calculations/task08_cross_divergence.md) |
| 09 | [calculations/task09_multicomponent.md](calculations/task09_multicomponent.md) |
| 10 | [calculations/task10_memory_growth.md](calculations/task10_memory_growth.md) |
| 11 | [theory/task11_sun_ni_comparison.md](theory/task11_sun_ni_comparison.md) |
| 12 | [theory/task12_communication_penalty.md](theory/task12_communication_penalty.md) |
| 13 | [theory/task13_failure_cases.md](theory/task13_failure_cases.md) |
| 14 | [theory/task14_isoefficiency.md](theory/task14_isoefficiency.md) |
| 15 | [theory/task15_energy_scaling.md](theory/task15_energy_scaling.md) |
| 16 | [calculations/task16_gemm_scaling.md](calculations/task16_gemm_scaling.md) |
| 17 | [calculations/task17_stencil_scaling.md](calculations/task17_stencil_scaling.md) |
| 18 | [analysis/task18_nbody_scaling.md](analysis/task18_nbody_scaling.md) |
| 19 | [analysis/task19_monte_carlo.md](analysis/task19_monte_carlo.md) |
| 20 | [analysis/task20_graph_processing.md](analysis/task20_graph_processing.md) |
| 21 | [benchmarks/simulator.py](benchmarks/simulator.py), `benchmarks/simulation_results.csv`, `plots/simulator_overlay.png` |
| 22 | [benchmarks/weak_scale_bench/](benchmarks/weak_scale_bench/) (`bench.py`, `README.md`) |
| 23 | `benchmarks/results.csv`, [benchmarks/fit_gustafson.py](benchmarks/fit_gustafson.py), `benchmarks/summary.csv`, `benchmarks/fit_results.json` |
| 24 | [benchmarks/plot_scaling.py](benchmarks/plot_scaling.py), `plots/scaling_comparison.png/.pdf`, [analysis/task24_hardware_audit.md](analysis/task24_hardware_audit.md) |
| 25 | this README |
