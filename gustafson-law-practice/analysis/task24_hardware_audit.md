# Task 24 — Hardware Audit & Empirical vs. Theoretical Discrepancy

> **Status:** hardware section complete. The results and discrepancy sections are filled
> in from `benchmarks/results.csv` / `benchmarks/fit_results.json` after running the
> benchmark on the host machine (see README → "Reproduce").

## 1. Host architecture (Apple M4, MacBook Pro)

| Property | Value | Source |
|---|---|---|
| SoC | Apple M4 (ARMv8.x / ARM64) | `sysctl machdep.cpu.brand_string` → `benchmarks/hardware.json` |
| Cores | 10 total: 4 Performance (P) + 6 Efficiency (E) | `hw.perflevel0/1.physicalcpu` |
| SMT / Hyper-Threading | **None**: 10 logical = 10 physical | `hw.logicalcpu == hw.physicalcpu` |
| L1 data cache | 128 KB per P-core, 64 KB per E-core | `hw.l1dcachesize` |
| L2 cache | 16 MB shared by the P-cluster; 4 MB shared by the E-cluster | `hw.perflevel*.l2cachesize` |
| Last level | System Level Cache (SLC) shared by CPU/GPU/NPU; no classic per-core L3 | Apple platform docs |
| Memory | Unified LPDDR5X, ~120 GB/s, shared by CPU, GPU and NPU | Apple M4 specs |
| Memory channels / NUMA | Single SoC memory controller complex, **UMA**: no NUMA boundaries | — |
| OS scheduling | macOS assigns threads to P- or E-cores dynamically (QoS); no user-level pinning | — |
| Python start method | `spawn` (macOS default since Python 3.8) | `hardware.json` |

## 2. Measured results

*(Fill from `python3 benchmarks/fit_gustafson.py` output.)*

| P | T mean (s) | T std (s) | S mean | S std | E |
|---|---|---|---|---|---|
| | | | | | |

* $\alpha_{observed}$ = … ± … (SE), $R^2$ = …
* Plot: `plots/scaling_comparison.png` / `.pdf`

## 3. Expected deviation mechanisms on this machine

1. **P-core / E-core heterogeneity (the dominant effect expected).** The benchmark gives every process the same $K$ points. For $P \le 4$, macOS can place every worker on a P-core. For $P > 4$, some workers must run on E-cores, which have lower clocks and fewer FP/SIMD pipelines. Under weak scaling $T(P) = \max_i T_i$, so the slowest (E-core) worker sets the time. Expected: a visible efficiency drop between $P = 4$ and $P = 6$, appearing as an increase of the fitted $\alpha$. Gustafson's model has no term for heterogeneous processors.
2. **Shared L2 and memory bandwidth.** P-cores share one 16 MB L2; E-cores share 4 MB. Each worker streams ~8 MB temporaries per 1M-point chunk, so the cluster L2 is contended and the unified memory bus is shared. The kernel is compute-heavy (sqrt + cos), so this should be secondary.
3. **Thermal and power management.** Sustained all-core load raises package temperature; a MacBook Pro M4 with a fan throttles little, but clock frequency still drops slightly when all 10 cores are busy. Runs are interleaved across $P$ to spread this effect, and the standard deviation across 5 runs captures it.
4. **Python IPC overhead.** Dispatch via pickle and pipes plus result gathering forms the serial part $\alpha$; spawn cost is excluded from the timed region (recorded separately in `pool_startup_s`).
5. **OS noise.** Background processes (Spotlight, browser) steal cycles, mostly visible at $P = 10$ when no core is free.
6. **NUMA:** not applicable (UMA SoC), so no NUMA-boundary step is expected, unlike on dual-socket x86 servers.

## 4. Discrepancy analysis

*(To be written from the measured data: compare measured points against the fitted
Gustafson line and Amdahl curve, locate the P where deviation starts, and attribute it
to the mechanisms above.)*
