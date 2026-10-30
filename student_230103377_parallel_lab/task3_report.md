# Task 3 Report: The Shared Resource Bottleneck

Apple M4, M = 10 cores (4 performance + 6 efficiency, no SMT). Medians of 3 trials; identical work per worker, so speedup = T_1 x workers / T_workers.

| workload | workers | T_1 (s) | T_workers (s) | speedup (ideal = workers) | efficiency |
|---|---|---|---|---|---|
| Compute_Bound | 4 | 0.308 | 0.325 | 3.79x | 95% |
| Compute_Bound | 10 | 0.308 | 0.600 | 5.14x | 51% |
| Memory_Bound | 4 | 0.593 | 0.637 | 3.72x | 93% |
| Memory_Bound | 10 | 0.593 | 1.101 | 5.38x | 54% |
| Memory_Bound_StreamCopy | 4 | 0.157 | 0.486 | 1.29x | 32% |
| Memory_Bound_StreamCopy | 10 | 0.157 | 1.168 | 1.34x | 13% |

**Q3.1.** On all 10 processors **Speedup_A = 5.14x** versus **Speedup_B = 5.38x** for the Python streaming multiply and **1.34x** for the memcpy STREAM copy; on the 4 performance cores 3.79x, 3.72x and 1.29x. Arithmetic scales with cores, moving bytes does not.

**Q3.2.** Python B scaled like A (5.38x vs 5.14x): CPython spends ~30 ns of interpreter work per element and moves only 0.5 GB/s per worker, so it is interpreter-bound, not bus-bound. The STREAM row removes the interpreter: one worker moves 68 GB/s, 4 workers 88 GB/s and 10 workers 92 GB/s in total, the practical limit of the 120 GB/s LPDDR5X bus. Cores have private L1/L2 but share one memory controller and DRAM bus; beyond its bandwidth extra cores only queue and stall on misses. Workload A's 20,000 floats stay in L1, so its bus traffic is negligible. The 6 efficiency cores add a smaller cap: even compute reaches 51% efficiency at M = 10.

**Q3.3.** Wherever arithmetic intensity is low: deep-learning inference outside large matmuls (embedding lookups, KV-cache attention, element-wise layers; hence fusion, quantisation, FlashAttention); analytical databases, whose scans and joins are bytes-per-second bound (columnar storage, compression); physics/CFD stencils and sparse matrices (tiling, structure-of-arrays); graph analytics, sorting and large reductions, where memory access, not core count, sets the ceiling.
