# Task 3 Report: The Shared Resource Bottleneck

Apple M4, M = 10 logical = 10 physical cores (4 performance + 6 efficiency), 16 GB LPDDR5X, CPython 3.13.5. Medians of 3 trials; each worker does identical work, so speedup = T_1 x workers / T_workers (ideal = workers).

| workload | workers | T_1 (s) | T_workers (s) | speedup (ideal = workers) | efficiency |
|---|---|---|---|---|---|
| Compute_Bound | 4 | 0.308 | 0.325 | 3.79x | 95% |
| Compute_Bound | 10 | 0.308 | 0.600 | 5.14x | 51% |
| Memory_Bound | 4 | 0.593 | 0.637 | 3.72x | 93% |
| Memory_Bound | 10 | 0.593 | 1.101 | 5.38x | 54% |
| Memory_Bound_StreamCopy | 4 | 0.157 | 0.486 | 1.29x | 32% |
| Memory_Bound_StreamCopy | 10 | 0.157 | 1.168 | 1.34x | 13% |

**Q3.1.** On all 10 processors **Speedup_A = 5.14x** (compute) versus **Speedup_B = 5.38x** for the Python streaming multiply and **1.34x** for the memcpy STREAM copy; on the 4 performance cores 3.79x, 3.72x and 1.29x. Arithmetic scales with cores, moving bytes does not.

**Q3.2.** The Python version of B scaled like A (5.38x vs 5.14x) because CPython spends about 30 ns of interpreter work per element and moves only 0.5 GB/s per worker, a fraction of the bus: it is interpreter-bound, not bus-bound. The STREAM row removes the interpreter: one worker already moves 68 GB/s, 4 workers 88 GB/s in total, 10 workers 92 GB/s, the practical limit of the 120 GB/s LPDDR5X bus. Cores have private L1/L2 but share one memory controller and DRAM bus, so beyond its bandwidth extra cores only queue and stall on misses. Workload A keeps 20,000 floats in L1 and does 500 sin/cos per value, so the bus stays idle. The 6 efficiency cores add a smaller cap: even compute reaches 51% efficiency at M = 10.

**Q3.3.** Wherever arithmetic intensity is low: deep-learning inference outside large matmuls (embedding lookups, KV-cache attention, element-wise layers; hence fusion, quantisation, FlashAttention); analytical databases, whose scans and hash joins are bytes-per-second bound (columnar storage, compression); physics/CFD stencils and sparse matrices (tiling, structure-of-arrays); graph analytics, sorting and large reductions, where memory access, not core count, sets the ceiling.
