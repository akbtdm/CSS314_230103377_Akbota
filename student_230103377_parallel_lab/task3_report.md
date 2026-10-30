# Task 3 Report: The Shared Resource Bottleneck

Apple M4, M = 10 logical = 10 physical cores (4 performance + 6 efficiency), 128 B cache line, 16 GB LPDDR5X; macOS 15.5, CPython 3.13.5. Medians of 3 trials; every worker does identical work, so speedup = T_1 x workers / T_workers (ideal = workers).

| workload | workers | T_1 (s) | T_workers (s) | speedup (ideal = workers) | efficiency |
|---|---|---|---|---|---|
| Compute_Bound | 4 | 0.308 | 0.325 | 3.79x | 95% |
| Compute_Bound | 10 | 0.308 | 0.600 | 5.14x | 51% |
| Memory_Bound | 4 | 0.593 | 0.637 | 3.72x | 93% |
| Memory_Bound | 10 | 0.593 | 1.101 | 5.38x | 54% |
| Memory_Bound_StreamCopy | 4 | 0.157 | 0.486 | 1.29x | 32% |
| Memory_Bound_StreamCopy | 10 | 0.157 | 1.168 | 1.34x | 13% |

**Q3.1.** On all 10 processors **Speedup_A = 5.14x** (compute) versus **Speedup_B = 5.38x** for the Python streaming multiply and **1.34x** for the memcpy STREAM copy. On the 4 performance cores: A = 3.79x, B = 3.72x, STREAM = 1.29x. Arithmetic scales with cores; moving bytes does not.

**Q3.2.** The Python version of B scaled like A (5.38x vs 5.14x) because CPython spends about 30 ns of interpreter work per element and moves only 0.5 GB/s per worker, a fraction of the bus: it is interpreter-bound. The STREAM row removes the interpreter: one worker already moves 68 GB/s, 4 workers reach 88 GB/s in total and 10 workers only 92 GB/s, the practical limit of this chip's 120 GB/s LPDDR5X bus. Each core has private L1/L2, but all ten share one memory controller and one DRAM bus; once aggregate demand exceeds its bandwidth, extra cores only queue for the same bus and stall on cache misses with nothing to compute. Workload A touches 20,000 floats that live in L1 and does 500 sin/cos iterations per value, so the bus stays idle. The 6 efficiency cores add a second, smaller cap: even compute reaches only 51% efficiency at M = 10.

**Q3.3.** Wherever arithmetic intensity is low: deep-learning inference outside the big matmuls (embedding lookups, long KV-cache attention, element-wise and normalisation layers, hence operator fusion, quantisation and FlashAttention); analytical databases, whose scans, hash joins and aggregations are bytes-per-second bound (columnar storage, compression, cache-conscious hash tables); physics and CFD stencil or sparse-matrix kernels (tiling, structure-of-arrays, mixed precision); and graph analytics, sorting and large reductions, where memory access, not core count, sets the ceiling.
