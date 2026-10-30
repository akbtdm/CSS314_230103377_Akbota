# Task 2 Report: Hunting the Hardware Ceiling

Apple M4: 10 physical cores = 10 logical processors (4 performance + 6 efficiency, no SMT), macOS 15.5, CPython 3.13.5. Student ID 230103377: N = 11,770,000, s = 0.05, K = 588,500 sorted serially (0.062 s), 11,181,500 elements mapped with chunksize 10,000; medians of 3 trials. "spec" is the required pipeline; "heavy" repeats the transform 8x per element as a diagnostic.

| p | spec total (s) | spec speedup | spec e(p) | heavy total (s) | heavy speedup | heavy eff. | Amdahl s=0.05 |
|---|---|---|---|---|---|---|---|
| 1 | 0.974 | 1.00x | - | 4.944 | 1.00x | 100% | 1.00x |
| 2 | 0.587 | 1.66x | 0.21 | 2.573 | 1.92x | 96% | 1.90x |
| 3 | 0.547 | 1.78x | 0.34 | 1.824 | 2.71x | 90% | 2.73x |
| 4 | 0.558 | 1.75x | 0.43 | 1.574 | 3.14x | 79% | 3.48x |
| 6 | 0.572 | 1.70x | 0.50 | 1.371 | 3.60x | 60% | 4.80x |
| 8 | 0.574 | 1.70x | 0.53 | 1.260 | 3.92x | 49% | 5.93x |
| 12 | 0.580 | 1.68x | 0.56 | 1.201 | 4.11x | 34% | 7.74x |
| 16 | 0.587 | 1.66x | 0.58 | 1.246 | 3.97x | 25% | 9.14x |

**Q2.1.** S_p <= 1 / (0.05 + 0.95 / p), so as p -> infinity the ceiling is **1 / s = 20x**.

**Q2.2.** The required pipeline peaked at **p = 3 (1.78x)** and degraded from p = 4 on (1.75x, 1.66x at p = 16). The heavy variant kept gaining until p = 12 (4.11x), with the knee at p = 4 (efficiency 79% -> 49% at p = 8), and fell to 3.97x at p = 16.

**Q2.3.** `machine_info.txt` reports `core_count: 10` and `thread_count: 10`: Apple Silicon has no hyperthreading. The spec pipeline halted at p = 3, far below either count, because its ceiling is not the cores but the parent process: pickling 11,181,500 floats into chunks and unpickling the results costs 0.216 s on the main process alone, and the parallel phase never drops below 0.48 s however many workers wait on it. That is a serial fraction of roughly 50%, not 5%, and the Karp-Flatt column (e rising to 0.58) confirms it. Once compute dominates (heavy variant) the hardware shows through: near-linear to p = 4 performance cores, diminishing gains from the 6 slower efficiency cores up to the 10 physical cores, and degradation at p = 16 from oversubscription. Hyperthreading, where present, would behave like the efficiency tier here: an SMT sibling shares one core's FP/ALU units, caches and branch predictors and only fills pipeline bubbles, so a compute-bound kernel such as `log(x+1)*cos(x)` gains 10-30% per physical core, never 2x.
