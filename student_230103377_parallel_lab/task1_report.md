# Task 1 Report: The Agglomeration Threshold

Apple M4, 10 physical cores (4 performance + 6 efficiency, no SMT), macOS 15.5, CPython 3.13.5. Student ID 230103377: seed = 230103377, N = 8,770,000, pool of 10 processes, median of 3 trials.

| C | tasks dispatched | median time (s) | vs. optimum |
|---|---|---|---|
| 1 | 8,770,000 | 116.162 | 365.5x |
| 10 | 877,000 | 12.090 | 38.0x |
| 100 | 87,700 | 1.636 | 5.1x |
| 1,000 | 8,770 | 0.404 | 1.3x |
| 10,000 | 877 | 0.318 | 1.0x |
| 100,000 | 88 | 0.331 | 1.0x |

**Q1.1.** Global minimum at **C = 10,000**: 0.318 s (877 chunks, about 88 per worker).

**Q1.2.** Ratio = Time(C=1) / Time(C=10,000) = 116.162 / 0.318 = **365.5x**. Fine granularity was 365 times slower.

**Q1.3.** At C = 1 `pool.map` creates 8,770,000 tasks. Each is pickled by the parent's single task-handler thread, written to a pipe, unpickled and executed by a worker, and its result makes the same trip back. That round trip cost 13.2 us per element here, while `sin(sqrt(x))` takes under 0.2 us, so over 95% of the time went to IPC, system calls and context switches, and the parent's one dispatcher thread, not the 10 workers, was the bottleneck. Foster's agglomeration amortises the fixed per-message cost over C elements: at C = 10,000 the cost drops to 0.036 us per element while 877 chunks still balance the load. Coarser is not better: C = 100,000 leaves only 88 chunks for 10 workers (8.8 each), so some workers get an extra chunk and the others idle for the straggler.
