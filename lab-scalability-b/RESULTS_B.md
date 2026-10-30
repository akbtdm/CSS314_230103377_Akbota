# RESULTS_B.md: Granularity and IPC Contention Diagnostics (Variant B)

Student ID 230103377 | branch `variant-b-230103377` | generated 2026-10-30 09:00 UTC

## Hardware

- Apple M4, 10 execution cores (4 performance + 6 efficiency, no SMT), 128-byte cache line; macOS 15.5; CPython 3.13.5, start method `spawn`.
- `CPUs: 10 | Default Pool: 10` (Part 0 check).
- Runtimes are medians of 3 runs of the whole `run_*` function as written in the handout.

## Multi-core matrix

| Configuration | T1 (s) | T2 (s) | T4 (s) | T8 (s) | S8 | e(2) | e(4) | e(8) | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| A: Micro (k=1) | 14.207 | 9.905 | 12.567 | 13.119 | 1.08x | 0.394 | 0.846 | 0.912 | Master dispatch serialization |
| A: Coarse (k=10,000) | 5.851 | 3.014 | 1.558 | 1.133 | 5.17x | 0.030 | 0.022 | 0.078 | Amortised dispatch, healthy strong scaling |
| B: Queue Choke | 1.787 | 2.674 | 2.741 | 3.036 | 0.59x | 1.993 | 1.712 | 1.799 | IPC lock contention explosion |
| C: Buffered Batch | 0.368 | 0.218 | 0.140 | 0.142 | 2.59x | 0.184 | 0.175 | 0.298 | Amortised synchronization |

## Sign-off self-check

- **Experiment A:** with chunksize = 1 the master dispatches 1,000,000 task envelopes at about 14.2 us each; runtime is 14.21 s at P = 1 and 13.12 s at P = 8 (S8 = 1.08x), i.e. the workers starve behind the single dispatching thread. With chunksize = 10,000 the same work takes 5.85 s at P = 1 and 1.13 s at P = 8 (S8 = 5.17x); micro-tasks are 12x slower at P = 8.
- **Experiment B:** T8 = 3.04 s vs T1 = 1.79 s (T8/T1 = 1.70, negative speedup: PASS); runtime rises monotonically with P: True; e(p) rises monotonically: False (e(2) = 1.99, e(8) = 1.80). One `put` per item costs about 4.5 us even with a single producer.
- **Experiment C:** 10,000,000 items in 0.142 s at P = 8 versus 400,000 items in 1.79 s for the unbuffered queue: throughput 315x higher (70,422,122 vs 223,868 items/s), with only 1,000 IPC crossings instead of one per item. S8 = 2.59x (below the 5.0x target), e(8) = 0.298 (above the 0.05 target); on the 4 performance cores S4 = 2.62x, e(4) = 0.175.

## Notes

- This CPU has 4 performance and 6 efficiency cores. With 8 equal static shares the slowest worker lands on an efficiency core (about half the speed), so S8 is capped near 4-5x for *any* well-behaved workload here; the P = 4 numbers show the scaling on homogeneous cores.
- macOS starts processes with `spawn`, which adds tens of milliseconds per experiment; it is inside the timed region because the handout's `run_*` functions time everything from `start()` to `join()`.
- Karp-Flatt e(p) > 1 is impossible for a pure serial fraction, so the Experiment B values are proof of contention by construction, independent of the exact T1.

## Reproduce

```
python3 exp_a_granularity.py
python3 exp_b_queue.py
python3 exp_c_buffered.py
python3 write_results_b.py
```
