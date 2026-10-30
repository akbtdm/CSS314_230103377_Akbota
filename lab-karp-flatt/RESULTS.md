# RESULTS.md: Empirical Scalability Diagnostics with the Karp-Flatt Metric

Student ID 230103377 | branch `practice-230103377` | generated 2026-10-30 06:25 UTC

## Hardware profile

- CPU: Apple M4, 10 execution cores (4 performance + 6 efficiency, no SMT), 128-byte cache line
- OS: macOS 15.5 (Darwin 24.5.0)
- Python: CPython 3.13.5, multiprocessing default start method `spawn`
- Every runtime below is a median over repeated runs (5 for A, 3 for B and C); T_P is wall-clock time of the whole `run_*` function exactly as written in the handout (pool/process start-up included).

## Empirical runtimes and Karp-Flatt metric

| Experiment | Mode | T1 (s) | T2 (s) | T4 (s) | e(2) | e(4) | S4 | E4 | Architectural diagnosis |
|---|---|---|---|---|---|---|---|---|---|
| A: Serial Tail | 0.40 s sleep + 16 x heavy_math(2M) | 2.332 | 1.413 | 0.965 | 0.212 | 0.218 | 2.42x | 60% | Constant e: fixed Amdahl serialization (sleep + pool start-up) |
| B: Lock Contention | 2M locked increments on one `Value` | 3.249 | 9.511 | 8.643 | 4.854 | 3.213 | 0.38x | 9% | Exploding e (>1 is impossible for pure serial code): mutex and cache-line thrashing |
| C: Privatized State | 20M increments, default `spawn` pool | 0.311 | 0.175 | 0.110 | 0.122 | 0.136 | 2.84x | 71% | Collapsed e: linear scalability restored up to the fixed pool start-up cost |
| C: Privatized State | 20M increments, `fork` pool | 0.272 | 0.157 | 0.103 | 0.159 | 0.172 | 2.64x | 66% | Same pattern with cheaper process start-up |
| C: Privatized State | 200M increments, `spawn` pool | 2.775 | 1.423 | 0.781 | 0.025 | 0.042 | 3.56x | 89% | Start-up amortised: e < 0.05, near-linear speedup |

## Rubric self-check

- **Experiment A:** e(4)/e(2) = 1.029 against the [0.80, 1.25] window: PASS. e(2) = 0.212 and e(4) = 0.218 are flat compared with experiment B, but above the handout's 0.10-0.14 because this CPU finishes one heavy_math(2M) chunk in about 0.121 s, so the fixed 0.40 s pause is already 17.2% of T1 instead of ~12%. The rest of e comes from pool start-up (a fixed cost, which lowers e as P grows) and from the M4 running every chunk ~7% slower once two or more cores are busy (lower all-core clock); that second effect is multiplicative rather than fixed, and the two together move e(2) and e(4) a few hundredths apart from run to run.
- **Experiment B:** T4 = 8.643 s > T1 = 3.249 s (T4/T1 = 2.66, speedup 0.38x: negative) (PASS). e(2) = 4.854 and e(4) = 3.213 are both far beyond 1.50 (PASS); a value above 1.0 cannot be produced by any serial fraction, so the diagnosis is contention by construction. The handout's ordering e(4) >> e(2) is not reproduced on this machine: on macOS a `multiprocessing.Lock` is a POSIX semaphore and every contended acquisition goes through a kernel sleep/wake hand-off of roughly 4.8 us, so the critical section is already fully serialised with two workers (T2 = 9.511 s, T4 = 8.643 s). With T_P flat, the (p - 1) denominator of the metric makes e(4) < e(2) even though the per-increment cost tripled from 1.62 us at P = 1.
- **Experiment C:** T4 = 0.110 s < T1 = 0.311 s with E4 = 71% at the handout's 20M increments (PASS). The residual e of 0.136 is the ~0.05 s needed to spawn a pool on macOS, which is a fixed serial cost competing with only 0.3 s of compute. Scaling the work to 200M increments amortises it: E4 = 89%, e(2) = 0.025, e(4) = 0.042, i.e. the e < 0.05 regime the handout describes.

## Diagnosis summary

The three experiments do the same amount of arithmetic per core but show three different Karp-Flatt signatures. A fixed serial path (A) gives a flat e(p) equal to the serial fraction plus constant overheads. A shared lock on every iteration (B) gives e(p) values far above 1, which no serial fraction can explain; the hardware signature is a 3x higher cost per increment and runtimes that get *worse* with more cores. Privatising the state and reducing once (C) collapses e(p) back to the start-up overhead, and near-linear speedup returns as soon as the work is large enough to hide that overhead. On this Apple M4 two platform details shape the numbers: process creation uses `spawn` by default (tens of milliseconds per pool) and semaphore hand-offs cost microseconds, so contention saturates with just two workers.

## Reproduce

```
python3 exp_a_serial.py
python3 exp_b_contention.py
python3 exp_c_fixed.py
python3 write_results.py
```

Raw per-run timings are in `results/exp_a_serial.json`, `results/exp_b_contention.json` and `results/exp_c_fixed.json`.
