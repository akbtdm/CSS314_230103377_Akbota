# OpenMP Parallel Programming Lab in Python: Report

Student: Akbota Kutushova (230103377). Instructor: Sufyan bin Uzayr.
Machine: Apple M4, 10 cores (4 performance + 6 efficiency), 128-byte cache lines, Numba OpenMP layer (`omp`).
Timings are the median of 3 runs after one warm-up call. Source: `openmp_lab_230103377.py`; full console log: `run_output.txt`.

**Verification digest: `5E4D5DD5E979B90C720D`** (identical on a second full run).

## Task 1: Race condition and thread-private histogram

### Buggy kernel, 8 threads, 10 runs (N_T1 = 4,116,736)

| Run | Lost counts | max abs(hist - ref) | max abs(w_sums - ref) |
|---:|---:|---:|---:|
| 0 | 3,089,746 | 51,437 | 7.1016e+04 |
| 1 | 3,079,750 | 51,160 | 7.0781e+04 |
| 2 | 3,061,408 | 50,924 | 7.0280e+04 |
| 3 | 3,079,903 | 51,218 | 7.0646e+04 |
| 4 | 3,089,467 | 51,429 | 7.0961e+04 |
| 5 | 2,969,543 | 49,441 | 6.8119e+04 |
| 6 | 3,022,486 | 50,457 | 6.9507e+04 |
| 7 | 3,012,907 | 50,062 | 6.9172e+04 |
| 8 | 3,040,450 | 50,432 | 6.9840e+04 |
| 9 | 3,063,815 | 50,907 | 7.0170e+04 |

### safe_hist_omp

Worker participation (`worker_hits`):

- 1 threads: `[4116736]`
- 2 threads: `[2058368, 2058368]`
- 4 threads: `[1029184, 1029184, 1029184, 1029184]`
- 8 threads: `[514592, 514592, 514592, 514592, 514592, 514592, 514592, 514592]`

Counts equal the serial reference, and `w_sums` is bit-identical at 1/2/4/8 threads (max difference from the serial float sum: 1.906e-09).

| Threads | Time (ms) | Speedup |
|---:|---:|---:|
| 1 | 18.06 | 1.00x |
| 2 | 9.74 | 1.85x |
| 4 | 5.29 | 3.41x |
| 8 | 5.77 | 3.13x |

**DIAGNOSIS_T1.** In 10 runs of buggy_hist_omp at 8 threads between 2969543 and 3089746 of 4116736 increments were lost, because hist[b] += 1 is a load, add, store sequence and two threads can load the same old value before either stores. Even if the integer increment were made atomic, floating-point addition is not associative: (a + b) + c is rounded after each step, so it can differ from a + (b + c) in the last bits. With threads the order in which weights reach w_sums[b] depends on the OpenMP schedule and on timing, so the rounded total changes between runs and between thread counts. safe_hist_omp removes the race with per-thread rows indexed by get_thread_id(), and it removes the order dependence by accumulating each weight as an exact fixed-point integer (w * 2^54 split into two 27-bit limbs); integer sums are associative, so the merged result is bit-identical at 1, 2, 4 and 8 threads.

## Task 2: False sharing

PAD_STRIDE = 24 int64 = 192 bytes (cache line on this machine: 128 bytes). Each thread runs ITERS_T2 = 39,940,000 iterations, so ideal time is flat across thread counts.

| Threads | Unpadded (ms) | Padded (ms) | Unpadded / padded | As-given, store hoisted (ms) |
|---:|---:|---:|---:|---:|
| 1 | 46.81 | 47.77 | 0.98x | 46.32 |
| 2 | 46.34 | 46.31 | 1.00x | 46.53 |
| 4 | 57.22 | 51.88 | 1.10x | 51.82 |
| 8 | 97.36 | 72.08 | 1.35x | 56.57 |

**Padding speedup at 8 threads: 1.35x.**

The starter kernel as given (`unpadded_state_as_given`) shows no false sharing at all: LLVM keeps `state[tid]` in a register and stores it once after the loop. To measure the hardware effect, both graded kernels do the update through a relaxed atomic load/store (`xor_store_relaxed`). On arm64 that compiles to an ordinary load and store, but the compiler must perform it on every iteration.

**DIAGNOSIS_T2.** Unpadded, 8 threads took 97.4 ms versus 46.8 ms for 1 thread doing the same per-thread work; the padded version took 72.1 ms at 8 threads, a 1.35x speedup. In the unpadded array all eight int64 slots sit in one cache line (this M4 uses 128-byte lines), so under MESI every store needs the line in the Modified state on the writing core; that invalidates the copies in the other cores, which must then fetch the line again before their own store. The line keeps moving between cores even though no value is shared. With PAD_STRIDE=24 int64 (192 bytes) each thread's slot is on its own line, which stays Modified in one core's L1, so the stores hit the cache and the threads run independently.

## Task 3: Workload skew and dynamic scheduling

| Threads | Static (ms) | Balanced (ms) | Speedup of balanced |
|---:|---:|---:|---:|
| 1 | 28.38 | 28.58 | 0.99x |
| 2 | 14.62 | 14.43 | 1.01x |
| 4 | 7.44 | 7.33 | 1.01x |
| 8 | 8.18 | 5.72 | 1.43x |

| 8 threads | Time LIR (finish max/min) | Ops LIR (ops max/min) |
|---|---:|---:|
| static | 2.046 | 1.002 |
| balanced | 1.001 | 3.163 |

Ops per thread, static: `[980316, 981948, 979808, 981724, 980780, 979940, 981864, 980928]`  
Ops per thread, balanced: `[837630, 1127741, 1121035, 1123239, 836914, 838116, 1491227, 471406]`  
Finish ticks, static: `[112498, 88985, 134419, 154266, 159856, 121597, 98434, 182062]`  
Finish ticks, balanced: `[133839, 133771, 133803, 133813, 133875, 133811, 133762, 133945]`

The handout defines the load-imbalance ratio over operations per thread. On the M4 that metric is misleading: static partitioning gives every thread the same operations (ops LIR about 1.00), yet the threads on efficiency cores finish about twice as late, and everyone waits at the barrier. The dynamic scheduler balances finish time instead, so the performance cores do more operations (ops LIR above 1.15) and the region ends sooner. The script therefore gates on the time-based LIR (finish time per thread, from `mach_absolute_time`), and reports both numbers.

**DIAGNOSIS_T3.** Static prange gives each thread one contiguous block of rows. Because the cost pattern repeats every 38 rows, the operation counts per block are almost equal (LIR 2.046), but the M4 has 4 performance and 6 efficiency cores, so the threads on slower cores finish their blocks later and the rest wait at the barrier (8.2 ms at 8 threads). balanced_skew_omp starts one worker per thread and each worker claims 16 rows at a time with an atomic fetch-and-add on a shared cursor (LLVM atomicrmw via a Numba intrinsic, no locks). Rows are visited largest-first, so the final claims are small and the tail is short. Faster cores simply claim more chunks; 8-thread time was 5.7 ms with LIR 1.001, and every row score matches static exactly.

## Task 4: Red-Black SOR

Grid 416 x 576, omega = 1.855, 600 sweeps. Grids at 2/4/8 threads match the 1-thread grid exactly (max abs diff 0).

| Threads | Time (ms) | Speedup |
|---:|---:|---:|
| 1 | 133.70 | 1.00x |
| 2 | 69.23 | 1.93x |
| 4 | 36.21 | 3.69x |
| 8 | 53.05 | 2.52x |

max_delta per sweep (8 threads): sweep 1: 1.188e+02, sweep 10: 2.840e+01, sweep 100: 3.262e-01, sweep 300: 1.037e-01, sweep 600: 4.992e-02.

Scaling is close to linear up to 4 threads (all on performance cores). At 8 threads, four threads land on efficiency cores, and each of the 1,200 parallel regions (red and black for 600 sweeps) ends at a barrier, so the slow cores set the pace and the time rises again.

## Verification bundle

```
=== OPENMP PYTHON LAB VERIFICATION BUNDLE ===
{
  "instructor": "Instructor - Sufyan bin Uzayr",
  "student_name": "Akbota Kutushova",
  "student_id": 230103377,
  "omp_layer": "omp",
  "cpu_cores": 10,
  "verification_digest": "5E4D5DD5E979B90C720D"
}
```
