# OpenMP Lab Practice Manual: Shared-Memory Concurrency (Python Edition)

Akbota Kutushova, ID 230103377. Apple M4 (4P + 6E cores, 128-byte cache line), macOS 15.5, Python 3.13.5, Numba 0.61 (OpenMP threading layer).

**Formal report:** [`REPORT.docx`](REPORT.docx), built from [`REPORT.md`](REPORT.md) (Sections I to V as required by manual section 8.2). All numbers were measured on this laptop on 2026-11-27 (run 14:58:04 to 15:00:41). AI assistance was used for the code, as the instructor allows; every number comes from `logs/` and `results/`.

## Layout (manual 8.1 deliverables)

| Path | Contents |
|---|---|
| `common/omp_numba.py` | Numba setup plus the LLVM atomic, CAS and racy load/add/store intrinsics that stand in for `#pragma omp atomic / critical` and dynamic/guided schedules |
| `lab1/` | `starter_fork_join.py` (manual starter, Task 1.1), `lab1_fork_join.py` (`team`: Numba team arrival order; `sweep`: Task 1.2; `saturate`: Task 1.3) |
| `lab2/lab2_pi.py` | Task 2.1 race, Task 2.2 critical (spin lock), atomic and `threading.Lock`, Tasks 2.3-2.4 reduction scaling |
| `lab3/lab3_mandelbrot.py` | static / dynamic / guided schedulers with chunk C, 4x4 matrix, per-thread work (Tasks 3.1-3.4) |
| `lab4/lab4_false_sharing.py` | unpadded, 64 B pad, 128 B pad, thread-local buffer, thread-local register (Tasks 4.1-4.3) |
| `lab5/lab5_merge_sort.py`, `lab5/starter_process_pool.py` | task merge sort with cutoff sweep and measured Work/Span (Tasks 5.1-5.3); manual's process-pool starter |
| `lab6/lab6_pipeline.py` | bonus 3-stage pipeline with bounded queues and graceful shutdown |
| `hw_info.sh` | hardware and software report (Section I) |
| `run_all.sh` | runs everything one at a time and `tee`s output to `logs/` |
| `make_plots.py` | rebuilds every PNG in `results/` from the CSVs |
| `logs/` | raw console output of the run used in the report |
| `results/` | raw CSV timing data and plots |

## Reproduce

```sh
cd week4_openmp_practice_manual
./run_all.sh                     # about 2 min 40 s on the M4; needs numpy, numba, psutil, matplotlib
pandoc REPORT.md -o REPORT.docx  # rebuild the report document
```

Single labs: `NUMBA_NUM_THREADS=16 python3 lab3/lab3_mandelbrot.py`, `python3 lab5/lab5_merge_sort.py sweep`, and so on. `NUMBA_THREADING_LAYER=omp` is set automatically.

## Key measured results

| Lab | Result |
|---|---|
| 1 | 10 Numba OpenMP threads kept 9.95 cores busy; 10 CPython threads kept 1.65 cores busy (GIL). Reused-pool region cost 4.4 us at P = 8, 60 us at P = 16 |
| 2 | Race: Pi = 1.53 / 0.91 / 0.62 at P = 2 / 4 / 8. Critical at P = 8: 269x slower than serial. Reduction S(8) = 4.97, S(16) = 5.58 |
| 3 | Serial 0.490 s; best dynamic,1 at P = 16: 72 ms (6.8x); block static at P = 8: 232 ms (imbalance 3.26) |
| 4 | Unpadded 0.132 s vs 128 B padded 0.040 s at P = 2 (3.3x); 64 B padding is not enough on 128 B lines |
| 5 | K = 1: 37.2 s (15 M tasks); best K = 100,000: 0.074 s; measured T1/T_inf = 8.7 |
| 6 | Stage 2 is the bottleneck; 4 filter workers give 3.7x throughput at 8 and 32 blur passes |

## Deviations from the manual (and why)

* **Real threads come from Numba, not CPython threads.** CPython 3.13 has the GIL, so plain threads give no speedup and hide races (measured, Labs 1 and 2). Numba with the `omp` threading layer gives real OpenMP threads. Missing `atomic`, `critical`, dynamic and guided are built from LLVM atomics in `common/omp_numba.py`.
* **The cache line is 128 B on Apple M4**, so Lab 4 tests the manual's 64 B padding *and* 128 B padding.
* **Task 1.1:** the manual's starter forces its print order with `sleep`, so a Numba team that records arrival order was added to show real non-determinism.
* **Task 1.3:** CPU usage was sampled with `psutil` (CSV and plot) instead of an Activity Monitor screenshot, which cannot be taken here.
* **Task 4.4 (`perf stat`):** skipped. It is Linux-only, and macOS hardware counters need privileged Instruments. No cache-miss numbers are given.
* **Lab 5:** the manual's process-pool starter cannot run K = 1 to 1000 on 5 M elements (it would spawn millions of processes). It was run once as given (N = 1 M, cutoff 50,000). The sweep uses a thread task pool with Numba `nogil` kernels and a shared LIFO queue (not per-thread work-stealing deques).
* **Java edition:** not done (Python edition chosen). Q4.4 about Java `@Contended` is answered from documentation and labelled as background.
* **macOS cannot pin threads to cores**, so the P-core / E-core placement is up to the OS. This is discussed where it affects results.
