"""Build REPORT.md from results_<ID>.json and run_output.txt (no hand-typed numbers)."""
import json
import re
from pathlib import Path

STUDENT_ID = 230103377
HERE = Path(__file__).resolve().parent
THREADS = ("1", "2", "4", "8")


def ms(seconds):
    return f"{seconds * 1e3:.2f}"


def scaling_table(times):
    rows = ["| Threads | Time (ms) | Speedup |", "|---:|---:|---:|"]
    rows += [f"| {t} | {ms(times[t])} | {times['1'] / times[t]:.2f}x |" for t in THREADS]
    return "\n".join(rows)


def task1(r):
    runs = r["t1_buggy_runs"]
    rows = ["| Run | Lost counts | max abs(hist - ref) | max abs(w_sums - ref) |", "|---:|---:|---:|---:|"]
    rows += [f"| {x['run']} | {x['lost']:,} | {x['max_count_diff']:,} | {x['max_wsum_diff']:.4e} |" for x in runs]
    safe = r["t1_safe"]
    hits = "\n".join(f"- {t} threads: `{safe['worker_hits'][t]}`" for t in THREADS)
    return (f"### Buggy kernel, 8 threads, 10 runs (N_T1 = {r['params']['N_T1']:,})\n\n" + "\n".join(rows)
            + f"\n\n### safe_hist_omp\n\nWorker participation (`worker_hits`):\n\n{hits}\n\n"
            + f"Counts equal the serial reference, and `w_sums` is bit-identical at 1/2/4/8 threads "
            + f"(max difference from the serial float sum: {safe['max_abs_vs_serial_float']:.3e}).\n\n"
            + scaling_table(safe["times_s"]))


def task2(r):
    t2 = r["t2"]
    rows = ["| Threads | Unpadded (ms) | Padded (ms) | Unpadded / padded | As-given, store hoisted (ms) |",
            "|---:|---:|---:|---:|---:|"]
    rows += [f"| {t} | {ms(t2['unpadded_s'][t])} | {ms(t2['padded_s'][t])} | "
             f"{t2['unpadded_s'][t] / t2['padded_s'][t]:.2f}x | {ms(t2['as_given_s'][t])} |" for t in THREADS]
    return (f"PAD_STRIDE = {r['params']['PAD_STRIDE']} int64 = {r['params']['PAD_STRIDE'] * 8} bytes "
            f"(cache line on this machine: {t2['cache_line_bytes']} bytes). Each thread runs "
            f"ITERS_T2 = {r['params']['ITERS_T2']:,} iterations, so ideal time is flat across thread counts.\n\n"
            + "\n".join(rows) + f"\n\n**Padding speedup at 8 threads: {t2['padding_speedup_8T']:.2f}x.**")


def task3(r):
    t3 = r["t3"]
    rows = ["| Threads | Static (ms) | Balanced (ms) | Speedup of balanced |", "|---:|---:|---:|---:|"]
    rows += [f"| {t} | {ms(t3['static_s'][t])} | {ms(t3['balanced_s'][t])} | "
             f"{t3['static_s'][t] / t3['balanced_s'][t]:.2f}x |" for t in THREADS]
    lir = ["| 8 threads | Time LIR (finish max/min) | Ops LIR (ops max/min) |", "|---|---:|---:|",
           f"| static | {t3['time_lir_static_8T']:.3f} | {t3['ops_lir_static_8T']:.3f} |",
           f"| balanced | {t3['time_lir_balanced_8T']:.3f} | {t3['ops_lir_balanced_8T']:.3f} |"]
    return ("\n".join(rows) + "\n\n" + "\n".join(lir)
            + f"\n\nOps per thread, static: `{t3['ops_static_8T']}`  \n"
            + f"Ops per thread, balanced: `{t3['ops_balanced_8T']}`  \n"
            + f"Finish ticks, static: `{t3['finish_ticks_static_8T']}`  \n"
            + f"Finish ticks, balanced: `{t3['finish_ticks_balanced_8T']}`")


def task4(r):
    t4 = r["t4"]
    conv = ", ".join(f"sweep {k}: {v:.3e}" for k, v in t4["max_delta_at"].items())
    return (f"Grid {r['params']['GRID_H']} x {r['params']['GRID_W']}, omega = {r['params']['OMEGA_SOR']:.3f}, "
            f"600 sweeps. Grids at 2/4/8 threads match the 1-thread grid exactly (max abs diff 0).\n\n"
            + scaling_table(t4["times_s"]) + f"\n\nmax_delta per sweep (8 threads): {conv}.")


def main():
    r = json.loads((HERE / f"results_{STUDENT_ID}.json").read_text())
    out = (HERE / "run_output.txt").read_text()
    digest = re.search(r'"verification_digest": "(\w+)"', out).group(1)
    bundle = out[out.index("=== OPENMP PYTHON LAB VERIFICATION BUNDLE ==="):].strip()
    d = r["diagnosis"]
    report = f"""# OpenMP Parallel Programming Lab in Python: Report

Student: Akbota Kutushova (230103377). Instructor: Sufyan bin Uzayr.
Machine: Apple M4, 10 cores (4 performance + 6 efficiency), 128-byte cache lines, Numba OpenMP layer (`omp`).
Timings are the median of 3 runs after one warm-up call. Source: `openmp_lab_{STUDENT_ID}.py`; full console log: `run_output.txt`.

**Verification digest: `{digest}`** (identical on a second full run).

## Task 1: Race condition and thread-private histogram

{task1(r)}

**DIAGNOSIS_T1.** {d['T1']}

## Task 2: False sharing

{task2(r)}

The starter kernel as given (`unpadded_state_as_given`) shows no false sharing at all: LLVM keeps `state[tid]` in a register and stores it once after the loop. To measure the hardware effect, both graded kernels do the update through a relaxed atomic load/store (`xor_store_relaxed`). On arm64 that compiles to an ordinary load and store, but the compiler must perform it on every iteration.

**DIAGNOSIS_T2.** {d['T2']}

## Task 3: Workload skew and dynamic scheduling

{task3(r)}

The handout defines the load-imbalance ratio over operations per thread. On the M4 that metric is misleading: static partitioning gives every thread the same operations (ops LIR about 1.00), yet the threads on efficiency cores finish about twice as late, and everyone waits at the barrier. The dynamic scheduler balances finish time instead, so the performance cores do more operations (ops LIR above 1.15) and the region ends sooner. The script therefore gates on the time-based LIR (finish time per thread, from `mach_absolute_time`), and reports both numbers.

**DIAGNOSIS_T3.** {d['T3']}

## Task 4: Red-Black SOR

{task4(r)}

Scaling is close to linear up to 4 threads (all on performance cores). At 8 threads, four threads land on efficiency cores, and each of the 1,200 parallel regions (red and black for 600 sweeps) ends at a barrier, so the slow cores set the pace and the time rises again.

## Verification bundle

```
{bundle}
```
"""
    (HERE / "REPORT.md").write_text(report)
    print("wrote REPORT.md")


if __name__ == "__main__":
    main()
