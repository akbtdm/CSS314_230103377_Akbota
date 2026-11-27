"""Lab 2: Pi by the midpoint rule; race, critical, atomic and reduction.

Numba OpenMP threads (real parallel threads, GIL released):
  A  naive race     : shared[0] = shared[0] + term (relaxed LOAD / ADD / STORE)
  B  critical       : spin lock built from compare-and-swap around the update
  B2 atomic         : one atomic fetch-add on a double (#pragma omp atomic)
  C  reduction      : prange `total += term` (Numba private accumulators)
Plain CPython threads (GIL) for comparison:
  A-py naive race, B-py threading.Lock critical section.

Sub-commands: race | critical | scaling | all
"""
import csv
import math
import os
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
os.environ.setdefault("NUMBA_NUM_THREADS", "16")
from omp_numba import (atomic_cas_i64, atomic_fadd_f64, atomic_store_i64,  # noqa: E402
                       banner, racy_add)

import numba  # noqa: E402
import numpy as np  # noqa: E402
from numba import njit, prange  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
N_RACE = 100_000_000
N_CRIT = 1_000_000
N_SCALE = 100_000_000


def write_csv(name, rows):
    path = os.path.join(RESULTS, name)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


# ---------------------------------------------------------------- Numba kernels
@njit
def pi_serial(n):
    step = 1.0 / n
    s = 0.0
    for i in range(n):
        x = (i + 0.5) * step
        s += 4.0 / (1.0 + x * x)
    return s * step


@njit(parallel=True)
def pi_race(shared, n):
    step = 1.0 / n
    for i in prange(n):
        x = (i + 0.5) * step
        racy_add(shared, 0, 4.0 / (1.0 + x * x))      # unprotected RMW
    return shared[0] * step


@njit(parallel=True)
def pi_critical(shared, lock, n):
    step = 1.0 / n
    for i in prange(n):
        x = (i + 0.5) * step
        term = 4.0 / (1.0 + x * x)
        while atomic_cas_i64(lock, 0, 0, 1) != 0:     # acquire (spin)
            pass
        shared[0] += term                             # critical section
        atomic_store_i64(lock, 0, 0)                  # release
    return shared[0] * step


@njit(parallel=True)
def pi_atomic(shared, n):
    step = 1.0 / n
    for i in prange(n):
        x = (i + 0.5) * step
        atomic_fadd_f64(shared, 0, 4.0 / (1.0 + x * x))
    return shared[0] * step


@njit(parallel=True)
def pi_reduction(n):
    step = 1.0 / n
    total = 0.0
    for i in prange(n):
        x = (i + 0.5) * step
        total += 4.0 / (1.0 + x * x)                  # reduction(+:total)
    return total * step


# ---------------------------------------------------------------- CPython threads
def py_serial(n):
    step = 1.0 / n
    s = 0.0
    for i in range(n):
        x = (i + 0.5) * step
        s += 4.0 / (1.0 + x * x)
    return s * step


def py_threads(n, p, use_lock):
    step = 1.0 / n
    shared = [0.0]
    lock = threading.Lock()
    chunk = n // p

    def body(start, end):
        for i in range(start, end):
            x = (i + 0.5) * step
            term = 4.0 / (1.0 + x * x)
            if use_lock:
                with lock:
                    shared[0] += term
            else:
                shared[0] += term

    ts = [threading.Thread(target=body,
                           args=(t * chunk, n if t == p - 1 else (t + 1) * chunk))
          for t in range(p)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return shared[0] * step


def timed(fn, *a):
    t0 = time.perf_counter()
    r = fn(*a)
    return r, time.perf_counter() - t0


def warm():
    pi_serial(1000)
    pi_race(np.zeros(1), 1000)
    pi_critical(np.zeros(1), np.zeros(1, dtype=np.int64), 1000)
    pi_atomic(np.zeros(1), 1000)
    pi_reduction(1000)


# ---------------------------------------------------------------- tasks
def task_race():
    banner(f"Lab 2 Task 2.1: naive race, N = {N_RACE:,}, 3 trials per P")
    rows = []
    for p in (1, 2, 4, 8):
        numba.set_num_threads(p)
        for trial in range(1, 4):
            pi, t = timed(pi_race, np.zeros(1), N_RACE)
            rows.append(dict(impl="numba_race", P=p, trial=trial, N=N_RACE, pi=f"{pi:.12f}",
                             abs_error=f"{abs(pi - math.pi):.3e}", time_s=f"{t:.4f}"))
            print(f"Numba race  P={p:<2} trial {trial}: pi = {pi:.12f} | "
                  f"error = {abs(pi - math.pi):.3e} | {t:.4f} s", flush=True)
    n_py = 2_000_000
    for p in (1, 2, 4, 8):
        pi, t = timed(py_threads, n_py, p, False)
        rows.append(dict(impl="cpython_threads_race", P=p, trial=1, N=n_py, pi=f"{pi:.12f}",
                         abs_error=f"{abs(pi - math.pi):.3e}", time_s=f"{t:.4f}"))
        print(f"CPython threads race P={p:<2} (N={n_py:,}): pi = {pi:.12f} | "
              f"error = {abs(pi - math.pi):.3e} | {t:.4f} s", flush=True)
    ref = py_serial(n_py)
    print(f"CPython serial reference N={n_py:,}: pi = {ref:.12f} | "
          f"error = {abs(ref - math.pi):.3e}")
    write_csv("lab2_race.csv", rows)


def task_critical():
    banner(f"Lab 2 Task 2.2: critical-section overhead, N = {N_CRIT:,}, mean of 5 trials")
    rows = []

    def mean_time(fn, *a, reps=5):
        ts, val = [], None
        for _ in range(reps):
            val, t = timed(fn, *a)
            ts.append(t)
        return val, statistics.mean(ts), ts

    pi, base, _ = mean_time(pi_serial, N_CRIT)
    rows.append(dict(impl="numba_serial", P=1, pi=f"{pi:.12f}", time_s=f"{base:.6f}",
                     overhead_pct="0.0"))
    print(f"Numba serial baseline       : pi = {pi:.12f} | {base*1e3:9.3f} ms")
    for name, fn, extra in (("numba_critical_spinlock", pi_critical, True),
                            ("numba_atomic", pi_atomic, False)):
        for p in (1, 2, 4, 8):
            numba.set_num_threads(p)
            if extra:
                pi, t, _ = mean_time(lambda: pi_critical(np.zeros(1),
                                                         np.zeros(1, dtype=np.int64), N_CRIT))
            else:
                pi, t, _ = mean_time(lambda: pi_atomic(np.zeros(1), N_CRIT))
            ov = (t - base) / base * 100
            rows.append(dict(impl=name, P=p, pi=f"{pi:.12f}", time_s=f"{t:.6f}",
                             overhead_pct=f"{ov:.1f}"))
            print(f"{name:<28} P={p}: pi = {pi:.12f} | {t*1e3:9.3f} ms | "
                  f"overhead vs serial = {ov:9.1f} %", flush=True)
    pi, pybase, _ = mean_time(py_serial, N_CRIT, reps=3)
    rows.append(dict(impl="cpython_serial", P=1, pi=f"{pi:.12f}", time_s=f"{pybase:.6f}",
                     overhead_pct="0.0"))
    print(f"CPython serial baseline     : pi = {pi:.12f} | {pybase*1e3:9.3f} ms")
    for p in (1, 2, 4, 8):
        pi, t, _ = mean_time(py_threads, N_CRIT, p, True, reps=3)
        ov = (t - pybase) / pybase * 100
        rows.append(dict(impl="cpython_threading_lock", P=p, pi=f"{pi:.12f}",
                         time_s=f"{t:.6f}", overhead_pct=f"{ov:.1f}"))
        print(f"cpython_threading_lock       P={p}: pi = {pi:.12f} | {t*1e3:9.3f} ms | "
              f"overhead vs serial = {ov:9.1f} %", flush=True)
    write_csv("lab2_critical.csv", rows)


def task_scaling():
    banner(f"Lab 2 Tasks 2.3-2.4: reduction strong scaling, N = {N_SCALE:,}, 5 trials")
    _, ts = timed(pi_serial, N_SCALE)
    pi_s, ts = timed(pi_serial, N_SCALE)
    print(f"Numba serial (no prange): pi = {pi_s:.12f} | {ts:.4f} s "
          f"| error = {abs(pi_s - math.pi):.2e}")
    rows = []
    t1 = None
    for p in (1, 2, 4, 8, 16):
        numba.set_num_threads(p)
        pi_reduction(N_SCALE)                     # warm-up at this P
        trials = []
        for _ in range(5):
            pi, t = timed(pi_reduction, N_SCALE)
            trials.append(t)
        avg = statistics.mean(trials)
        sd = statistics.stdev(trials)
        if p == 1:
            t1 = avg
        s = t1 / avg
        rows.append(dict(P=p, mean_s=f"{avg:.5f}", stdev_s=f"{sd:.5f}",
                         trials=";".join(f"{x:.5f}" for x in trials),
                         speedup=f"{s:.3f}", efficiency=f"{s/p:.3f}", pi=f"{pi:.12f}",
                         abs_error=f"{abs(pi - math.pi):.2e}"))
        print(f"P={p:<2}: T = {avg:.5f} s (sd {sd:.5f}) | S = {s:5.2f} | E = {s/p:5.2f} | "
              f"pi = {pi:.12f} | trials " + " ".join(f"{x:.4f}" for x in trials), flush=True)
    write_csv("lab2_scaling.csv", rows)
    with open(os.path.join(RESULTS, "lab2_serial_baseline.txt"), "w") as fh:
        fh.write(f"numba_serial_s={ts:.5f}\npi={pi_s:.12f}\n")


if __name__ == "__main__":
    warm()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("race", "all"):
        task_race()
    if cmd in ("critical", "all"):
        task_critical()
    if cmd in ("scaling", "all"):
        task_scaling()
