"""Lab 1: fork-join model, team creation and thread scoping (Python edition).

Sub-commands
  team       Numba/OpenMP team: every thread takes a ticket with an atomic
             fetch-and-add, so the arrival order is recorded (Task 1.1).
  sweep      Task 1.2: create+join cost of a team for P in {1..64} with
             (a) ThreadPoolExecutor, (b) raw threading.Thread,
             (c) a Numba OpenMP parallel region; plus a fixed total workload
             split over P threads (oversubscription penalty), Numba and
             plain Python threads (GIL).
  saturate   Task 1.3: 10,000,000 square roots per thread while a sampler
             records per-core CPU utilisation with psutil (stand-in for
             Activity Monitor, which cannot be screenshotted from here).

Run:  NUMBA_NUM_THREADS=64 python3 lab1/lab1_fork_join.py sweep
"""
import csv
import math
import multiprocessing as mp
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
os.environ.setdefault("NUMBA_NUM_THREADS", "64")
from omp_numba import atomic_fetch_add_i64, banner  # noqa: E402

import numba  # noqa: E402
import numpy as np  # noqa: E402
import psutil  # noqa: E402
from numba import njit, prange  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
P_VALUES = [1, 2, 4, 8, 16, 32, 64]
SQRT_PER_THREAD = 10_000_000          # Task 1.3 workload
FIXED_TOTAL_NUMBA = 2_000_000_000     # Task 1.2 fixed total work (Numba)
FIXED_TOTAL_PY = 4_000_000            # Task 1.2 fixed total work (Python threads)


# ---------------------------------------------------------------- Numba kernels
@njit(parallel=True)
def numba_team(ticket, order_rank, order_tid, p):
    # #pragma omp parallel num_threads(p): one prange iteration per thread
    for rank in prange(p):
        slot = atomic_fetch_add_i64(ticket, 0, 1)   # arrival order
        order_rank[slot] = rank
        order_tid[slot] = numba.get_thread_id()      # omp_get_thread_num()


@njit(parallel=True)
def numba_empty_region(counter, p):
    for rank in prange(p):
        atomic_fetch_add_i64(counter, 0, 1)


@njit(parallel=True)
def numba_sqrt_split(out, total, p):
    per = total // p
    for rank in prange(p):
        s = 0.0
        base = rank * per
        for i in range(per):
            s += math.sqrt(base + i)
        out[rank] = s


# ---------------------------------------------------------------- Python helpers
def _noop(_):
    return None


def py_sqrt(n, start=0):
    s = 0.0
    for i in range(start, start + n):
        s += math.sqrt(i)
    return s


def time_executor_team(p, reps):
    t0 = time.perf_counter()
    for _ in range(reps):
        with ThreadPoolExecutor(max_workers=p) as ex:
            for f in [ex.submit(_noop, r) for r in range(p)]:
                f.result()
    return (time.perf_counter() - t0) / reps


def time_raw_threads(p, reps):
    t0 = time.perf_counter()
    for _ in range(reps):
        ts = [threading.Thread(target=_noop, args=(r,)) for r in range(p)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    return (time.perf_counter() - t0) / reps


def time_numba_region(p, reps):
    numba.set_num_threads(p)
    counter = np.zeros(1, dtype=np.int64)
    t0 = time.perf_counter()
    numba_empty_region(counter, p)
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(reps):
        numba_empty_region(counter, p)
    steady = (time.perf_counter() - t0) / reps
    assert counter[0] == p * (reps + 1)
    return first, steady


def time_python_split(p):
    per = FIXED_TOTAL_PY // p
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=p) as ex:
        list(ex.map(lambda r: py_sqrt(per, r * per), range(p)))
    return time.perf_counter() - t0


# ---------------------------------------------------------------- sub-commands
def cmd_team(p=8):
    numba.set_num_threads(p)
    ticket = np.zeros(1, dtype=np.int64)
    rank = np.full(p, -1, dtype=np.int64)
    tid = np.full(p, -1, dtype=np.int64)
    numba_team(ticket, rank, tid, p)
    print(f"Numba OpenMP team of {p}: arrival order (rank/omp thread id): "
          + " ".join(f"{r}/{t}" for r, t in zip(rank, tid)))


def cmd_sweep():
    banner("Lab 1 Task 1.2: team create+join cost and oversubscription sweep")
    numba_team(np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64),
               np.zeros(1, dtype=np.int64), 1)              # JIT warm-up
    numba_empty_region(np.zeros(1, dtype=np.int64), 1)
    numba_sqrt_split(np.zeros(1), 1000, 1)
    rows = []
    print(f"{'P':>3} | {'Executor us':>11} | {'Thread us':>9} | {'Numba 1st us':>12} | "
          f"{'Numba us':>8} | {'Numba fixed-work s':>18} | {'Py-threads fixed-work s':>23}")
    for p in P_VALUES:
        ex_t = time_executor_team(p, 20)
        th_t = time_raw_threads(p, 20)
        nb_first, nb_t = time_numba_region(p, 200)
        out = np.zeros(p)
        numba_sqrt_split(out, FIXED_TOTAL_NUMBA, p)            # warm at this P
        trials = []
        for _ in range(3):
            t0 = time.perf_counter()
            numba_sqrt_split(out, FIXED_TOTAL_NUMBA, p)
            trials.append(time.perf_counter() - t0)
        nb_work = sum(trials) / len(trials)
        py_work = time_python_split(p)
        rows.append(dict(P=p, executor_create_join_us=ex_t * 1e6,
                         thread_create_join_us=th_t * 1e6,
                         numba_region_first_us=nb_first * 1e6,
                         numba_region_steady_us=nb_t * 1e6,
                         numba_fixed_work_s=nb_work,
                         numba_fixed_work_trials=";".join(f"{x:.4f}" for x in trials),
                         py_threads_fixed_work_s=py_work))
        print(f"{p:>3} | {ex_t*1e6:>11.1f} | {th_t*1e6:>9.1f} | {nb_first*1e6:>12.1f} | "
              f"{nb_t*1e6:>8.2f} | {nb_work:>18.4f} | {py_work:>23.4f}", flush=True)
    path = os.path.join(RESULTS, "lab1_oversubscription.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Numba fixed work = {FIXED_TOTAL_NUMBA:,} sqrt in total split over P; "
          f"Python fixed work = {FIXED_TOTAL_PY:,} sqrt split over P threads (GIL).")
    print(f"wrote {path}")


class Sampler(threading.Thread):
    """Samples per-core CPU % every 0.1 s (psutil) until stopped."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self._stop_evt = threading.Event()

    def run(self):
        psutil.cpu_percent(percpu=True)
        t0 = time.perf_counter()
        while not self._stop_evt.is_set():
            vals = psutil.cpu_percent(interval=0.1, percpu=True)
            self.samples.append((time.perf_counter() - t0, vals))

    def stop(self):
        self._stop_evt.set()
        self.join()


def _mp_worker(n):
    return py_sqrt(n)


def run_sampled(label, fn):
    s = Sampler()
    s.start()
    time.sleep(0.3)
    t0 = time.perf_counter()
    fn()
    wall = time.perf_counter() - t0
    time.sleep(0.3)
    s.stop()
    busy = [vals for (_, vals) in s.samples]
    arr = np.array(busy)
    # keep only samples inside the run (skip 0.3 s idle margins: first/last 3)
    inner = arr[3:-3] if len(arr) > 8 else arr
    per_core = inner.mean(axis=0)
    total = per_core.sum()
    print(f"{label:<38} wall {wall:7.3f} s | per-core mean %: "
          + " ".join(f"{v:5.1f}" for v in per_core)
          + f" | sum {total:6.1f}% (= {total/100:.2f} cores busy)", flush=True)
    return label, wall, s.samples, per_core


def cmd_saturate(p=10):
    banner(f"Lab 1 Task 1.3: CPU saturation, {p} threads x {SQRT_PER_THREAD:,} sqrt each")
    print("psutil per-core order cpu0..cpu9; sampling interval 0.1 s")
    numba.set_num_threads(p)
    out = np.zeros(p)
    numba_sqrt_split(out, 1000 * p, p)        # warm-up JIT

    def py_threads():
        with ThreadPoolExecutor(max_workers=p) as ex:
            list(ex.map(lambda r: py_sqrt(SQRT_PER_THREAD), range(p)))

    def numba_threads():
        # one region = 10M sqrt per thread (~ms); repeat so the sampler sees it
        for _ in range(150):
            numba_sqrt_split(out, SQRT_PER_THREAD * p, p)

    def processes():
        ctx = mp.get_context("spawn")
        with ctx.Pool(p) as pool:
            pool.map(_mp_worker, [SQRT_PER_THREAD] * p)

    results = [run_sampled("A: Python threads (GIL)", py_threads),
               run_sampled("B: Numba prange OpenMP threads (x150)", numba_threads),
               run_sampled("C: multiprocessing spawn pool", processes)]
    path = os.path.join(RESULTS, "lab1_cpu_saturation.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "t_s"] + [f"cpu{i}" for i in range(os.cpu_count())])
        for label, _, samples, _ in results:
            for t, vals in samples:
                w.writerow([label, f"{t:.2f}"] + [f"{v:.1f}" for v in vals])
    summ = os.path.join(RESULTS, "lab1_cpu_saturation_summary.csv")
    with open(summ, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "wall_s"] + [f"cpu{i}_mean_pct" for i in range(os.cpu_count())]
                   + ["sum_pct"])
        for label, wall, _, per_core in results:
            w.writerow([label, f"{wall:.3f}"] + [f"{v:.1f}" for v in per_core]
                       + [f"{per_core.sum():.1f}"])
    print(f"wrote {path} and {summ}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    if cmd == "team":
        cmd_team(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    elif cmd == "sweep":
        cmd_sweep()
    elif cmd == "saturate":
        cmd_saturate()
    else:
        sys.exit(f"unknown sub-command {cmd}")
