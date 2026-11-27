"""Lab 5: task-based parallel merge sort with a sequential cutoff K.

CPython cannot run `#pragma omp task` natively, so this file builds the task
model explicitly:
  * a pool of P worker threads pulls tasks from a shared LIFO task queue
    (depth-first, like a work-stealing deque's owner end);
  * a "split" task either sorts its range sequentially (size <= K) or creates
    two child tasks; when the second child finishes, the parent's "merge" task
    becomes ready (this is `taskwait` written as a continuation, so no worker
    ever blocks waiting);
  * the leaf sort and the merge are Numba @njit(nogil=True) kernels, so the
    workers really run in parallel while the Python scheduling code holds the
    GIL. Per-task Python overhead is what K controls.
Every task records its own duration, which gives the measured Work (sum of all
task times) and Span (longest root-to-leaf path) of the actual task DAG.

Sub-commands: sweep (Task 5.1-5.2), workspan (Task 5.3)
"""
import csv
import math
import os
import queue
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from omp_numba import banner  # noqa: E402

import numpy as np  # noqa: E402
from numba import njit  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
N = 5_000_000
CUTOFFS = [1, 10, 100, 1_000, 10_000, 50_000, 100_000]
WORKERS = 10
DEADLINE_S = 60.0
TRIALS = 3


@njit(nogil=True, cache=False)
def leaf_sort(a, lo, hi):
    a[lo:hi].sort()


@njit(nogil=True, cache=False)
def merge(a, buf, lo, mid, hi):
    i, j, k = lo, mid, lo
    while i < mid and j < hi:
        if a[i] <= a[j]:
            buf[k] = a[i]
            i += 1
        else:
            buf[k] = a[j]
            j += 1
        k += 1
    while i < mid:
        buf[k] = a[i]
        i += 1
        k += 1
    while j < hi:
        buf[k] = a[j]
        j += 1
        k += 1
    a[lo:hi] = buf[lo:hi]


@njit(nogil=True)
def serial_merge_sort(a, buf, lo, hi, cutoff):
    if hi - lo <= cutoff:
        leaf_sort(a, lo, hi)
        return
    mid = (lo + hi) // 2
    serial_merge_sort(a, buf, lo, mid, cutoff)
    serial_merge_sort(a, buf, mid, hi, cutoff)
    merge(a, buf, lo, mid, hi)


def n_nodes_bound(n, cutoff):
    depth = max(0, math.ceil(math.log2(max(1, n / max(1, cutoff))))) + 1
    return (1 << (depth + 1))


class TaskSort:
    """P workers + shared LIFO task queue; node ids use heap numbering."""

    def __init__(self, a, cutoff, workers, record=False, deadline=DEADLINE_S):
        self.a, self.buf = a, np.empty_like(a)
        self.cutoff, self.workers = cutoff, workers
        size = n_nodes_bound(len(a), cutoff)
        self.lo = np.zeros(size, dtype=np.int64)
        self.hi = np.zeros(size, dtype=np.int64)
        self.pending = np.zeros(size, dtype=np.int8)
        self.dur = np.zeros(size) if record else None
        self.kind = np.zeros(size, dtype=np.int8) if record else None  # 1 leaf, 2 merge
        self.q = queue.LifoQueue()
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.tasks = 0
        self.deadline = deadline
        self.aborted = False

    def _complete(self, node):
        if node == 0:
            self.done.set()
            return
        parent = (node - 1) // 2
        with self.lock:
            self.pending[parent] -= 1
            ready = self.pending[parent] == 0
        if ready:
            self.q.put((1, parent))                     # merge task now ready

    def _worker(self):
        rec = self.dur is not None
        while not self.done.is_set():
            item = self.q.get()
            if item is None:
                return
            kind, node = item
            self.tasks += 1                             # GIL-protected int
            lo, hi = self.lo[node], self.hi[node]
            if kind == 0 and hi - lo <= self.cutoff:    # leaf: sequential sort
                t0 = time.perf_counter() if rec else 0.0
                leaf_sort(self.a, lo, hi)
                if rec:
                    self.dur[node] = time.perf_counter() - t0
                    self.kind[node] = 1
                self._complete(node)
            elif kind == 0:                             # split: spawn two tasks
                mid = (lo + hi) // 2
                l, r = 2 * node + 1, 2 * node + 2
                self.lo[l], self.hi[l], self.lo[r], self.hi[r] = lo, mid, mid, hi
                self.pending[node] = 2
                self.q.put((0, r))
                self.q.put((0, l))
            else:                                       # merge after "taskwait"
                mid = (lo + hi) // 2
                t0 = time.perf_counter() if rec else 0.0
                merge(self.a, self.buf, lo, mid, hi)
                if rec:
                    self.dur[node] = time.perf_counter() - t0
                    self.kind[node] = 2
                self._complete(node)

    def run(self):
        self.lo[0], self.hi[0] = 0, len(self.a)
        threads = [threading.Thread(target=self._worker, daemon=True)
                   for _ in range(self.workers)]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        self.q.put((0, 0))
        finished = self.done.wait(self.deadline)
        elapsed = time.perf_counter() - t0
        if not finished:
            self.aborted = True
            self.done.set()
        for _ in threads:
            self.q.put(None)
        for t in threads:
            t.join(timeout=5)
        return elapsed, finished

    def span_and_work(self):
        """Longest path and total task time over the recorded DAG."""
        span = np.zeros_like(self.dur)
        for node in range(len(self.dur) - 1, -1, -1):
            k = self.kind[node]
            if k == 1:
                span[node] = self.dur[node]
            elif k == 2:
                span[node] = self.dur[node] + max(span[2 * node + 1], span[2 * node + 2])
        return float(self.dur.sum()), float(span[0])


def make_data(seed=230103377):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 10_000_000, size=N, dtype=np.int64)


def warm():
    a = np.random.default_rng(1).integers(0, 100, 2000)
    serial_merge_sort(a.copy(), np.empty_like(a), 0, len(a), 16)
    TaskSort(a.copy(), 100, 2).run()


def cmd_sweep():
    banner(f"Lab 5 Tasks 5.1-5.2: task merge sort, N = {N:,}, {WORKERS} worker threads, "
           f"{TRIALS} trials (1 for K <= 10), deadline {DEADLINE_S:.0f} s")
    warm()
    data = make_data()
    ref = np.sort(data)
    t0 = time.perf_counter()
    np.sort(data)
    t_np = time.perf_counter() - t0
    a = data.copy()
    t0 = time.perf_counter()
    serial_merge_sort(a, np.empty_like(a), 0, N, 1)
    t_ser = time.perf_counter() - t0
    assert np.array_equal(a, ref)
    print(f"reference np.sort: {t_np:.4f} s | Numba serial merge sort (K=1, no tasks): "
          f"{t_ser:.4f} s")
    rows = []
    for k in CUTOFFS:
        trials = 1 if k <= 10 else TRIALS
        times, tasks, status = [], 0, "ok"
        for _ in range(trials):
            a = data.copy()
            ts = TaskSort(a, k, WORKERS)
            el, fin = ts.run()
            tasks = ts.tasks
            if not fin:
                status = f"DNF (> {DEADLINE_S:.0f} s)"
                times.append(el)
                break
            assert np.all(a[:-1] <= a[1:]) and np.array_equal(a, ref)   # Task 5.1
            times.append(el)
        m = statistics.mean(times)
        rows.append(dict(K=k, log10K=f"{math.log10(k):.2f}", mean_s=f"{m:.4f}",
                         trials=";".join(f"{x:.4f}" for x in times), tasks_executed=tasks,
                         status=status))
        print(f"K={k:>7}: {status:<14} mean {m:8.4f} s | trials "
              + " ".join(f"{x:.4f}" for x in times) + f" | tasks executed {tasks:,}",
              flush=True)
    with open(os.path.join(RESULTS, "lab5_cutoff_sweep.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(RESULTS, "lab5_baselines.txt"), "w") as fh:
        fh.write(f"np_sort_s={t_np:.4f}\nnumba_serial_mergesort_K1_s={t_ser:.4f}\n")
    print("wrote results/lab5_cutoff_sweep.csv, results/lab5_baselines.txt; "
          "all finished runs verified sorted (assert)")


def cmd_workspan(k=50_000):
    banner(f"Lab 5 Task 5.3: measured Work/Span of the task DAG, N = {N:,}, K = {k:,}")
    warm()
    data = make_data()
    rows = []
    for workers in (1, WORKERS):
        a = data.copy()
        ts = TaskSort(a, k, workers, record=True)
        el, fin = ts.run()
        assert fin and np.all(a[:-1] <= a[1:])
        work, span = ts.span_and_work()
        rows.append(dict(workers=workers, K=k, wall_s=f"{el:.4f}",
                         work_T1_s=f"{work:.4f}", span_Tinf_s=f"{span:.4f}",
                         parallelism=f"{work/span:.2f}",
                         leaf_tasks=int((ts.kind == 1).sum()),
                         merge_tasks=int((ts.kind == 2).sum())))
        print(f"workers={workers:<2}: wall {el:.4f} s | Work (sum of task times) {work:.4f} s"
              f" | Span (longest path) {span:.4f} s | T1/Tinf = {work/span:.2f} | "
              f"leaves {(ts.kind == 1).sum()} merges {(ts.kind == 2).sum()}", flush=True)
        if workers == WORKERS:
            depth_rows = []
            for node in range(len(ts.kind)):
                if ts.kind[node] == 2:
                    d = int(math.floor(math.log2(node + 1)))
                    depth_rows.append((d, ts.hi[node] - ts.lo[node], ts.dur[node]))
            levels = sorted(set(r[0] for r in depth_rows))
            print("merge level | merges | size per merge | mean merge time (ms)")
            for d in levels:
                sel = [r for r in depth_rows if r[0] == d]
                print(f"{d:>11} | {len(sel):>6} | {sel[0][1]:>14,} | "
                      f"{statistics.mean(r[2] for r in sel) * 1e3:10.3f}")
    with open(os.path.join(RESULTS, "lab5_workspan.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote results/lab5_workspan.csv")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    if cmd == "sweep":
        cmd_sweep()
    elif cmd == "workspan":
        cmd_workspan(int(sys.argv[2]) if len(sys.argv) > 2 else 50_000)
    else:
        sys.exit(f"unknown sub-command {cmd}")
