"""Milestone 3 / Experiment 1: race inducer matrix.

A shared counter lives in multiprocessing.Array('i', 1). N worker processes each run
exactly 1,000,000 read-modify-write increments (val = arr[0]; val += 1; arr[0] = val).

Modes
  unsync : uninhibited loop on the raw ctypes buffer (arr.get_obj()), no lock at all
  lock   : the same read-modify-write wrapped in an explicit multiprocessing.Lock
  wrapper: (optional, --wrapper) the literal arr[0] syntax on the default synchronized
           Array, which takes arr's internal lock separately for the get and for the set,
           so the increment is still NOT atomic

All workers wait on a Barrier; runtime = master time from barrier release to last join.
Spawn passes the Array/Lock/Barrier handles to the children (multiprocessing's own
reduction); the counter data itself only ever lives in shared memory.

Usage: python3 race_inducer.py [--workers 1 2 4 8 16] [--reps 3] [--lock-reps 1] [--wrapper]
"""
import argparse
import multiprocessing as mp
import statistics
import sys
import time

from lab_constants import describe

ITERATIONS = 1_000_000


def worker(arr, lock, barrier, loop_s, wid: int, mode: str) -> None:
    raw = arr.get_obj()
    barrier.wait()
    t0 = time.perf_counter()
    if mode == "unsync":
        for _ in range(ITERATIONS):
            val = raw[0]
            val += 1
            raw[0] = val
    elif mode == "lock":
        for _ in range(ITERATIONS):
            with lock:
                val = raw[0]
                val += 1
                raw[0] = val
    else:  # "wrapper": per-access internal lock, RMW still racy
        for _ in range(ITERATIONS):
            val = arr[0]
            val += 1
            arr[0] = val
    loop_s[wid] = time.perf_counter() - t0


def run(n: int, mode: str):
    arr = mp.Array("i", 1)
    lock = mp.Lock()
    barrier = mp.Barrier(n + 1)
    loop_s = mp.RawArray("d", n)
    procs = [mp.Process(target=worker, args=(arr, lock, barrier, loop_s, i, mode)) for i in range(n)]
    for p in procs:
        p.start()
    barrier.wait()
    t0 = time.perf_counter()
    for p in procs:
        p.join()
    ms = (time.perf_counter() - t0) * 1e3
    if any(p.exitcode != 0 for p in procs):
        raise RuntimeError(f"worker failed in mode={mode} n={n}")
    return arr.get_obj()[0], ms, max(loop_s) * 1e3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--reps", type=int, default=3, help="repetitions of the unsynchronized run")
    ap.add_argument("--lock-reps", type=int, default=1)
    ap.add_argument("--wrapper", action="store_true", help="also run the per-access-lock variant")
    a = ap.parse_args()
    print(describe())
    print(f"cpu_count={mp.cpu_count()} start_method={mp.get_start_method()} iterations/worker={ITERATIONS:,}")
    rows = []
    for n in a.workers:
        expected = n * ITERATIONS
        uns = []
        for r in range(a.reps):
            count, ms, loop_ms = run(n, "unsync")
            print(f"N={n:2d} unsync rep{r + 1}: count={count:,} loss={100 * (1 - count / expected):.2f}% "
                  f"runtime={ms:.1f} ms (slowest worker loop {loop_ms:.1f} ms)", flush=True)
            uns.append((count, ms))
        lck = []
        for r in range(a.lock_reps):
            count, ms, loop_ms = run(n, "lock")
            print(f"N={n:2d} lock   rep{r + 1}: count={count:,} runtime={ms:.1f} ms "
                  f"(slowest worker loop {loop_ms:.1f} ms) {'OK' if count == expected else 'WRONG'}", flush=True)
            lck.append((count, ms))
        wrp = None
        if a.wrapper:
            count, ms, _ = run(n, "wrapper")
            print(f"N={n:2d} wrapper     : count={count:,} loss={100 * (1 - count / expected):.2f}% "
                  f"runtime={ms:.1f} ms", flush=True)
            wrp = (count, ms)
        med_count = int(statistics.median(c for c, _ in uns))
        rows.append((n, expected, med_count, [c for c, _ in uns], statistics.median(m for _, m in uns),
                     statistics.median(m for _, m in lck), lck[0][0], wrp))
    print("\n| Workers (N) | Expected Total | Observed Count (Unsynchronized, median of reps) | Update Loss Rate (%) "
          "| Runtime: Unsynchronized (ms) | Runtime: multiprocessing.Lock (ms) | Lock count |")
    print("|---|---|---|---|---|---|---|")
    for n, exp, med, all_c, ums, lms, lc, _ in rows:
        print(f"| {n} | {exp:,} | {med:,} (reps: {', '.join(f'{c:,}' for c in all_c)}) | "
              f"{100 * (1 - med / exp):.2f}% | {ums:.1f} | {lms:.1f} | {lc:,} |")
    if a.wrapper:
        print("\n| N | wrapper count (arr[0] with per-access lock) | loss % | runtime ms |\n|---|---|---|---|")
        for n, exp, *_, wrp in rows:
            print(f"| {n} | {wrp[0]:,} | {100 * (1 - wrp[0] / exp):.2f}% | {wrp[1]:.1f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
