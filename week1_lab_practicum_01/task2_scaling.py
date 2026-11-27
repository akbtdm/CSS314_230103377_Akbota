"""Task 2: multi-thread scaling benchmark (compute-bound).

Workload : count primes below 20,000,000 by trial division.
Runtime  : Python 3.13 + Numba (@njit(nogil=True)) kernels run by N real OS
           threads (threading.Thread). nogil releases the GIL, so the N threads
           execute truly in parallel on the cores.
Each N in {1,2,4,8,16,32} is run 3 times; wall-clock time is logged.
Usage: python3 task2_scaling.py | tee logs/task2_output.txt
"""
import json
import os
import platform
import threading
import time

from numba import njit

LIMIT = 20_000_000
THREAD_COUNTS = (1, 2, 4, 8, 16, 32)
RUNS = 3
BLOCK = 2_000  # numbers per work block; blocks are dealt round-robin to threads
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "task2_scaling.json")


@njit(nogil=True, cache=False)
def count_primes_strided(limit, block, worker, n_workers):
    """Count primes in blocks worker, worker+n_workers, ... (cyclic distribution)."""
    total = 0
    start = worker * block
    while start < limit:
        end = min(start + block, limit)
        for n in range(max(start, 2), end):
            if n < 4:
                total += 1
                continue
            if n % 2 == 0:
                continue
            is_prime = True
            d = 3
            while d * d <= n:
                if n % d == 0:
                    is_prime = False
                    break
                d += 2
            if is_prime:
                total += 1
        start += n_workers * block
    return total


def run_once(n_threads):
    partial = [0] * n_threads

    def worker(i):
        partial[i] = count_primes_strided(LIMIT, BLOCK, i, n_threads)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0
    return elapsed, sum(partial)


def sieve_count(limit):
    import numpy as np
    s = np.ones(limit, dtype=bool)
    s[:2] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return int(s.sum())


def demo(n_threads, seconds):
    """Keep N threads busy for a while so `top`/`ps -M` can observe them."""
    count_primes_strided(10_000, BLOCK, 0, 1)
    t_end = time.time() + seconds
    while time.time() < t_end:
        run_once(n_threads)


def imbalance(n_threads):
    """Record when each thread finishes, to expose P-core vs E-core imbalance."""
    count_primes_strided(10_000, BLOCK, 0, 1)
    finish = [0.0] * n_threads

    def worker(i):
        count_primes_strided(LIMIT, BLOCK, i, n_threads)
        finish[i] = time.perf_counter()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    done = sorted(f - t0 for f in finish)
    print(f"N={n_threads:2d} per-thread finish times (s, sorted): " + " ".join(f"{d:.3f}" for d in done))
    print(f"      fastest/slowest = {done[0]:.3f}/{done[-1]:.3f} s -> threads idle-waiting "
          f"{100 * (1 - sum(done) / (n_threads * done[-1])):.1f}% of the wall time")


def main():
    print(f"Host: {platform.machine()} {platform.platform()}  os.cpu_count()={os.cpu_count()}")
    print(f"Python {platform.python_version()}  workload: primes < {LIMIT:,} (trial division)")
    expected = sieve_count(LIMIT)  # reference answer from a numpy sieve
    print(f"reference prime count (sieve): {expected:,}")
    count_primes_strided(10_000, BLOCK, 0, 1)  # JIT compile outside timing
    rows = []
    for n in THREAD_COUNTS:
        times = []
        for r in range(1, RUNS + 1):
            elapsed, primes = run_once(n)
            if primes != expected:
                raise RuntimeError(f"wrong prime count {primes} for N={n}")
            times.append(elapsed)
            print(f"N={n:2d} run {r}: {elapsed:.4f} s  (primes={primes})", flush=True)
        rows.append({"threads": n, "runs": times, "avg": sum(times) / RUNS})
    t1 = rows[0]["avg"]
    print("\nN  | Run1 (s) | Run2 (s) | Run3 (s) | Avg TN (s) | Speedup | Efficiency")
    for row in rows:
        s = t1 / row["avg"]
        row["speedup"] = s
        row["efficiency"] = s / row["threads"]
        r1, r2, r3 = row["runs"]
        print(f"{row['threads']:2d} | {r1:8.4f} | {r2:8.4f} | {r3:8.4f} | {row['avg']:10.4f} | "
              f"{s:6.2f}x | {100 * row['efficiency']:6.1f}%")
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({"limit": LIMIT, "rows": rows}, f, indent=2)
    print(f"\nsaved {OUT_JSON}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2 and sys.argv[1] == "--imbalance":
        for n in (4, 8, 10, 16):
            imbalance(n)
    elif len(sys.argv) == 4 and sys.argv[1] == "--demo":
        demo(int(sys.argv[2]), float(sys.argv[3]))
    else:
        main()
