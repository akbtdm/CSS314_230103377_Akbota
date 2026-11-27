"""TASK 5 - Thread Oversubscription Threshold.

10,000 small CPU-bound tasks: each hashes a 1 KB string through 500 rounds of
HMAC-SHA256 (hashlib.pbkdf2_hmac). That call runs in C with the GIL released,
so the threads really run in parallel. For comparison the same batch is also
run with a pure-Python loop of hashlib.sha256 on 1 KB (the GIL is held for
inputs under 2 KB), which shows what CPython threads do without GIL release.
"""
import hashlib
import os
import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

N_TASKS = 10_000
HASH_ROUNDS = 500
PAYLOAD = (b"230103377-" * 103)[:1024]  # 1 KB string
SALT = b"css314-task5"
REPEATS = 3
WORSE_THRESHOLD = 0.02  # >2% slower than the best time counts as worsening


def task_parallel(i):
    """1 KB string, 500 HMAC-SHA256 rounds in C, GIL released."""
    return hashlib.pbkdf2_hmac("sha256", PAYLOAD + i.to_bytes(4, "little"), SALT, HASH_ROUNDS)[0]


def task_gil_bound(i):
    """Same idea in a Python loop: sha256 of 1 KB, 500 times, GIL held."""
    digest = PAYLOAD
    for _ in range(HASH_ROUNDS):
        digest = hashlib.sha256(PAYLOAD + digest[:4]).digest()
    return digest[0] + i % 2


def run_batch(fn, pool_size):
    with ThreadPoolExecutor(max_workers=pool_size) as pool:
        t0 = time.perf_counter()
        checksum = sum(pool.map(fn, range(N_TASKS), chunksize=1))
        elapsed = time.perf_counter() - t0
    return elapsed, checksum


def measure(fn, pool_sizes, repeats):
    times, checksums = {}, set()
    for p in pool_sizes:
        samples = []
        for _ in range(repeats):
            elapsed, checksum = run_batch(fn, p)
            samples.append(elapsed)
            checksums.add(checksum)
        times[p] = statistics.median(samples)
    assert len(checksums) == 1, "results differ between pool sizes"
    return times


def statuses(pool_sizes, times):
    best = min(pool_sizes, key=lambda p: times[p])
    marks, threshold_found = {}, None
    for p in pool_sizes:
        if p == best:
            marks[p] = "BEST"
        elif pool_sizes.index(p) < pool_sizes.index(best):
            marks[p] = "scaling"
        elif times[p] > times[best] * (1 + WORSE_THRESHOLD):
            if threshold_found is None:
                threshold_found = p
                marks[p] = "<-- WORSENING STARTS (context-switch overhead)"
            else:
                marks[p] = "worse"
        else:
            marks[p] = "plateau (within 2% of best)"
    return marks, best, threshold_found


def print_table(title, pool_sizes, times):
    marks, best, threshold = statuses(pool_sizes, times)
    print(f"\n{title}")
    print(f"{'Pool Size':>9} | {'Execution Time (ms)':>19} | {'Speedup vs 1 thread':>19} | Status")
    print("-" * 95)
    for p in pool_sizes:
        print(f"{p:>9} | {times[p] * 1e3:>19.1f} | {times[1] / times[p]:>18.2f}x | {marks[p]}")
    if threshold is None:
        print(f"Fastest pool: {best}. No pool size after it was more than 2% slower.")
    else:
        print(f"Fastest pool: {best}. Execution time starts worsening at pool size {threshold} "
              f"({times[threshold] / times[best]:.2f}x the best time).")


def main():
    cores = os.cpu_count() or 1
    pool_sizes = sorted({1, max(1, cores // 2), cores, cores * 2, cores * 4, cores * 16, 256})
    print("TASK 5 - Thread Oversubscription Threshold")
    print(f"Machine: {platform.machine()} {platform.system()}, os.cpu_count() = {cores} "
          f"(Apple M4: 4 performance + 6 efficiency cores), Python {platform.python_version()}")
    print(f"Batch: {N_TASKS:,} tasks, each = 1 KB string hashed {HASH_ROUNDS} times. "
          f"Pool sizes: {pool_sizes}. Time = median of {REPEATS} runs.")

    print_table("A) pbkdf2_hmac (C code, GIL released -> real parallel threads)",
                pool_sizes, measure(task_parallel, pool_sizes, REPEATS))
    print_table("B) pure-Python sha256 loop (GIL held -> one thread runs at a time)",
                pool_sizes, measure(task_gil_bound, pool_sizes, 1))


if __name__ == "__main__":
    main()
