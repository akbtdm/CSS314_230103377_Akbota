"""Supplement to task1_amdahl.py: cost of an empty Pool (spawn + teardown).

Same harness as run_bench() but each worker gets chunk_size = 0, so the
measured time is pure process start-up, IPC and shutdown: the serial part
that keeps Task 1 away from linear speedup. Median of 3 runs per p.
"""
import statistics
import time
import multiprocessing as mp

from task1_amdahl import cpu_intensive


def empty_pool(workers):
    start = time.perf_counter()
    with mp.Pool(processes=workers) as pool:
        pool.map(cpu_intensive, [0] * workers)
    return time.perf_counter() - start


if __name__ == "__main__":
    print("Cores | empty Pool time, median of 3 (s)")
    for c in [1, 2, 4, 8, 12, 16]:
        print(f"{c:5d} | {statistics.median(empty_pool(c) for _ in range(3)):.4f}")
