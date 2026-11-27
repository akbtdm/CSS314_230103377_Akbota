"""Supplement to task4_roofline.py: measured DRAM bandwidth with 1-8 processes.

task4_roofline.py times pool start-up, np.ones() and fresh page allocation
together. Here each process first allocates and touches its own arrays
(2 x 16,000,000 float64 = 256 MB, far larger than the 16 MB L2), waits at a barrier, and only then times
REPS passes of b = a * 2.0 (reads 8 B + writes 8 B per element, in place,
no new pages). Aggregate GB/s = total bytes moved / slowest worker's time.
"""
import multiprocessing as mp
import time

import numpy as np

N = 16_000_000
REPS = 10
BYTES_PER_PASS = 16 * N  # 8 B read + 8 B written per element


def worker(barrier, queue):
    a = np.ones(N, dtype=np.float64)
    b = np.empty_like(a)
    np.multiply(a, 2.0, out=b)  # touch every page before timing
    barrier.wait()
    t0 = time.perf_counter()
    for _ in range(REPS):
        np.multiply(a, 2.0, out=b)
    queue.put(time.perf_counter() - t0)


def measure(workers):
    barrier, queue = mp.Barrier(workers), mp.Queue()
    procs = [mp.Process(target=worker, args=(barrier, queue)) for _ in range(workers)]
    for p in procs: p.start()
    times = [queue.get() for _ in procs]
    for p in procs: p.join()
    wall = max(times)
    return wall, workers * REPS * BYTES_PER_PASS / wall / 1e9


if __name__ == "__main__":
    print(f"Streaming b = a*2.0 over {N:,} float64 per worker, {REPS} passes, {BYTES_PER_PASS / 1e6:.0f} MB moved per pass")
    base = None
    for w in [1, 2, 4, 8]:
        wall, gbs = measure(w)
        base = base or gbs
        print(f"Workers: {w} | kernel time (slowest worker): {wall:.4f}s | aggregate bandwidth: {gbs:6.1f} GB/s | vs 1 worker: {gbs / base:.2f}x")
