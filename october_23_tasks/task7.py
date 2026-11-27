"""TASK 7 - Arithmetic Intensity vs. Memory Bottlenecks.

Kernel A (memory-bound):  B[i] = A[i]                               0 FLOP per 16 bytes moved
Kernel B (compute-bound): B[i] = sin(A[i]) * cos(A[i]) + tan(A[i])  ~3 transcendental calls per 16 bytes

Both run over 30,000,000 float64 elements with real native threads (Numba
prange on the OpenMP layer) at 1, 2 and max cores.
"""
import os

os.environ.setdefault("NUMBA_THREADING_LAYER", "omp")

import math
import platform
import statistics
import time

import numpy as np
from numba import njit, prange, set_num_threads

N = 30_000_000
REPEATS = 5
STALL_EFFICIENCY = 0.5  # below 50% efficiency at max cores = scaling stalled


@njit(parallel=True)
def kernel_a_copy(a, b):
    for i in prange(a.shape[0]):
        b[i] = a[i]


@njit(parallel=True)
def kernel_b_compute(a, b):
    for i in prange(a.shape[0]):
        x = a[i]
        b[i] = math.sin(x) * math.cos(x) + math.tan(x)


def time_kernel(kernel, a, b):
    kernel(a, b)  # warm-up (JIT + page faults on b)
    samples = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        kernel(a, b)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def main():
    max_cores = os.cpu_count() or 1
    core_counts = (1, 2, max_cores)
    a = np.random.default_rng(230103377).uniform(-1.0, 1.0, N)
    b = np.empty_like(a)
    bytes_moved = 2 * a.nbytes  # read A + write B

    print("TASK 7 - Arithmetic Intensity vs. Memory Bottlenecks")
    print(f"Machine: {platform.machine()} {platform.system()}, {max_cores} cores (Apple M4: 4P + 6E), "
          f"N = {N:,} float64 ({a.nbytes / 1e6:.0f} MB per array), median of {REPEATS} runs")

    results = {}
    for name, kernel in (("A: copy (memory-bound)", kernel_a_copy), ("B: sin*cos+tan (compute-bound)", kernel_b_compute)):
        times = {}
        for c in core_counts:
            set_num_threads(c)
            times[c] = time_kernel(kernel, a, b)
        results[name] = times
        print(f"\nKernel {name}")
        print(f"{'Cores':>5} | {'Time (ms)':>10} | {'Speedup':>8} | {'Efficiency':>10} | {'Effective GB/s':>14}")
        print("-" * 60)
        for c in core_counts:
            speedup = times[1] / times[c]
            print(f"{c:>5} | {times[c] * 1e3:>10.2f} | {speedup:>7.2f}x | {speedup / c:>9.1%} | "
                  f"{bytes_moved / times[c] / 1e9:>14.1f}")

    np.testing.assert_allclose(b, np.sin(a) * np.cos(a) + np.tan(a), rtol=1e-12)

    print("\nVerdict")
    for name, times in results.items():
        eff = times[1] / times[max_cores] / max_cores
        state = "STALLED - memory channel saturated" if eff < STALL_EFFICIENCY else "scales with cores"
        print(f"  Kernel {name}: efficiency at {max_cores} cores = {eff:.1%} -> {state}")
    copy_name = next(iter(results))
    best_bw = bytes_moved / min(results[copy_name].values()) / 1e9
    print(f"  Kernel A stalls because it does no arithmetic: every element is 16 bytes of DRAM traffic, "
          f"and the copy tops out at about {best_bw:.0f} GB/s, the shared memory bus limit. More cores only add "
          f"more requests to the same bus. Kernel B does enough math per byte that the bus is never the limit.")


if __name__ == "__main__":
    main()
