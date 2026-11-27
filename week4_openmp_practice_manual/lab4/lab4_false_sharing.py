"""Lab 4: false sharing, cache-line padding and thread-local accumulators.

Each of P OpenMP threads (Numba prange over ranks) increments its own int64
counter ITERATIONS times. Variants:
  unpadded   counters[tid]          stride 1  (8 B apart; 16 per 128 B line)
  pad64      counters[tid * 8]      stride 8  (64 B apart, the manual's padding;
                                               on Apple M4 two counters still
                                               share one 128 B line)
  pad128     counters[tid * 16]     stride 16 (128 B apart = M4 line size)
  local_mem  private 256 B buffer per thread (same load/add/store per step),
             one store to counters[tid] at the end
  local_reg  plain scalar accumulator (register), one store at the end; LLVM is
             free to optimise the loop, which is the point of the variant

The increment is a relaxed atomic LOAD + ADD + relaxed atomic STORE (racy_add):
the same machine code as `volatile long` in C (plain ldr/add/str on ARM64), so
LLVM cannot keep the counter in a register for the whole loop. Without this,
LLVM turns the loop into one store and false sharing disappears.
"""
import csv
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
os.environ.setdefault("NUMBA_NUM_THREADS", "16")
from omp_numba import banner, racy_add  # noqa: E402

import numba  # noqa: E402
import numpy as np  # noqa: E402
from numba import njit, prange  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
ITERATIONS = int(os.environ.get("LAB4_ITERS", 100_000_000))
P_VALUES = [1, 2, 4, 8, 16]
TRIALS = 3
VARIANTS = ("unpadded", "pad64", "pad128", "local_mem", "local_reg")
LINE = 128      # hw.cachelinesize on this Apple M4


@njit(parallel=True)
def strided_test(counters, num_threads, iters, stride):
    for tid in prange(num_threads):
        idx = tid * stride
        for _ in range(iters):
            racy_add(counters, idx, 1)


@njit(parallel=True)
def local_mem_test(counters, num_threads, iters):
    for tid in prange(num_threads):
        # thread-private 256 B buffer; slot 16 is >= 128 B from any neighbour
        local = np.zeros(32, dtype=np.int64)
        for _ in range(iters):
            racy_add(local, 16, 1)             # same memory op, private line
        counters[tid] = local[16]              # single shared write


@njit(parallel=True)
def local_reg_test(counters, num_threads, iters):
    for tid in prange(num_threads):
        s = 0                                  # plain scalar: lives in a register
        for _ in range(iters):
            s += 1                             # (LLVM may fold the whole loop)
        counters[tid] = s                      # single shared write


def aligned_zeros(n):
    """int64 zeros whose first element starts on a 128-byte boundary."""
    raw = np.zeros(n + LINE // 8, dtype=np.int64)
    off = (-raw.ctypes.data % LINE) // 8
    out = raw[off:off + n]
    assert out.ctypes.data % LINE == 0
    return out


def run(variant, p, iters):
    stride = {"unpadded": 1, "pad64": 8, "pad128": 16}.get(variant, 1)
    counters = aligned_zeros(p * stride)
    numba.set_num_threads(p)
    t0 = time.perf_counter()
    if variant == "local_mem":
        local_mem_test(counters, p, iters)
    elif variant == "local_reg":
        local_reg_test(counters, p, iters)
    else:
        strided_test(counters, p, iters, stride)
    t = time.perf_counter() - t0
    assert counters.sum() == p * iters, (variant, p, counters)
    return t


def main():
    banner(f"Lab 4: false sharing, {ITERATIONS:,} increments per thread, "
           f"{TRIALS} trials, counters 128-byte aligned")
    for v in VARIANTS:
        run(v, 2, 1000)
    rows = []
    for p in P_VALUES:
        for v in VARIANTS:
            ts = [run(v, p, ITERATIONS) for _ in range(TRIALS)]
            m = statistics.mean(ts)
            ns = m / ITERATIONS * 1e9
            rows.append(dict(variant=v, P=p, mean_s=f"{m:.4f}",
                             trials=";".join(f"{x:.4f}" for x in ts),
                             ns_per_increment_per_thread=f"{ns:.3f}"))
            print(f"P={p:<2} {v:<9}: mean {m:8.4f} s | trials "
                  + " ".join(f"{x:.4f}" for x in ts)
                  + f" | {ns:6.2f} ns per increment", flush=True)
    path = os.path.join(RESULTS, "lab4_false_sharing.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
