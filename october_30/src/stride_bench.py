"""Milestone 1, Task 1.1: stride latency cliff benchmark.

Traverses a 2^24-element float64 array (~134 MB, larger than any cache) with
strides 1..128, accumulating `total += data[i]`, and reports the wall-clock
time per element touched. Stride-1 reads consume every double in a fetched
cache line; once the stride (in bytes) reaches the cache-line size, every
element read costs a separate cache-line fetch from DRAM.

Usage:
    python3 src/stride_bench.py
Writes:
    results/stride_results.tsv
"""
import argparse
import statistics
import time

import numpy as np

from common import cache_line_bytes, load_salted_params, results_path

STRIDES = (1, 2, 4, 8, 16, 32, 64, 128)
WARMUP_ITERATIONS = 5
TIMED_ITERATIONS = 5
BYTES_PER_ELEMENT = 8

try:
    from numba import njit
except ImportError:  # pragma: no cover - fallback when Numba is missing
    njit = None


def _strided_sum_loop(data, stride):
    # Eight independent accumulators keep the loop memory-bound instead of
    # bound by the latency of a single floating-point add chain.
    n = data.shape[0]
    s0 = s1 = s2 = s3 = s4 = s5 = s6 = s7 = 0.0
    block = 8 * stride
    i = 0
    while i + block <= n:
        s0 += data[i]
        s1 += data[i + stride]
        s2 += data[i + 2 * stride]
        s3 += data[i + 3 * stride]
        s4 += data[i + 4 * stride]
        s5 += data[i + 5 * stride]
        s6 += data[i + 6 * stride]
        s7 += data[i + 7 * stride]
        i += block
    while i < n:
        s0 += data[i]
        i += stride
    return s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7


if njit is not None:
    strided_sum = njit(nogil=True)(_strided_sum_loop)
    BACKEND = "numba-njit scalar loop (8 accumulators)"
else:
    def strided_sum(data, stride):
        return float(data[::stride].sum())

    BACKEND = "numpy strided-view reduction"


def elements_touched(n: int, stride: int) -> int:
    return (n + stride - 1) // stride


def time_stride(data, stride):
    for _ in range(WARMUP_ITERATIONS):
        strided_sum(data, stride)
    samples = []
    total = 0.0
    for _ in range(TIMED_ITERATIONS):
        t0 = time.perf_counter_ns()
        total = strided_sum(data, stride)
        samples.append(time.perf_counter_ns() - t0)
    return samples, total


def main():
    parser = argparse.ArgumentParser(description="Measure read latency per element across memory strides.")
    parser.add_argument("--elements", type=int, default=None, help="Override buffer size (default: salted buffer_elements)")
    args = parser.parse_args()

    params = load_salted_params()
    n = args.elements or int(params["buffer_elements"])
    line_bytes = cache_line_bytes()
    elements_per_line = line_bytes // BYTES_PER_ELEMENT

    data = np.ones(n, dtype=np.float64)
    print(f"Buffer: {n} float64 = {n * BYTES_PER_ELEMENT / 2**20:.1f} MiB | cache line {line_bytes} B "
          f"= {elements_per_line} doubles | backend: {BACKEND}")
    print(f"{'stride':>6} {'step_B':>7} {'touched':>10} {'median_ms':>10} {'ns/elem':>8} {'GB/s':>7}  note")

    rows = []
    for stride in STRIDES:
        samples, total = time_stride(data, stride)
        touched = elements_touched(n, stride)
        if total != float(touched):
            raise RuntimeError(f"stride {stride}: reduction returned {total}, expected {touched}")
        median_ns = statistics.median(samples)
        min_ns = min(samples)
        ns_per_element = median_ns / touched
        useful_gbps = touched * BYTES_PER_ELEMENT / median_ns  # bytes/ns == GB/s
        step_bytes = stride * BYTES_PER_ELEMENT
        if step_bytes < line_bytes:
            lines_touched = n * BYTES_PER_ELEMENT // line_bytes
            note = f"{elements_per_line // stride} reads per line"
        else:
            lines_touched = touched
            note = "1 read per line"
        rows.append((stride, step_bytes, touched, lines_touched, median_ns, min_ns, ns_per_element, useful_gbps))
        print(f"{stride:>6} {step_bytes:>7} {touched:>10} {median_ns / 1e6:>10.3f} {ns_per_element:>8.3f} {useful_gbps:>7.2f}  {note}")

    base = rows[0][6]
    cliff = rows[-1][6] / base if base else float("nan")
    print(f"Latency per element rises {cliff:.1f}x from stride 1 to stride {STRIDES[-1]}.")

    out = results_path("stride_results.tsv")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(f"# buffer_elements={n}\tbytes_per_element={BYTES_PER_ELEMENT}\tcache_line_bytes={line_bytes}\t"
                 f"warmup={WARMUP_ITERATIONS}\ttimed={TIMED_ITERATIONS}\tbackend={BACKEND}\n")
        fh.write("stride\tstep_bytes\telements_touched\tcache_lines_touched\tmedian_time_ns\tmin_time_ns\t"
                 "ns_per_element\tuseful_bandwidth_gbps\n")
        for stride, step_bytes, touched, lines, median_ns, min_ns, ns_el, gbps in rows:
            fh.write(f"{stride}\t{step_bytes}\t{touched}\t{lines}\t{int(median_ns)}\t{int(min_ns)}\t{ns_el:.4f}\t{gbps:.3f}\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
