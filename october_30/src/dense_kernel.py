"""Milestone 1, Task 1.2: loop interchange on a dense matrix-multiply kernel.

Algorithm A (naive i-j-k): the innermost loop walks B down a column, so every
step jumps N doubles ahead in memory (stride-N access).
Algorithm B (interchanged i-k-j): A[i, k] is hoisted into a scalar and the
innermost loop updates row C[i, :] from row B[k, :] with stride-1 access.

Both kernels accumulate C[i, j] over k in the same order, so their results
must agree bit-for-bit; both are also checked against NumPy's BLAS matmul.

Usage:
    python3 src/dense_kernel.py
Writes:
    results/loop_interchange.json
"""
import argparse
import hashlib
import json
import statistics
import time

import numpy as np

from common import load_host_profile, load_salted_params, results_path

RUNS = 3
PURE_PYTHON_SIZE = 128
REL_TOLERANCE = 1e-12

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


def _matmul_ijk(A, B, C):
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]


def _matmul_ikj(A, B, C):
    n = A.shape[0]
    for i in range(n):
        for k in range(n):
            a = A[i, k]
            for j in range(n):
                C[i, j] += a * B[k, j]


if njit is not None:
    matmul_ijk = njit(_matmul_ijk)
    matmul_ikj = njit(_matmul_ikj)
    BACKEND = "numba-njit compiled loops"
else:
    matmul_ijk, matmul_ikj = _matmul_ijk, _matmul_ikj
    BACKEND = "pure Python loops over NumPy arrays"


def pure_python_ijk(A, B, n):
    C = [0.0] * (n * n)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i * n + j] += A[i * n + k] * B[k * n + j]
    return C


def pure_python_ikj(A, B, n):
    C = [0.0] * (n * n)
    for i in range(n):
        for k in range(n):
            a = A[i * n + k]
            for j in range(n):
                C[i * n + j] += a * B[k * n + j]
    return C


def timed_runs(fn, *args, runs=RUNS):
    """Call fn(*args) `runs` times; return (list of seconds, result of the last call)."""
    times, result = [], None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn(*args)
        times.append(time.perf_counter() - t0)
    return times, result


def run_kernel(kernel, A, B):
    C = np.zeros_like(A)
    kernel(A, B, C)
    return C


def salted_digest(student_id: str, matrix: np.ndarray) -> str:
    rounded = np.round(matrix, 6)
    h = hashlib.sha256()
    h.update(student_id.encode("utf-8"))
    h.update(rounded.tobytes())
    return h.hexdigest()


def benchmark_size(A_full, B_full, m, student_id):
    A = np.ascontiguousarray(A_full[:m, :m])
    B = np.ascontiguousarray(B_full[:m, :m])
    flops = 2.0 * m ** 3

    naive_times, C_naive = timed_runs(run_kernel, matmul_ijk, A, B)
    inter_times, C_inter = timed_runs(run_kernel, matmul_ikj, A, B)
    reference = A @ B

    naive_med = statistics.median(naive_times)
    inter_med = statistics.median(inter_times)
    abs_diff = float(np.max(np.abs(C_inter - reference)))
    rel_diff = float(np.max(np.abs(C_inter - reference) / np.maximum(np.abs(reference), 1e-300)))
    result = {
        "M": m,
        "matrix_bytes": int(m * m * 8),
        "naive_ijk": {"times_s": naive_times, "median_s": naive_med, "gflops": flops / naive_med / 1e9},
        "interchanged_ikj": {"times_s": inter_times, "median_s": inter_med, "gflops": flops / inter_med / 1e9},
        "speedup": naive_med / inter_med,
        "correctness": {
            "naive_vs_interchanged_bitwise_equal": bool(np.array_equal(C_naive, C_inter)),
            "naive_vs_interchanged_max_abs_diff": float(np.max(np.abs(C_naive - C_inter))),
            "interchanged_vs_numpy_max_abs_diff": abs_diff,
            "interchanged_vs_numpy_max_rel_diff": rel_diff,
            "within_tolerance": rel_diff <= REL_TOLERANCE,
            "rel_tolerance": REL_TOLERANCE,
        },
        "salted_sha256_rounded_6dp": salted_digest(student_id, C_inter),
    }
    print(f"M={m:>5}: naive i-j-k {naive_med:8.4f} s | i-k-j {inter_med:8.4f} s | speedup {result['speedup']:6.2f}x | "
          f"bitwise={result['correctness']['naive_vs_interchanged_bitwise_equal']} rel_err={rel_diff:.2e}")
    return result


def benchmark_pure_python(A_full, B_full, size):
    n = size
    A = A_full[:n, :n].ravel().tolist()
    B = B_full[:n, :n].ravel().tolist()
    naive_times, C_naive = timed_runs(pure_python_ijk, A, B, n)
    inter_times, C_inter = timed_runs(pure_python_ikj, A, B, n)
    naive_med, inter_med = statistics.median(naive_times), statistics.median(inter_times)
    max_diff = max(abs(x - y) for x, y in zip(C_naive, C_inter))
    print(f"pure Python n={n}: i-j-k {naive_med:.3f} s | i-k-j {inter_med:.3f} s | speedup {naive_med / inter_med:.2f}x")
    return {
        "size": n,
        "naive_ijk_median_s": naive_med,
        "interchanged_ikj_median_s": inter_med,
        "speedup": naive_med / inter_med,
        "max_abs_diff": max_diff,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare i-j-k and i-k-j matrix multiplication loop orders.")
    parser.add_argument("--skip-full", action="store_true", help="Only run the M = N/2 sub-block required by the lab")
    parser.add_argument("--pure-python-size", type=int, default=PURE_PYTHON_SIZE,
                        help="Block size for the interpreter-only comparison (0 to skip)")
    args = parser.parse_args()

    params = load_salted_params()
    N, seed, student_id = int(params["N"]), int(params["seed"]), str(params["student_id"])
    rng = np.random.default_rng(seed)
    A_full = rng.random((N, N), dtype=np.float64)
    B_full = rng.random((N, N), dtype=np.float64)

    if njit is not None:  # trigger JIT compilation outside the timed region
        tiny = np.ones((4, 4))
        run_kernel(matmul_ijk, tiny, tiny)
        run_kernel(matmul_ikj, tiny, tiny)

    print(f"N={N} seed={seed} backend={BACKEND} runs={RUNS}")
    sizes = [N // 2] if args.skip_full else [N // 2, N]
    blocks = [benchmark_size(A_full, B_full, m, student_id) for m in sizes]
    pure = benchmark_pure_python(A_full, B_full, args.pure_python_size) if args.pure_python_size else None

    try:
        caches = load_host_profile()["caches"]
        cache_info = {
            "cache_line_bytes": caches.get("cache_line_bytes"),
            "l1d_bytes": caches.get("l1d_bytes"),
            "l2_bytes": caches.get("l2_bytes"),
            "l3_bytes": caches.get("l3_bytes"),
        }
    except (FileNotFoundError, KeyError):
        cache_info = {}

    primary = blocks[0]
    output = {
        "student_id": student_id,
        "N": N,
        "seed": seed,
        "backend": BACKEND,
        "runs_per_kernel": RUNS,
        "primary_block_M": primary["M"],
        "T_naive_s": primary["naive_ijk"]["median_s"],
        "T_interchanged_s": primary["interchanged_ikj"]["median_s"],
        "speedup": primary["speedup"],
        "blocks": blocks,
        "pure_python_reference": pure,
        "host_caches": cache_info,
    }
    out = results_path("loop_interchange.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
