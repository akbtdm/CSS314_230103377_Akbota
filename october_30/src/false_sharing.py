"""Milestone 2: false sharing pathology, remediation, and Karp-Flatt diagnosis.

P worker processes each increment their own 64-bit counter K times inside a
shared ctypes array. Three layouts are compared:

  unpadded     worker p writes slot p          -> all counters on ONE cache line
  padded_64B   worker p writes slot 8*p        -> 64 B apart (lab default; on a
                                                  128 B line two workers still share)
  padded_line  worker p writes slot (line/8)*p -> one counter per physical cache line
  privatized   worker keeps a local counter and writes the shared slot once

The array is created with lock=False: every worker owns a distinct slot, so no
synchronisation is needed and the measurement isolates cache-coherence traffic
rather than lock contention. Each worker times only its own loop after a
Barrier releases all workers together; T_P is the slowest worker's loop time.

Two Karp-Flatt readings are reported. `karp_flatt_e` applies the lab formula
e(P) = (P*T_P/T_1 - 1)/(P - 1) directly to the measured loop times; because every
worker performs K increments regardless of P, a perfectly scaling layout gives
T_P == T_1 and e == 1.0, while contention pushes e well above 1. `karp_flatt_e_scaled`
treats the total work P*K as the problem size (T_1 scaled to P*T_1), so a perfectly
scaling layout gives e == 0 and contention gives e > 0.

Usage:
    python3 src/false_sharing.py                 # both backends if Numba is present
    python3 src/false_sharing.py --backend python
Writes:
    results/false_sharing_karp_flatt.tsv
"""
import argparse
import ctypes
import multiprocessing as mp
import statistics
import time

import numpy as np

from common import cache_line_bytes, results_path

PROCESS_COUNTS = (1, 2, 4, 8)
ITERATIONS = 5_000_000
REPEATS = 3
BYTES_PER_SLOT = 8
VARIANTS = ("unpadded", "padded_64B", "padded_line", "privatized")

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


# --------------------------------------------------------------------------- kernels
def python_shared(arr, slot, iterations, _mask):
    for _ in range(iterations):
        arr[slot] += 1


def python_private(arr, slot, iterations, _mask):
    local = 0
    for _ in range(iterations):
        local += 1
    arr[slot] = local


def _numba_shared(view, slot, iterations, mask):
    # mask is 0 at run time; the compiler cannot prove it, so the read-modify-write
    # really hits memory every iteration (equivalent of `volatile` in C).
    for i in range(iterations):
        view[slot + (i & mask)] += 1


def _numba_private(view, slot, iterations, mask):
    local = 0
    for i in range(iterations):
        local += 1 + (i & mask)
    view[slot] = local


if njit is not None:
    numba_shared = njit(nogil=True)(_numba_shared)
    numba_private = njit(nogil=True)(_numba_private)


def kernel_for(backend, variant):
    if backend == "python":
        return python_private if variant == "privatized" else python_shared
    return numba_private if variant == "privatized" else numba_shared


# --------------------------------------------------------------------------- worker
def worker_main(backend, variant, raw, slot, iterations, barrier, durations, worker_id):
    kernel = kernel_for(backend, variant)
    if backend == "numba":
        target = np.ctypeslib.as_array(raw)
        kernel(np.zeros(4, dtype=np.int64), 0, 4, 0)  # JIT compile before the barrier
    else:
        target = raw
    barrier.wait()
    t0 = time.perf_counter_ns()
    kernel(target, slot, iterations, 0)
    durations[worker_id] = (time.perf_counter_ns() - t0) / 1e9


# --------------------------------------------------------------------------- driver
def slot_stride(variant, line_bytes):
    if variant == "unpadded":
        return 1
    if variant == "padded_64B":
        return 64 // BYTES_PER_SLOT
    return line_bytes // BYTES_PER_SLOT  # padded_line and privatized


def aligned_base_slot(raw, line_bytes):
    """First slot whose address sits on a cache-line boundary."""
    addr = ctypes.addressof(raw)
    return ((-addr) % line_bytes) // BYTES_PER_SLOT


def run_group(ctx, backend, variant, P, iterations, line_bytes):
    stride = slot_stride(variant, line_bytes)
    slots_per_line = line_bytes // BYTES_PER_SLOT
    raw = mp.RawArray(ctypes.c_int64, slots_per_line + max(PROCESS_COUNTS) * stride)
    base = aligned_base_slot(raw, line_bytes)
    durations = mp.RawArray(ctypes.c_double, P)
    barrier = ctx.Barrier(P)

    procs = [ctx.Process(target=worker_main,
                         args=(backend, variant, raw, base + p * stride, iterations, barrier, durations, p))
             for p in range(P)]
    wall0 = time.perf_counter()
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
    wall = time.perf_counter() - wall0
    if any(proc.exitcode != 0 for proc in procs):
        raise RuntimeError(f"{backend}/{variant} P={P}: a worker exited with a non-zero code")

    total = sum(raw[base + p * stride] for p in range(P))
    return max(durations), wall, total, stride * BYTES_PER_SLOT


def karp_flatt(T1, TP, P):
    """Lab formula on raw times; equals 1.0 when T_P == T_1 (fixed work per worker)."""
    if P == 1:
        return float("nan")
    return (P * TP / T1 - 1.0) / (P - 1.0)


def karp_flatt_scaled(T1, TP, P):
    """Karp-Flatt with the serial reference scaled to the same total work P*K."""
    if P == 1:
        return float("nan")
    return (TP / T1 - 1.0) / (P - 1.0)


def main():
    parser = argparse.ArgumentParser(description="Reproduce and remediate false sharing across processes.")
    parser.add_argument("--backend", choices=("python", "numba", "both"), default="both")
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    args = parser.parse_args()

    backends = ["python", "numba"] if args.backend == "both" else [args.backend]
    if "numba" in backends and njit is None:
        print("Numba not available; running the Python backend only.")
        backends = ["python"]

    ctx = mp.get_context("spawn")
    line_bytes = cache_line_bytes()
    print(f"cache line = {line_bytes} B | K = {args.iterations:,} increments per worker | repeats = {args.repeats} "
          f"| start method = spawn")

    rows = []
    for backend in backends:
        print(f"\n=== backend: {backend} ===")
        print(f"{'variant':<12} {'P':>2} {'T_loop_s':>9} {'ns/inc':>8} {'S_P':>6} {'eff%':>6} {'e(P)':>8} {'e_scaled':>9}  count")
        for variant in VARIANTS:
            T1 = None
            for P in PROCESS_COUNTS:
                loop_times, walls, totals, stride_b = [], [], [], None
                for _ in range(args.repeats):
                    t_loop, wall, total, stride_b = run_group(ctx, backend, variant, P, args.iterations, line_bytes)
                    loop_times.append(t_loop)
                    walls.append(wall)
                    totals.append(total)
                T_P = statistics.median(loop_times)
                if P == 1:
                    T1 = T_P
                expected = P * args.iterations
                count_ok = all(t == expected for t in totals)
                speedup = T1 / T_P
                speedup_scaled = P * T1 / T_P
                eff = speedup_scaled / P * 100.0
                e = karp_flatt(T1, T_P, P)
                e_scaled = karp_flatt_scaled(T1, T_P, P)
                ns_per_inc = T_P / args.iterations * 1e9
                rows.append((backend, variant, P, args.iterations, args.repeats, stride_b, T_P, min(loop_times),
                             statistics.median(walls), ns_per_inc, speedup, speedup_scaled, eff, e, e_scaled,
                             totals[-1], expected, count_ok))
                e_txt = f"{e:8.3f} {e_scaled:9.3f}" if P > 1 else "       -         -"
                print(f"{variant:<12} {P:>2} {T_P:>9.4f} {ns_per_inc:>8.2f} {speedup:>6.2f} {eff:>6.1f} {e_txt}  "
                      f"{totals[-1]}/{expected} {'ok' if count_ok else 'MISMATCH'}", flush=True)

    out = results_path("false_sharing_karp_flatt.tsv")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(f"# cache_line_bytes={line_bytes}\titerations_per_worker={args.iterations}\trepeats={args.repeats}\t"
                 f"start_method=spawn\tlock=False\ttiming=max_worker_loop_time_after_barrier\n")
        fh.write("backend\tvariant\tP\tK\trepeats\tslot_stride_bytes\tT_P_s\tT_P_min_s\tparent_wall_s\t"
                 "ns_per_increment\tspeedup_T1_over_TP\tscaled_speedup_P_T1_over_TP\tscaled_efficiency_pct\t"
                 "karp_flatt_e\tkarp_flatt_e_scaled\ttotal_count\texpected_count\tcount_ok\n")
        for r in rows:
            (backend, variant, P, K, reps, stride_b, T_P, T_min, wall, ns_inc, S, S_sc, eff, e, e_sc,
             total, expected, ok) = r
            e_txt = "" if P == 1 else f"{e:.6f}"
            e_sc_txt = "" if P == 1 else f"{e_sc:.6f}"
            fh.write(f"{backend}\t{variant}\t{P}\t{K}\t{reps}\t{stride_b}\t{T_P:.6f}\t{T_min:.6f}\t{wall:.6f}\t"
                     f"{ns_inc:.3f}\t{S:.4f}\t{S_sc:.4f}\t{eff:.2f}\t{e_txt}\t{e_sc_txt}\t{total}\t{expected}\t{ok}\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
