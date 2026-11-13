"""Phase 4: skewed triangular workload under block-static, cyclic-static and shared-counter
dynamic chunk scheduling, with per-worker load-imbalance telemetry (task4_scheduling.py)."""
import math
import multiprocessing as mp
import time

from harness import run_benchmark, emit_verify_tag
from labutil import (build_parser, resolve_student_id, results_dir, save_json,
                     hardware_info, median)

N_ITEMS = 12_000
CHUNK_SIZES = [1, 4, 16, 64, 256, 1024]
WARMUP, TIMED = 1, 5


def work_item(i: int) -> float:
    """K(i) = 1 + floor((i/60)^2) heavy sin/cos/sqrt iterations (exact integer form)."""
    k = 1 + (i * i) // 3600
    sin, cos, sqrt = math.sin, math.cos, math.sqrt
    x = 0.5 + (i % 97) * 0.01
    for _ in range(k):
        x = sqrt(1.0 + sin(x) * cos(x))
    return x


def block_worker(t, p, n, times, sums) -> None:
    t0 = time.perf_counter_ns()
    acc = 0.0
    for i in range(t * n // p, (t + 1) * n // p):
        acc += work_item(i)
    times[t] = (time.perf_counter_ns() - t0) / 1e6
    sums[t] = acc


def cyclic_worker(t, p, n, times, sums) -> None:
    t0 = time.perf_counter_ns()
    acc = 0.0
    for i in range(t, n, p):
        acc += work_item(i)
    times[t] = (time.perf_counter_ns() - t0) / 1e6
    sums[t] = acc


def dynamic_worker(t, n, chunk, counter, times, sums, claims) -> None:
    t0 = time.perf_counter_ns()
    acc = 0.0
    n_claims = 0
    lock = counter.get_lock()
    while True:
        with lock:
            start = counter.value
            counter.value = start + chunk
        if start >= n:
            break
        n_claims += 1
        for i in range(start, min(start + chunk, n)):
            acc += work_item(i)
    times[t] = (time.perf_counter_ns() - t0) / 1e6
    sums[t] = acc
    claims[t] = n_claims


def run_static(target, p, n, times, sums) -> None:
    procs = [mp.Process(target=target, args=(t, p, n, times, sums)) for t in range(p)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()


def run_dynamic(p, n, chunk, counter, times, sums, claims) -> None:
    counter.value = 0
    procs = [mp.Process(target=dynamic_worker, args=(t, n, chunk, counter, times, sums, claims))
             for t in range(p)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()


def imbalance(worker_ms) -> float:
    return max(worker_ms) / (sum(worker_ms) / len(worker_ms))


def bench_strategy(label, run_fn, p, times, sums, warmup, timed, extra=None) -> dict:
    snapshots = []

    def trial() -> None:
        run_fn()
        snapshots.append(list(times))

    stats = run_benchmark(trial, warmup_runs=warmup, timed_runs=timed)
    timed_snaps = snapshots[-timed:]
    lambdas = [imbalance(s) for s in timed_snaps]
    row = {"strategy": label, "min_ms": stats.min_ms, "median_ms": stats.median_ms,
           "iqr_ms": stats.iqr_ms, "max_ms": stats.max_ms,
           "lambda_median": median(lambdas), "lambda_runs": lambdas,
           "worker_ms_last": timed_snaps[-1],
           "worker_ms_max": max(timed_snaps[-1]), "worker_ms_mean": sum(timed_snaps[-1]) / p,
           "checksum": sum(sums)}
    if extra:
        row.update(extra)
    print(f"[{label:>16}] T_wall med={stats.median_ms:9.1f} ms iqr={stats.iqr_ms:7.1f}  "
          f"lambda={row['lambda_median']:.3f}  "
          f"workers(ms)=" + " ".join(f"{t:.0f}" for t in timed_snaps[-1]))
    return row


def main() -> None:
    args = build_parser(__doc__).parse_args()
    sid = resolve_student_id(args.id)
    hw = hardware_info()
    p = hw["physical_cores"]
    n = N_ITEMS // 10 if args.quick else N_ITEMS
    chunks = [1, 64, 1024] if args.quick else CHUNK_SIZES
    warmup, timed = (1, 1) if args.quick else (WARMUP, TIMED)
    total_iters = sum(1 + (i * i) // 3600 for i in range(n))

    print(f"=== PHASE 4: scheduling  N={n}  p={p}  total inner iterations={total_iters}  "
          f"K(0)={1}  K(N-1)={1 + (n - 1) ** 2 // 3600}  warmup={warmup} timed={timed} ===")
    times, sums = mp.Array("d", p), mp.Array("d", p)
    claims = mp.Array("i", p)
    counter = mp.Value("i", 0)

    rows = [bench_strategy("S1 block static", lambda: run_static(block_worker, p, n, times, sums),
                           p, times, sums, warmup, timed),
            bench_strategy("S2 cyclic static", lambda: run_static(cyclic_worker, p, n, times, sums),
                           p, times, sums, warmup, timed)]
    for chunk in chunks:
        row = bench_strategy(f"S3 dynamic C={chunk}",
                             lambda: run_dynamic(p, n, chunk, counter, times, sums, claims),
                             p, times, sums, warmup, timed, extra={"chunk": chunk})
        row["claims_per_worker_last"] = list(claims)
        rows.append(row)

    dynamic_rows = [r for r in rows if "chunk" in r]
    best = min(dynamic_rows, key=lambda r: r["median_ms"])
    print(f"\nC* = {best['chunk']}  (T_wall {best['median_ms']:.1f} ms, lambda {best['lambda_median']:.3f})")
    tag = emit_verify_tag("PHASE4", sid, rows[0]["median_ms"], best["median_ms"])
    print("\n" + tag)
    save_json(results_dir(args.quick) / "phase4.json", {
        "student_id": sid, "p": p, "n": n, "total_inner_iterations": total_iters,
        "warmup_runs": warmup, "timed_runs": timed, "rows": rows, "best_chunk": best["chunk"],
        "verify": tag, "hash_inputs": {"primary": rows[0]["median_ms"], "secondary": best["median_ms"]},
    })


if __name__ == "__main__":
    main()
