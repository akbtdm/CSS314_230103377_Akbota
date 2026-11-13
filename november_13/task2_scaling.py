"""Phase 2: strong (Amdahl) vs weak (Gustafson) scaling with a serial Stage A and a
Pool-parallel Stage B (task2_scaling.py)."""
import math
import multiprocessing as mp
import time

from harness import run_benchmark, emit_verify_tag
from labutil import (build_parser, resolve_student_id, results_dir, save_json,
                     hardware_info, partition, median)

N_STRONG = 1_600_000
N0_WEAK = 250_000
SERIAL_FRACTION = 0.05
STAGE_B_ITERS = 35
WARMUP, TIMED = 2, 7


def stage_a(n_serial: int) -> float:
    """Strictly serial prefix reduction: 12 transcendental ops (6 sin + 6 sqrt) per element."""
    sin, sqrt = math.sin, math.sqrt
    acc = 0.0
    for i in range(n_serial):
        x = 0.5 + (i & 1023) * 1e-3
        x = sqrt(sin(x) + 1.5)
        x = sqrt(sin(x) + 1.5)
        x = sqrt(sin(x) + 1.5)
        x = sqrt(sin(x) + 1.5)
        x = sqrt(sin(x) + 1.5)
        x = sqrt(sin(x) + 1.5)
        acc += x
    return acc


def stage_b_chunk(bounds) -> float:
    """Embarrassingly parallel kernel: 35 multiply-add + sin iterations per element."""
    start, end = bounds
    sin = math.sin
    a, b = 1.0000001, 1e-3
    acc = 0.0
    for i in range(start, end):
        x = 0.5 + (i & 1023) * 1e-3
        for _ in range(STAGE_B_ITERS):
            x = sin(x) * a + b
        acc += x
    return acc


def run_trial(pool, p: int, n: int, record: list) -> None:
    t0 = time.perf_counter_ns()
    serial_sum = stage_a(int(SERIAL_FRACTION * n))
    t1 = time.perf_counter_ns()
    partial = pool.map(stage_b_chunk, partition(n, p), chunksize=1)
    t2 = time.perf_counter_ns()
    record.append(((t1 - t0) / 1e6, (t2 - t1) / 1e6, serial_sum + sum(partial)))


def sweep(label: str, p_values, n_for_p, warmup: int, timed: int):
    rows = []
    for p in p_values:
        n = n_for_p(p)
        record = []
        with mp.Pool(processes=p) as pool:  # created outside the timed region
            stats = run_benchmark(lambda: run_trial(pool, p, n, record),
                                  warmup_runs=warmup, timed_runs=timed)
        timed_records = record[-timed:]
        row = {"p": p, "n": n, "min_ms": stats.min_ms, "median_ms": stats.median_ms,
               "iqr_ms": stats.iqr_ms, "max_ms": stats.max_ms,
               "stage_a_ms": median(r[0] for r in timed_records),
               "stage_b_ms": median(r[1] for r in timed_records),
               "checksum": timed_records[-1][2]}
        rows.append(row)
        print(f"[{label}] p={p:2d} N={n:>9}: T={row['median_ms']:9.2f} ms "
              f"(iqr {row['iqr_ms']:.2f})  stageA={row['stage_a_ms']:.2f} ms "
              f"stageB={row['stage_b_ms']:.2f} ms  sum={row['checksum']:.6f}")
    return rows


def add_strong_metrics(rows):
    t1 = rows[0]["median_ms"]
    s = rows[0]["stage_a_ms"] / t1
    prev_speedup = None
    for row in rows:
        p = row["p"]
        row["speedup"] = t1 / row["median_ms"]
        row["efficiency"] = row["speedup"] / p
        row["amdahl_speedup"] = 1.0 / (s + (1.0 - s) / p)
        row["marginal_speedup"] = None if prev_speedup is None else row["speedup"] - prev_speedup
        prev_speedup = row["speedup"]
    return {"serial_fraction_s": s, "amdahl_s_inf": 1.0 / s}


def add_weak_metrics(rows):
    t1 = rows[0]["median_ms"]
    for row in rows:
        p = row["p"]
        row["weak_efficiency"] = t1 / row["median_ms"]
        row["alpha"] = row["stage_a_ms"] / row["median_ms"]
        row["gustafson_speedup"] = p - row["alpha"] * (p - 1)


def print_strong(rows, summary):
    print(f"\nTable 2A — strong scaling, N={rows[0]['n']}  "
          f"s={summary['serial_fraction_s']:.4f}  S_inf=1/s={summary['amdahl_s_inf']:.2f}")
    print(f"{'p':>3} {'T_ms':>10} {'iqr':>7} {'S_p':>7} {'E_p':>7} {'Amdahl':>7} {'dS':>7}")
    for r in rows:
        ds = "-" if r["marginal_speedup"] is None else f"{r['marginal_speedup']:.3f}"
        print(f"{r['p']:>3} {r['median_ms']:>10.2f} {r['iqr_ms']:>7.2f} {r['speedup']:>7.3f} "
              f"{r['efficiency']:>7.3f} {r['amdahl_speedup']:>7.3f} {ds:>7}")


def print_weak(rows):
    print("\nTable 2B — weak scaling, N(p) = p * N0")
    print(f"{'p':>3} {'N':>9} {'T_ms':>10} {'iqr':>7} {'E_weak':>7} {'alpha':>7} {'S_G':>7}")
    for r in rows:
        print(f"{r['p']:>3} {r['n']:>9} {r['median_ms']:>10.2f} {r['iqr_ms']:>7.2f} "
              f"{r['weak_efficiency']:>7.3f} {r['alpha']:>7.4f} {r['gustafson_speedup']:>7.3f}")


def main() -> None:
    args = build_parser(__doc__).parse_args()
    sid = resolve_student_id(args.id)
    hw = hardware_info()
    p_phys, p_log = hw["physical_cores"], hw["logical_threads"]

    n_strong = N_STRONG // 100 if args.quick else N_STRONG
    n0_weak = N0_WEAK // 100 if args.quick else N0_WEAK
    warmup, timed = (1, 3) if args.quick else (WARMUP, TIMED)
    p_values = [1, 2, 3] if args.quick else list(range(1, p_log + 1))

    print(f"=== PHASE 2: scaling  P_phys={p_phys} P_log={p_log} warmup={warmup} timed={timed} ===")
    strong = sweep("2A strong", p_values, lambda p: n_strong, warmup, timed)
    summary = add_strong_metrics(strong)
    print_strong(strong, summary)

    weak = sweep("2B weak", p_values, lambda p: p * n0_weak, warmup, timed)
    add_weak_metrics(weak)
    print_weak(weak)

    s_strong_plog = strong[-1]["speedup"]
    s_gust_plog = weak[-1]["gustafson_speedup"]
    summary["amdahl_speedup_at_plog"] = strong[-1]["amdahl_speedup"]
    summary["observed_strong_speedup_at_plog"] = s_strong_plog
    summary["gustafson_speedup_at_plog"] = s_gust_plog
    tag = emit_verify_tag("PHASE2", sid, s_strong_plog, s_gust_plog)
    print("\n" + tag)
    save_json(results_dir(args.quick) / "phase2.json", {
        "student_id": sid, "p_phys": p_phys, "p_log": p_log,
        "n_strong": n_strong, "n0_weak": n0_weak, "warmup_runs": warmup, "timed_runs": timed,
        "strong": strong, "weak": weak, "summary": summary,
        "verify": tag, "hash_inputs": {"primary": s_strong_plog, "secondary": s_gust_plog},
    })


if __name__ == "__main__":
    main()
