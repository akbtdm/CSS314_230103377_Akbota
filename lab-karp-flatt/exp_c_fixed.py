"""Experiment C: the privatization fix.

Each worker accumulates into a private local variable with zero shared state and
the results are reduced exactly once at the end. Scaling should be restored and
the Karp-Flatt metric should collapse towards zero.

Three series are recorded, because on macOS the default `spawn` start method
costs tens of milliseconds per pool and that fixed cost is itself a serial
fraction visible at the handout's 20 M increments:
    spawn, 20 M increments   (handout defaults, macOS default start method)
    fork,  20 M increments   (Linux-like start-up cost)
    spawn, 200 M increments  (work large enough to amortise start-up)

Usage:
    python3 exp_c_fixed.py [--repeats 3] [--cores 1 2 4]
Writes:
    results/exp_c_fixed.json
"""
import argparse
import multiprocessing as mp
import time
from multiprocessing import Pool

from kf_report import DEFAULT_CORES, DEFAULT_REPEATS, benchmark, print_rows, row_for, save, summarize

SERIES = (("spawn", 20_000_000), ("fork", 20_000_000), ("spawn", 200_000_000))


def private_worker(iterations):
    # Thread-local accumulation in private memory space (zero shared state)
    local_sum = 0
    for _ in range(iterations):
        local_sum += 1
    return local_sum


def run_privatized(p, total_increments=20_000_000, ctx=None):
    increments_per_worker = total_increments // p
    tasks = [increments_per_worker] * p
    pool_factory = ctx.Pool if ctx is not None else Pool
    t0 = time.perf_counter()
    with pool_factory(processes=p) as pool:
        results = pool.map(private_worker, tasks)
    total = sum(results)  # Synchronize exactly ONCE at the end
    t1 = time.perf_counter()
    assert total == total_increments, f"counted {total}, expected {total_increments}"
    return t1 - t0


def main():
    parser = argparse.ArgumentParser(description="Experiment C: privatized counters, scaling restored.")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--cores", type=int, nargs="+", default=list(DEFAULT_CORES))
    args = parser.parse_args()

    series_out = []
    for method, total in SERIES:
        ctx = mp.get_context(method)
        label = f"exp C {method} {total // 1_000_000}M"
        all_times = benchmark(lambda p, t=total, c=ctx: run_privatized(p, t, c), args.cores, args.repeats, label)
        rows = summarize(all_times, args.cores)
        print(f"Experiment C (privatized) | start method = {method} | total increments = {total:,}")
        print_rows(rows)
        t1, t4 = row_for(rows, 1)["T_s"], row_for(rows, 4)["T_s"]
        e4_eff = row_for(rows, 4)["efficiency_pct"]
        print(f"T4 < T1: {t4 < t1} | E4 = {e4_eff:.0f}% (rubric > 70%)\n")
        series_out.append({
            "start_method": method,
            "total_increments": total,
            "rows": rows,
            "all_times": {str(p): t for p, t in all_times.items()},
            "rubric_t4_below_t1": t4 < t1,
            "rubric_e4_efficiency_above_70": e4_eff > 70.0,
        })

    save("exp_c_fixed", {
        "repeats": args.repeats,
        "cores": args.cores,
        "series": series_out,
        "diagnosis": "Collapsed e(p): private accumulation with a single final reduction restores near-linear scaling; "
                     "residual e comes from process start-up, which spawn makes visible at small workloads.",
    })


if __name__ == "__main__":
    main()
