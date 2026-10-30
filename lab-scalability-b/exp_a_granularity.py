"""Experiment A: the granularity trap.

Collatz step counts for 1..1,000,000 are mapped over a Pool twice: with chunksize=1
(one IPC task envelope per number, "micro-tasks") and with chunksize=10,000 (100
amortised chunks). With micro-tasks the master process spends longer pickling and
dispatching each task than a worker spends computing it, so adding workers cannot
help; with coarse chunks the same work shows healthy strong scaling.

Usage:
    python3 exp_a_granularity.py [--repeats 3] [--cores 1 2 4 8]
Writes:
    results/exp_a_granularity.json
"""
import argparse
import time
from multiprocessing import Pool

from kf_report import DEFAULT_REPEATS, benchmark, print_rows, row_for, save, summarize

CORES = (1, 2, 4, 8)
TOTAL_ITEMS = 1_000_000
CHUNK_SIZES = ((1, "micro_k1"), (10_000, "coarse_k10000"))


def collatz_steps(n):
    steps = 0
    while n > 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps


def run_granularity(workers, total_items=TOTAL_ITEMS, chunk_size=1):
    numbers = range(1, total_items + 1)
    t0 = time.perf_counter()
    with Pool(processes=workers) as pool:
        results = pool.map(collatz_steps, numbers, chunksize=chunk_size)
    return time.perf_counter() - t0, sum(results)


def main():
    parser = argparse.ArgumentParser(description="Experiment A: chunksize 1 vs 10,000 across worker counts.")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--cores", type=int, nargs="+", default=list(CORES))
    args = parser.parse_args()

    checksums = set()

    def timed(p, k):
        elapsed, total = run_granularity(p, chunk_size=k)
        checksums.add(total)
        return elapsed

    series = []
    for k, name in CHUNK_SIZES:
        all_times = benchmark(lambda p, k=k: timed(p, k), args.cores, args.repeats, f"exp A {name}")
        rows = summarize(all_times, args.cores)
        print(f"Experiment A | chunksize = {k:,} | tasks dispatched = {TOTAL_ITEMS // k:,}")
        print_rows(rows)
        r1, r8 = row_for(rows, 1), row_for(rows, 8)
        print(f"  T8/T1 = {r8['T_s'] / r1['T_s']:.2f} | us per task at P=1: {r1['T_s'] / (TOTAL_ITEMS // k) * 1e6:.1f}\n")
        series.append({"chunksize": k, "name": name, "tasks": TOTAL_ITEMS // k, "rows": rows,
                       "all_times": {str(p): t for p, t in all_times.items()}})

    assert len(checksums) == 1, f"inconsistent Collatz checksums: {checksums}"
    micro, coarse = series[0]["rows"], series[1]["rows"]
    penalty = {p: row_for(micro, p)["T_s"] / row_for(coarse, p)["T_s"] for p in args.cores}
    print("micro/coarse runtime ratio per P: " + ", ".join(f"P={p}: {v:.1f}x" for p, v in penalty.items()))

    save("exp_a_granularity", {
        "repeats": args.repeats, "cores": args.cores, "total_items": TOTAL_ITEMS,
        "collatz_steps_checksum": checksums.pop(),
        "series": series,
        "micro_over_coarse_ratio": {str(p): v for p, v in penalty.items()},
        "diagnosis": "chunksize=1: master dispatch serialization, flat runtime and high e(p); "
                     "chunksize=10,000: amortised dispatch, healthy strong scaling.",
    })


if __name__ == "__main__":
    main()
