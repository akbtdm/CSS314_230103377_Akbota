"""Experiment B: the contention traffic jam.

No serial code at all: every worker just increments a counter. But the counter is
a single shared integer behind one mutex, acquired on every iteration, so the
cache line holding the lock and the counter bounces between cores and the kernel
puts waiters to sleep. Expect T4 > T1 and a Karp-Flatt metric that surges.

Usage:
    python3 exp_b_contention.py [--repeats 3] [--cores 1 2 4] [--total 2000000]
Writes:
    results/exp_b_contention.json
"""
import argparse
import time
from multiprocessing import Lock, Process, Value

from kf_report import DEFAULT_CORES, DEFAULT_REPEATS, benchmark, print_rows, row_for, save, summarize


def contended_worker(shared_counter, lock, iterations):
    for _ in range(iterations):
        with lock:  # Heavy lock contention & cache-line bouncing
            shared_counter.value += 1


def run_contended(p, total_increments=2_000_000):
    increments_per_worker = total_increments // p
    counter = Value('i', 0)
    lock = Lock()
    procs = [Process(target=contended_worker, args=(counter, lock, increments_per_worker)) for _ in range(p)]
    t0 = time.perf_counter()
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
    t1 = time.perf_counter()
    assert counter.value == total_increments, f"counted {counter.value}, expected {total_increments}"
    return t1 - t0


def main():
    parser = argparse.ArgumentParser(description="Experiment B: lock contention, surging Karp-Flatt metric.")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--cores", type=int, nargs="+", default=list(DEFAULT_CORES))
    parser.add_argument("--total", type=int, default=2_000_000)
    args = parser.parse_args()

    all_times = benchmark(lambda p: run_contended(p, args.total), args.cores, args.repeats, "exp B")
    rows = summarize(all_times, args.cores)
    print("Experiment B (lock contention)")
    print_rows(rows)

    t1, t4 = row_for(rows, 1)["T_s"], row_for(rows, 4)["T_s"]
    e2, e4 = row_for(rows, 2)["karp_flatt_e"], row_for(rows, 4)["karp_flatt_e"]
    print(f"T4/T1 = {t4 / t1:.2f} (negative speedup if > 1) | e(4)/e(2) = {e4 / e2:.2f} | "
          f"ns per locked increment at P=1: {t1 / args.total * 1e9:.0f}")

    save("exp_b_contention", {
        "repeats": args.repeats,
        "cores": args.cores,
        "total_increments": args.total,
        "rows": rows,
        "all_times": {str(p): t for p, t in all_times.items()},
        "t4_over_t1": t4 / t1,
        "e_ratio_4_over_2": e4 / e2,
        "rubric_negative_speedup": t4 > t1,
        "rubric_e_exceeds_1_5": e2 > 1.5 and e4 > 1.5,
        "e4_greater_than_e2": e4 > e2,
        "diagnosis": "Exploding e(p): mutex thrashing and cache-line bouncing, not serial code.",
    })


if __name__ == "__main__":
    main()
