"""Experiment B: the inter-process queue choke-point.

Every worker pushes each of its results through one shared multiprocessing.Queue,
one put per item, while the master drains the queue. The Queue is a pipe guarded by
a lock with a feeder thread per producer, so every put crosses a lock, a pipe write
and a context switch. Expect runtime to grow with P and the Karp-Flatt metric to surge.

Usage:
    python3 exp_b_queue.py [--repeats 3] [--cores 1 2 4 8] [--total 400000]
Writes:
    results/exp_b_queue.json
"""
import argparse
import time
from multiprocessing import Process, Queue

from kf_report import DEFAULT_REPEATS, benchmark, print_rows, row_for, save, summarize

CORES = (1, 2, 4, 8)
TOTAL_ITEMS = 400_000


def queue_worker(out_queue, items_per_worker):
    for i in range(items_per_worker):
        out_queue.put(i % 7)  # pathological pattern: every iteration crosses an IPC boundary


def run_queue_benchmark(p, total_items=TOTAL_ITEMS):
    q = Queue()
    items_per_worker = total_items // p
    workers = [Process(target=queue_worker, args=(q, items_per_worker)) for _ in range(p)]
    t0 = time.perf_counter()
    for w in workers:
        w.start()
    collected, checksum = 0, 0
    while collected < items_per_worker * p:  # master consumes every item from the shared channel
        checksum += q.get()
        collected += 1
    for w in workers:
        w.join()
    elapsed = time.perf_counter() - t0
    expected = p * sum(i % 7 for i in range(items_per_worker))
    assert checksum == expected, f"checksum {checksum} != expected {expected}"
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Experiment B: one Queue.put per item, surging Karp-Flatt metric.")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--cores", type=int, nargs="+", default=list(CORES))
    parser.add_argument("--total", type=int, default=TOTAL_ITEMS)
    args = parser.parse_args()

    all_times = benchmark(lambda p: run_queue_benchmark(p, args.total), args.cores, args.repeats, "exp B")
    rows = summarize(all_times, args.cores)
    print("Experiment B (queue choke-point)")
    print_rows(rows)

    t1 = row_for(rows, 1)["T_s"]
    times = [row_for(rows, p)["T_s"] for p in args.cores]
    es = [row_for(rows, p)["karp_flatt_e"] for p in args.cores if p > 1]
    monotonic_time = all(a < b for a, b in zip(times, times[1:]))
    monotonic_e = all(a < b for a, b in zip(es, es[1:]))
    print(f"T8/T1 = {row_for(rows, 8)['T_s'] / t1:.2f} | runtime monotonic in P: {monotonic_time} | "
          f"e(p) monotonic: {monotonic_e} | items/s at P=1: {args.total / t1:,.0f} | us per item at P=1: {t1 / args.total * 1e6:.1f}")

    save("exp_b_queue", {
        "repeats": args.repeats, "cores": args.cores, "total_items": args.total,
        "rows": rows, "all_times": {str(p): t for p, t in all_times.items()},
        "t8_over_t1": row_for(rows, 8)["T_s"] / t1,
        "runtime_monotonic_increase": monotonic_time,
        "karp_flatt_monotonic_increase": monotonic_e,
        "items_per_second_p1": args.total / t1,
        "diagnosis": "Surging e(p): lock, pipe and context-switch thrashing on the shared queue, not serial code.",
    })


if __name__ == "__main__":
    main()
