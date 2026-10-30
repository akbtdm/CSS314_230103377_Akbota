"""Experiment C: local buffer chunking (amortised synchronisation).

Workers accumulate results in a private list and flush one pre-aggregated value to
the shared Queue every BATCH_SIZE items, then send a None sentinel. The total volume
is 25x larger than in Experiment B (10 M vs 400 k items) yet far fewer IPC crossings
happen, so scaling should be restored and the Karp-Flatt metric should collapse.

Usage:
    python3 exp_c_buffered.py [--repeats 3] [--cores 1 2 4 8] [--total 10000000]
Writes:
    results/exp_c_buffered.json
"""
import argparse
import time
from multiprocessing import Process, Queue

from kf_report import DEFAULT_REPEATS, benchmark, print_rows, row_for, save, summarize

CORES = (1, 2, 4, 8)
TOTAL_ITEMS = 10_000_000
BATCH_SIZE = 10_000


def buffered_worker(out_queue, items_per_worker, batch_size=BATCH_SIZE):
    buffer = []
    for i in range(items_per_worker):
        buffer.append(i % 7)
        if len(buffer) >= batch_size:
            out_queue.put(sum(buffer))  # flush one pre-aggregated chunk per batch_size items
            buffer.clear()
    if buffer:
        out_queue.put(sum(buffer))
    out_queue.put(None)  # sentinel end-of-work token


def run_buffered(p, total_items=TOTAL_ITEMS):
    q = Queue()
    items_per_worker = total_items // p
    workers = [Process(target=buffered_worker, args=(q, items_per_worker)) for _ in range(p)]
    t0 = time.perf_counter()
    for w in workers:
        w.start()
    total_sum, active = 0, p
    while active > 0:
        val = q.get()
        if val is None:
            active -= 1
        else:
            total_sum += val
    for w in workers:
        w.join()
    elapsed = time.perf_counter() - t0
    expected = p * sum(i % 7 for i in range(items_per_worker))
    assert total_sum == expected, f"total {total_sum} != expected {expected}"
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Experiment C: buffered batches on the queue, scaling restored.")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--cores", type=int, nargs="+", default=list(CORES))
    parser.add_argument("--total", type=int, default=TOTAL_ITEMS)
    args = parser.parse_args()

    all_times = benchmark(lambda p: run_buffered(p, args.total), args.cores, args.repeats, "exp C")
    rows = summarize(all_times, args.cores)
    print(f"Experiment C (buffered batches of {BATCH_SIZE:,})")
    print_rows(rows)

    r1, r4, r8 = row_for(rows, 1), row_for(rows, 4), row_for(rows, 8)
    puts = args.total // BATCH_SIZE
    print(f"IPC crossings: {puts:,} puts for {args.total:,} items (vs one per item in Experiment B) | "
          f"S4 = {r4['speedup']:.2f}x, S8 = {r8['speedup']:.2f}x, e(8) = {r8['karp_flatt_e']:.3f} | "
          f"items/s at P=8: {args.total / r8['T_s']:,.0f}")

    save("exp_c_buffered", {
        "repeats": args.repeats, "cores": args.cores, "total_items": args.total, "batch_size": BATCH_SIZE,
        "ipc_puts": puts,
        "rows": rows, "all_times": {str(p): t for p, t in all_times.items()},
        "items_per_second_p1": args.total / r1["T_s"],
        "items_per_second_p8": args.total / r8["T_s"],
        "rubric_s8_above_5": r8["speedup"] > 5.0,
        "rubric_e8_at_most_0_05": r8["karp_flatt_e"] <= 0.05,
        "diagnosis": "Collapsed e(p): one synchronisation per batch instead of per item; remaining loss is "
                     "process start-up and the slower efficiency cores at P=8.",
    })


if __name__ == "__main__":
    main()
