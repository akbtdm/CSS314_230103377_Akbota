"""Task 3 redesign: lockless thread-local accumulation (map-reduce).

Each thread counts into a private local variable (map), writes its partial
sum once into its own slot of `partials`, and the main thread adds the
slots after join() (reduce). No lock, no shared mutable counter.
The LockedCounter from task3_sync.py is re-run in the same process so the
speedup is measured side by side.
"""
import statistics
import threading
import time

from task3_sync import NUM_THREADS, TOTAL_OPS, LockedCounter, bench

RUNS = 5


def lockless_bench():
    ops_per_thread = TOTAL_OPS // NUM_THREADS
    partials = [0] * NUM_THREADS  # slot i is written once, only by thread i

    def work(slot):
        local = 0
        for _ in range(ops_per_thread):
            local += 1
        partials[slot] = local

    threads = [threading.Thread(target=work, args=(i,)) for i in range(NUM_THREADS)]
    start = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    total = sum(partials)  # single reduction stage
    return total, time.perf_counter() - start


if __name__ == "__main__":
    locked_times, lockless_times = [], []
    for run in range(1, RUNS + 1):
        val_locked, t_locked = bench(LockedCounter)
        val_free, t_free = lockless_bench()
        if val_free != TOTAL_OPS or val_locked != TOTAL_OPS:
            raise SystemExit(f"run {run}: wrong total (lockless={val_free}, locked={val_locked})")
        locked_times.append(t_locked)
        lockless_times.append(t_free)
        print(f"Run {run}: Locked = {val_locked:,} in {t_locked:.4f}s | "
              f"Lockless = {val_free:,} in {t_free:.4f}s | speedup {t_locked / t_free:.2f}x")
    med_l, med_f = statistics.median(locked_times), statistics.median(lockless_times)
    print(f"Median Locked: {med_l:.4f}s | Median Lockless: {med_f:.4f}s | "
          f"Speedup: {med_l / med_f:.2f}x | all {RUNS} lockless totals == {TOTAL_OPS:,}")
