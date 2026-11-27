"""Bonus task: threading.Thread vs multiprocessing.Pool, T = 1..32 workers.

cpu_work() is the handout's code, unchanged. total_work = 50,000,000 is split
evenly across T workers. Pool times include process start-up (spawn), which is
part of the real cost of using processes. Results go to results/bonus_scaling.json
and results/bonus_scaling.png.
"""
import json
import os
import threading
import time
from multiprocessing import Pool

TOTAL_WORK = 50_000_000
WORKERS = [1, 2, 4, 8, 16, 32]
HERE = os.path.dirname(os.path.abspath(__file__))


def cpu_work(n):
    # Pure CPU crunch
    count = 0
    for i in range(n):
        count += i * i
    return count


def run_threads(t):
    chunk = TOTAL_WORK // t
    threads = [threading.Thread(target=cpu_work, args=(chunk,)) for _ in range(t)]
    start = time.perf_counter()
    for th in threads: th.start()
    for th in threads: th.join()
    return time.perf_counter() - start


def run_pool(t):
    chunk = TOTAL_WORK // t
    start = time.perf_counter()
    with Pool(processes=t) as pool:
        pool.map(cpu_work, [chunk] * t)
    return time.perf_counter() - start


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(WORKERS, [r["threading_s"] for r in rows], marker="o", label="threading.Thread (GIL)")
    ax.plot(WORKERS, [r["pool_s"] for r in rows], marker="s", label="multiprocessing.Pool")
    ax.axvline(4, color="grey", ls=":", lw=1)
    ax.axvline(10, color="grey", ls="--", lw=1)
    ax.text(4.1, ax.get_ylim()[1] * 0.95, "4 P-cores", fontsize=8, va="top")
    ax.text(10.3, ax.get_ylim()[1] * 0.95, "10 cores", fontsize=8, va="top")
    ax.set_xscale("log", base=2)
    ax.set_xticks(WORKERS, [str(w) for w in WORKERS])
    ax.set_xlabel("Workers T")
    ax.set_ylabel("Execution time [s]")
    ax.set_title("cpu_work(50,000,000 total) on Apple M4 (4P + 6E), CPython 3.13.5")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    print(f"OS Reported Logical Cores: {os.cpu_count()} | total_work = {TOTAL_WORK:,}")
    print("    T | threading [s] | Pool [s] | Pool speedup vs T=1")
    print("-" * 55)
    rows = []
    for t in WORKERS:
        th = run_threads(t)
        pl = run_pool(t)
        rows.append({"workers": t, "threading_s": th, "pool_s": pl})
        print(f"{t:5d} | {th:13.4f} | {pl:8.4f} | {rows[0]['pool_s'] / pl:6.2f}x")
    with open(os.path.join(HERE, "results", "bonus_scaling.json"), "w") as f:
        json.dump(rows, f, indent=2)
    plot(rows, os.path.join(HERE, "results", "bonus_scaling.png"))
    print("saved results/bonus_scaling.json and results/bonus_scaling.png")
