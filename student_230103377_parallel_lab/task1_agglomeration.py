"""Task 1: the agglomeration threshold (Foster's PCAM in action).

Sweeps the chunksize C of multiprocessing.Pool.map over a fixed array of N random
floats with the pool fixed to the PHYSICAL core count. A small C means millions of
tiny tasks, so dispatch, pickling and IPC overhead dominate; a very large C leaves
cores idle while the last stragglers finish.

Parameters derived from the student ID:
    SEED = STUDENT_ID
    N    = 5_000_000 + (STUDENT_ID % 1000) * 10_000

Usage:
    python3 task1_agglomeration.py
Writes:
    task1_results.csv  (StudentID, PhysicalCores, ChunkSize, ElapsedSeconds; median of 3 trials)
"""
import csv
import math
import os
import random
import statistics
import subprocess
import time
from multiprocessing import Pool

STUDENT_ID = 230103377
SEED = STUDENT_ID
N = 5_000_000 + (STUDENT_ID % 1000) * 10_000
CHUNK_SIZES = [1, 10, 100, 1000, 10000, 100000]
REPEATS = 3
WARMUP_ITEMS = 10_000
HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(HERE, "task1_results.csv")


def physical_core_count():
    """Physical cores from sysctl on macOS; falls back to os.cpu_count()."""
    try:
        out = subprocess.run(["sysctl", "-n", "hw.physicalcpu"], capture_output=True, text=True, check=False)
        if out.returncode == 0 and out.stdout.strip():
            return int(out.stdout.strip())
    except (FileNotFoundError, ValueError):
        pass
    return os.cpu_count() or 1


def compute_kernel(x):
    return math.sin(math.sqrt(x))


def generate_data(n, seed):
    random.seed(seed)
    return [random.uniform(1.0, 100.0) for _ in range(n)]


def timed_map(pool, data, chunk):
    t0 = time.perf_counter()
    pool.map(compute_kernel, data, chunksize=chunk)
    return time.perf_counter() - t0


def main():
    cores = physical_core_count()
    print(f"StudentID={STUDENT_ID} SEED={SEED} N={N:,} PhysicalCores={cores} repeats={REPEATS}", flush=True)
    t0 = time.perf_counter()
    data = generate_data(N, SEED)
    print(f"Generated {len(data):,} floats in [1.0, 100.0] in {time.perf_counter() - t0:.2f} s", flush=True)

    rows, medians = [], {}
    with Pool(processes=cores) as pool:
        pool.map(compute_kernel, data[:WARMUP_ITEMS], chunksize=100)  # start the workers before timing
        for c in CHUNK_SIZES:
            trials = [timed_map(pool, data, c) for _ in range(REPEATS)]
            med = statistics.median(trials)
            medians[c] = med
            tasks = math.ceil(N / c)
            print(f"ChunkSize: {c:>6d} | tasks: {tasks:>9,d} | trials: " + ", ".join(f"{t:.3f}" for t in trials)
                  + f" | median: {med:.4f} s", flush=True)
            rows.append((STUDENT_ID, cores, c, f"{med:.6f}"))

    best_c = min(medians, key=medians.get)
    ratio = medians[1] / medians[best_c]
    print(f"Optimal chunk size: C={best_c} ({medians[best_c]:.4f} s) | Time(C=1)/Time(C_opt) = {ratio:.1f}x")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["StudentID", "PhysicalCores", "ChunkSize", "ElapsedSeconds"])
        writer.writerows(rows)
    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
