"""Task 2: hunting the hardware ceiling (empirical Amdahl analysis).

Two-phase pipeline over N floats. Phase A sorts the first K = 5% of the elements on
the main process (strictly serial). Phase B maps g(x) = log(x + 1) * cos(x) over the
remaining N - K elements with a Pool of p workers. The sweep p in {1, 2, 3, 4, 6, 8,
12, 16} crosses this machine's performance-core count (4), its physical/logical core
count (10) and goes into oversubscription (12, 16).

Two extra diagnostics explain *why* scaling halts where it does:
  * the parent-side serialization floor: time the main process needs just to pickle
    the input chunks and unpickle same-sized results, work no worker can share;
  * a "heavy" variant of the same sweep whose kernel repeats the transform
    HEAVY_REPEATS times per element, so compute dominates IPC and the core ceiling
    itself becomes visible.

Parameters derived from the student ID:
    N = 8_000_000 + (STUDENT_ID % 500) * 10_000
    s = 0.05, K = int(N * s)

Usage:
    python3 task2_amdahl.py
Writes:
    task2_results.csv
"""
import csv
import math
import os
import pickle
import random
import statistics
import subprocess
import time
from multiprocessing import Pool

STUDENT_ID = 230103377
N = 8_000_000 + (STUDENT_ID % 500) * 10_000
SERIAL_FRACTION = 0.05
K = int(N * SERIAL_FRACTION)
PROCESS_COUNTS = [1, 2, 3, 4, 6, 8, 12, 16]
CHUNKSIZE = 10_000
REPEATS = 3
WARMUP_ITEMS = 1_000
HEAVY_REPEATS = 8
HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(HERE, "task2_results.csv")


def sysctl_int(key):
    try:
        out = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=False)
        return int(out.stdout.strip()) if out.returncode == 0 and out.stdout.strip() else None
    except (FileNotFoundError, ValueError):
        return None


def parallel_kernel(x):
    return math.log(x + 1.0) * math.cos(x)


def heavy_kernel(x):
    """Diagnostic variant: the lab transform applied HEAVY_REPEATS times per element."""
    y = x
    for _ in range(HEAVY_REPEATS):
        y = math.log(abs(y) + 1.0) * math.cos(y) + x
    return y


def amdahl_speedup(p, s=SERIAL_FRACTION):
    return 1.0 / (s + (1.0 - s) / p)


def karp_flatt(speedup, p):
    if p < 2:
        return float("nan")
    return (1.0 / speedup - 1.0 / p) / (1.0 - 1.0 / p)


def serial_phase(raw_data):
    t0 = time.perf_counter()
    sorted(raw_data[:K])
    return time.perf_counter() - t0


def parallel_phase(pool, kernel, parallel_data):
    t0 = time.perf_counter()
    pool.map(kernel, parallel_data, chunksize=CHUNKSIZE)
    return time.perf_counter() - t0


def serialization_floor(parallel_data):
    """Parent-only cost of pickling the input chunks and unpickling same-sized results."""
    t0 = time.perf_counter()
    blobs = [pickle.dumps(parallel_data[i:i + CHUNKSIZE]) for i in range(0, len(parallel_data), CHUNKSIZE)]
    for blob in blobs:
        pickle.loads(blob)
    return time.perf_counter() - t0


def run_sweep(variant, kernel, raw_data, parallel_data, floor):
    rows, t1_total = [], None
    for p in PROCESS_COUNTS:
        serial_trials, parallel_trials = [], []
        with Pool(processes=p) as pool:
            pool.map(kernel, parallel_data[:WARMUP_ITEMS], chunksize=100)  # workers are up before timing
            for _ in range(REPEATS):
                serial_trials.append(serial_phase(raw_data))
                parallel_trials.append(parallel_phase(pool, kernel, parallel_data))
        t_serial = statistics.median(serial_trials)
        t_parallel = statistics.median(parallel_trials)
        total = t_serial + t_parallel
        if p == 1:
            t1_total = total
        speedup = t1_total / total
        rows.append({
            "Variant": variant, "Processes": p, "SerialTime": t_serial, "ParallelTime": t_parallel,
            "TotalTime": total, "Speedup": speedup, "EfficiencyPct": speedup / p * 100.0,
            "AmdahlPredictedSpeedup": amdahl_speedup(p), "KarpFlattSerialFraction": karp_flatt(speedup, p),
            "ParentSerializationFloorS": floor,
        })
        print(f"[{variant}] p: {p:2d} | Serial: {t_serial:.4f}s | Parallel: {t_parallel:.4f}s | Total: {total:.4f}s | "
              f"Speedup: {speedup:.2f}x (Amdahl s=0.05 -> {amdahl_speedup(p):.2f}x) | "
              f"Efficiency: {speedup / p * 100:.0f}% | e(p)={karp_flatt(speedup, p):.3f}", flush=True)
    return rows


def main():
    physical, logical = sysctl_int("hw.physicalcpu") or os.cpu_count(), os.cpu_count()
    print(f"StudentID={STUDENT_ID} N={N:,} s={SERIAL_FRACTION} K={K:,} | physical cores={physical} "
          f"logical processors={logical} | chunksize={CHUNKSIZE} repeats={REPEATS}", flush=True)
    random.seed(STUDENT_ID)
    t0 = time.perf_counter()
    raw_data = [random.uniform(1.0, 100.0) for _ in range(N)]
    parallel_data = raw_data[K:]
    print(f"Allocated dataset of {len(raw_data):,} elements in {time.perf_counter() - t0:.2f} s "
          f"(serial segment {K:,}, parallel segment {len(parallel_data):,})", flush=True)

    floor = serialization_floor(parallel_data)
    print(f"Parent serialization floor (pickle {len(parallel_data):,} floats in {CHUNKSIZE}-element chunks + "
          f"unpickle same-sized results): {floor:.4f} s", flush=True)

    rows = run_sweep("spec", parallel_kernel, raw_data, parallel_data, floor)
    rows += run_sweep(f"heavy_x{HEAVY_REPEATS}", heavy_kernel, raw_data, parallel_data, floor)

    for variant in ("spec", f"heavy_x{HEAVY_REPEATS}"):
        sub = [r for r in rows if r["Variant"] == variant]
        best = max(sub, key=lambda r: r["Speedup"])
        print(f"[{variant}] theoretical ceiling 1/s = {1 / SERIAL_FRACTION:.0f}x | best measured {best['Speedup']:.2f}x "
              f"at p={best['Processes']} | physical={physical} logical={logical}")

    fields = ["Variant", "Processes", "SerialTime", "ParallelTime", "TotalTime", "Speedup", "EfficiencyPct",
              "AmdahlPredictedSpeedup", "KarpFlattSerialFraction", "ParentSerializationFloorS"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
