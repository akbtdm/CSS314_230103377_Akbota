#!/usr/bin/env python3
"""
Task 22 - Empirical Weak-Scaling Benchmark (Python multiprocessing).

Workload: numerical integration of f(x) = sqrt(3 x^2 + 1) * cos(x / 4) on [0, 1),
the same kernel family as the Karp-Flatt midterm.

Weak scaling: every worker always integrates K points (K = --k, constant per
process). With P workers the global problem has P*K points, so total work
grows linearly with P and the ideal wall-clock time stays constant.

Measurement design
  * The Pool is created and warmed up BEFORE timing, so process spawn cost
    (large on macOS, which uses 'spawn') is excluded from T(P). It is recorded
    separately in column pool_startup_s.
  * T(P) = partition + dispatch + compute + gather + final reduction (master).
    This is the full parallel run time that Gustafson's alpha is measured on.
  * Runs are interleaved (run 1 for all P, then run 2 for all P, ...) so that
    thermal drift does not bias one configuration.

Output: results.csv with one row per (run, P).
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time

import numpy as np

K1, K2 = 3, 4
CHUNK = 1_000_000  # bounds temporary arrays to ~8 MB each


def kernel(start: int, count: int, total: int) -> float:
    """Integrate a contiguous slice [start, start+count) of a 'total'-point grid."""
    dx = 1.0 / total
    acc = 0.0
    end = start + count
    for lo in range(start, end, CHUNK):
        hi = min(lo + CHUNK, end)
        x = np.arange(lo, hi, dtype=np.float64) * dx
        acc += float(np.sum(np.sqrt(K1 * x * x + 1.0) * np.cos(x / K2)))
    return acc * dx


def _warmup(_):
    return kernel(0, 1000, 1000)


def run_once(pool, p: int, k: int) -> tuple:
    t0 = time.perf_counter()
    total = p * k
    tasks = [(r * k, k, total) for r in range(p)]        # serial: partition
    partials = pool.starmap(kernel, tasks, chunksize=1)  # parallel: compute
    result = float(sum(partials))                         # serial: reduction
    return time.perf_counter() - t0, result


def hardware_info() -> dict:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "cpu_count_logical": os.cpu_count(),
        "mp_start_method": mp.get_start_method(),
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
    }
    if sys.platform == "darwin":
        keys = ["machdep.cpu.brand_string", "hw.physicalcpu", "hw.logicalcpu", "hw.memsize",
                "hw.perflevel0.physicalcpu", "hw.perflevel0.l2cachesize",
                "hw.perflevel1.physicalcpu", "hw.perflevel1.l2cachesize",
                "hw.l1dcachesize", "hw.cachelinesize"]
        for key in keys:
            try:
                info[key] = subprocess.run(["sysctl", "-n", key], capture_output=True,
                                           text=True, timeout=5).stdout.strip()
            except Exception:
                pass
    elif sys.platform.startswith("linux"):
        try:
            info["lscpu"] = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=5).stdout
        except Exception:
            pass
    return info


def default_p_list(max_p: int):
    base = [1, 2, 4, 6, 8, 10, 12, 16, 24, 32]
    lst = sorted({p for p in base if p <= max_p} | {max_p})
    return lst


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=20_000_000, help="points per process (constant)")
    ap.add_argument("--procs", type=int, nargs="+", help="process counts (default: 1,2,4,... up to max)")
    ap.add_argument("--max-procs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--cooldown", type=float, default=1.0, help="seconds to sleep between measurements")
    ap.add_argument("--out", default=os.path.join(here, "..", "results.csv"))
    args = ap.parse_args(argv)

    procs = sorted(set(args.procs)) if args.procs else default_p_list(args.max_procs)
    if len(procs) < 5:
        print(f"[!] Warning: only {len(procs)} distinct process counts; Task 23 requires >= 5.")

    hw = hardware_info()
    out = os.path.abspath(args.out)
    with open(os.path.join(os.path.dirname(out), "hardware.json"), "w") as f:
        json.dump(hw, f, indent=2)

    print("=" * 70)
    print(f"Weak-scaling benchmark | K = {args.k:,} points/process | runs = {args.runs}")
    print(f"Host: {hw.get('machdep.cpu.brand_string', hw['processor'])} | logical CPUs: {hw['cpu_count_logical']} "
          f"| start method: {hw['mp_start_method']}")
    print(f"Process counts: {procs}")
    print("=" * 70)

    rows = []
    for run in range(1, args.runs + 1):
        for p in procs:
            t_s = time.perf_counter()
            pool = mp.Pool(processes=p)
            pool.map(_warmup, range(p), chunksize=1)  # force every worker to start + import numpy
            startup = time.perf_counter() - t_s
            wall, result = run_once(pool, p, args.k)
            pool.close()
            pool.join()
            rows.append({"run": run, "procs": p, "k_per_proc": args.k, "total_points": p * args.k,
                         "wall_s": round(wall, 6), "pool_startup_s": round(startup, 6),
                         "result": round(result, 10)})
            print(f"run {run} | P = {p:3d} | T(P) = {wall:8.4f} s | startup = {startup:6.3f} s | I = {result:.8f}")
            time.sleep(args.cooldown)

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[+] Raw data: {out}")
    print(f"[+] Hardware info: {os.path.join(os.path.dirname(out), 'hardware.json')}")
    print("[+] Next: python3 benchmarks/fit_gustafson.py")


if __name__ == "__main__":
    main()
