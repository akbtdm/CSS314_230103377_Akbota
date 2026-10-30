"""Shared helpers for the Karp-Flatt practice lab: metric arithmetic, repeated
benchmarking with medians, console tables in the handout's format and JSON output."""
import json
import multiprocessing as mp
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DEFAULT_CORES = (1, 2, 4)
DEFAULT_REPEATS = 3


def karp_flatt(t1, tp, p):
    """e(p) = [(p * Tp / T1) - 1] / (p - 1) for p >= 2."""
    if p < 2:
        return None
    return ((p * tp / t1) - 1.0) / (p - 1.0)


def _sysctl(key):
    try:
        out = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None
    return out.stdout.strip() or None


def _sysctl_int(key):
    value = _sysctl(key)
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def host_profile():
    return {
        "cpu_model": _sysctl("machdep.cpu.brand_string") or platform.processor() or None,
        "execution_cores": os.cpu_count(),
        "physical_cores": _sysctl_int("hw.physicalcpu"),
        "performance_cores": _sysctl_int("hw.perflevel0.physicalcpu"),
        "efficiency_cores": _sysctl_int("hw.perflevel1.physicalcpu"),
        "cache_line_bytes": _sysctl_int("hw.cachelinesize"),
        "os": f"{platform.system()} {platform.release()}",
        "macos_version": platform.mac_ver()[0] or None,
        "python": f"{platform.python_implementation()} {platform.python_version()}",
        "default_start_method": mp.get_start_method(),
    }


def benchmark(run_fn, cores, repeats, label=""):
    """Call run_fn(p) `repeats` times for every p in cores; return {p: [seconds, ...]}."""
    all_times = {}
    for p in cores:
        times = []
        for r in range(repeats):
            t = run_fn(p)
            times.append(t)
            print(f"  [{label}] P={p} run {r + 1}/{repeats}: {t:.3f} s", file=sys.stderr, flush=True)
        all_times[p] = times
    return all_times


def summarize(all_times, cores):
    medians = {p: statistics.median(all_times[p]) for p in cores}
    t1 = medians[cores[0]]
    rows = []
    for p in cores:
        tp = medians[p]
        sp = t1 / tp
        rows.append({
            "P": p,
            "T_s": tp,
            "T_min_s": min(all_times[p]),
            "speedup": sp,
            "efficiency_pct": sp / p * 100.0,
            "karp_flatt_e": karp_flatt(t1, tp, p),
        })
    return rows


def row_for(rows, p):
    return next(r for r in rows if r["P"] == p)


def print_rows(rows):
    for r in rows:
        if r["P"] == 1:
            print(f"P=1 | Runtime={r['T_s']:.3f}s (baseline T1)")
        else:
            print(f"P={r['P']} | Runtime={r['T_s']:.3f}s | Speedup={r['speedup']:.2f}x | "
                  f"Efficiency={r['efficiency_pct']:.0f}% | Karp-Flatt e={r['karp_flatt_e']:.3f}")


def save(name, payload):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    record = {"experiment": name, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "host": host_profile(), **payload}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    print(f"Wrote {path}")
    return path
