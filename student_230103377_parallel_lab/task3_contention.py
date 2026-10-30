"""Task 3: the shared-resource bottleneck (compute-bound vs memory-bound scaling).

Workload A, compute-bound: 20,000 floats per worker, 500 sin/cos iterations each.
Workload B, memory-bound:  20,000,000 floats per worker, one multiplication per element,
                           streamed through a stdlib array('d') buffer (160 MB per worker).
                           A Python list of 20 M fresh float objects would need ~800 MB per
                           worker and push 10 workers into swap, which would corrupt the timing.
Workload B2, STREAM copy:  extra stdlib-only check: 128 MiB buffers copied with slice
                           assignment (memcpy speed), i.e. zero interpreter work per byte, so
                           the DRAM bus itself is the only limiter.

Each workload runs on 1 worker, on the performance-core count and on M = logical
processors. Every worker gets the same amount of work, so the throughput speedup is
T_1 * workers / T_workers (ideal = workers), exactly as in the lab template.

Usage:
    python3 task3_contention.py
Writes:
    task3_results.csv
"""
import csv
import math
import os
import statistics
import subprocess
import time
from array import array
from multiprocessing import Pool, cpu_count

STUDENT_ID = 230103377
M_LOGICAL = cpu_count()
A_COUNT, A_ITERATIONS = 20_000, 500
B_SIZE = 20_000_000
STREAM_BYTES, STREAM_PASSES = 128 * 1024 * 1024, 40
REPEATS = 3
HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(HERE, "task3_results.csv")


def sysctl_int(key):
    try:
        out = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=False)
        return int(out.stdout.strip()) if out.returncode == 0 and out.stdout.strip() else None
    except (FileNotFoundError, ValueError):
        return None


def compute_bound_worker(count):
    total = 0.0
    for _ in range(count):
        x = 0.5
        for _ in range(A_ITERATIONS):
            x = math.sin(x) + math.cos(x)
        total += x
    return total


def memory_bound_worker(size):
    arr = array("d", [1.0001]) * size          # 20 M doubles = 160 MB, contiguous
    for i in range(len(arr)):                  # streaming pass, one multiply per element
        arr[i] = arr[i] * 1.0001
    return arr[0]


def stream_copy_worker(args):
    nbytes, passes = args
    src, dst = bytearray(nbytes), bytearray(nbytes)
    for _ in range(passes):
        dst[:] = src                           # memcpy: nbytes read + nbytes written
        src, dst = dst, src
    return len(src)


WORKLOADS = [
    # name, worker, full argument, warm-up argument, description
    ("Compute_Bound", compute_bound_worker, A_COUNT, 20,
     f"{A_COUNT:,} floats x {A_ITERATIONS} sin/cos iterations per worker"),
    ("Memory_Bound", memory_bound_worker, B_SIZE, 1_000,
     f"{B_SIZE:,} doubles per worker, y = x * 1.0001 streamed through array('d')"),
    ("Memory_Bound_StreamCopy", stream_copy_worker, (STREAM_BYTES, STREAM_PASSES), (4096, 1),
     f"{STREAM_PASSES} memcpy passes over {STREAM_BYTES // 2**20} MiB per worker (no arithmetic)"),
]


def bench(worker, arg, warm_arg, procs):
    with Pool(processes=procs) as pool:
        pool.map(worker, [warm_arg] * procs)   # workers are up and imports done before timing
        trials = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            pool.map(worker, [arg] * procs)
            trials.append(time.perf_counter() - t0)
    return statistics.median(trials), trials


def main():
    perf_cores = sysctl_int("hw.perflevel0.physicalcpu") or sysctl_int("hw.physicalcpu") or M_LOGICAL
    worker_counts = sorted({perf_cores, M_LOGICAL} - {1})
    print(f"StudentID={STUDENT_ID} | logical processors M={M_LOGICAL} | performance cores={perf_cores} | "
          f"repeats={REPEATS}", flush=True)

    rows = []
    for name, worker, arg, warm_arg, description in WORKLOADS:
        t1, t1_trials = bench(worker, arg, warm_arg, 1)
        print(f"{name:<24} 1 worker : {t1:.3f} s  (trials {', '.join(f'{t:.3f}' for t in t1_trials)})", flush=True)
        for procs in worker_counts:
            tm, tm_trials = bench(worker, arg, warm_arg, procs)
            speedup = (t1 * procs) / tm
            rows.append({"Workload": name, "SingleProcessTime": t1, "MultiProcessTime": tm, "Speedup": speedup,
                         "Workers": procs, "EfficiencyPct": speedup / procs * 100.0, "Description": description})
            print(f"{name:<24} {procs:>2} workers: {tm:.3f} s  (trials {', '.join(f'{t:.3f}' for t in tm_trials)})"
                  f" -> speedup {speedup:.2f}x of ideal {procs}, efficiency {speedup / procs * 100:.0f}%", flush=True)

    fields = ["Workload", "SingleProcessTime", "MultiProcessTime", "Speedup", "Workers", "EfficiencyPct", "Description"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
