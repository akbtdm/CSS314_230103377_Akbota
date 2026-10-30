"""Experiment A: the stubborn serial tail.

Compute work is split cleanly across a process pool, but the master sleeps for a
fixed 0.40 s before dispatching. That fixed cost is a pure Amdahl serial fraction,
so the Karp-Flatt metric should stay flat as workers are added.

Usage:
    python3 exp_a_serial.py [--repeats 3] [--cores 1 2 4]
Writes:
    results/exp_a_serial.json
"""
import argparse
import math
import time
from multiprocessing import Pool

from kf_report import DEFAULT_CORES, DEFAULT_REPEATS, benchmark, print_rows, row_for, save, summarize

SERIAL_PAUSE_S = 0.40


def heavy_math(n):
    return sum(math.sin(i) * math.cos(i) for i in range(1, n))


def run_experiment(workers, total_chunks=16, chunk_size=2_000_000):
    t0 = time.perf_counter()
    time.sleep(SERIAL_PAUSE_S)  # Intentional sequential bottleneck
    with Pool(processes=workers) as pool:
        pool.map(heavy_math, [chunk_size] * total_chunks)
    return time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser(description="Experiment A: fixed serial tail, flat Karp-Flatt metric.")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--cores", type=int, nargs="+", default=list(DEFAULT_CORES))
    args = parser.parse_args()

    all_times = benchmark(run_experiment, args.cores, args.repeats, "exp A")
    rows = summarize(all_times, args.cores)
    print("Experiment A (serial tail)")
    print_rows(rows)

    e2, e4 = row_for(rows, 2)["karp_flatt_e"], row_for(rows, 4)["karp_flatt_e"]
    ratio = e4 / e2
    nominal_serial = SERIAL_PAUSE_S / row_for(rows, 1)["T_s"]
    print(f"e(4)/e(2) = {ratio:.2f}  (rubric window 0.80-1.25) | nominal serial fraction 0.40 s / T1 = {nominal_serial:.3f}")

    save("exp_a_serial", {
        "repeats": args.repeats,
        "cores": args.cores,
        "chunks": 16, "chunk_size": 2_000_000, "serial_pause_s": SERIAL_PAUSE_S,
        "rows": rows,
        "all_times": {str(p): t for p, t in all_times.items()},
        "e_ratio_4_over_2": ratio,
        "nominal_serial_fraction": nominal_serial,
        "rubric_flat_e": 0.80 <= ratio <= 1.25,
        "diagnosis": "Constant e(p): fixed Amdahl serialization (the 0.40 s pause plus pool start-up).",
    })


if __name__ == "__main__":
    main()
