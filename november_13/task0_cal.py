"""Phase 0: CPU/cache topology audit and timer noise-floor calibration (task0_cal.py)."""
import multiprocessing as mp
import sys
import time

from harness import run_benchmark, emit_verify_tag
from labutil import (build_parser, resolve_student_id, results_dir, save_json,
                     hardware_info)

CLOCK_READS = 2_000_000


def measure_clock(n_reads: int):
    """Average perf_counter_ns() call latency and the smallest non-zero tick observed."""
    perf = time.perf_counter_ns
    first = prev = perf()
    min_tick = None
    for _ in range(n_reads):
        now = perf()
        delta = now - prev
        if delta > 0 and (min_tick is None or delta < min_tick):
            min_tick = delta
        prev = now
    avg_ns = (prev - first) / n_reads
    return avg_ns, min_tick or 0


def empty_fn() -> None:
    return None


def main() -> None:
    args = build_parser(__doc__).parse_args()
    sid = resolve_student_id(args.id)
    n_reads = CLOCK_READS // 100 if args.quick else CLOCK_READS

    hw = hardware_info()
    print("=== PHASE 0: hardware topology audit ===")
    for key, value in hw.items():
        print(f"  {key:20s} {value}")
    print(f"  {'mp.cpu_count()':20s} {mp.cpu_count()}")
    print(f"  {'start method':20s} {mp.get_start_method()}")
    print(f"  {'sys.version':20s} {sys.version.split()[0]}")

    print(f"\n=== timer calibration: {n_reads} perf_counter_ns() reads ===")
    avg_ns, min_tick = measure_clock(n_reads)
    print(f"  avg call latency     {avg_ns:.3f} ns")
    print(f"  min non-zero tick    {min_tick} ns")
    noise = run_benchmark(empty_fn, warmup_runs=3, timed_runs=9)
    print("  noise floor (empty fn through run_benchmark): "
          f"min={noise.min_ms * 1e3:.3f} med={noise.median_ms * 1e3:.3f} "
          f"iqr={noise.iqr_ms * 1e3:.3f} max={noise.max_ms * 1e3:.3f} us")

    tag = emit_verify_tag("PHASE0", sid, avg_ns, float(min_tick))
    print("\n" + tag)
    save_json(results_dir(args.quick) / "phase0.json", {
        "student_id": sid,
        "hardware": hw,
        "clock_reads": n_reads,
        "avg_call_latency_ns": avg_ns,
        "min_tick_ns": min_tick,
        "noise_floor_ms": noise.__dict__,
        "verify": tag,
        "hash_inputs": {"primary": avg_ns, "secondary": float(min_tick)},
    })


if __name__ == "__main__":
    main()
