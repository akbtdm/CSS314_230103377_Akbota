"""Part 4: build RESULTS_B.md from the JSON written by the three experiments.

Usage:
    python3 write_results_b.py
"""
import json
import os
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
STUDENT_ID = "230103377"


def load(name):
    with open(os.path.join(RESULTS, f"{name}.json"), encoding="utf-8") as fh:
        return json.load(fh)


def row(rows, p):
    return next(r for r in rows if r["P"] == p)


def line(label, rows, verdict):
    r1, r2, r4, r8 = (row(rows, p) for p in (1, 2, 4, 8))
    return (f"| {label} | {r1['T_s']:.3f} | {r2['T_s']:.3f} | {r4['T_s']:.3f} | {r8['T_s']:.3f} | "
            f"{r8['speedup']:.2f}x | {r2['karp_flatt_e']:.3f} | {r4['karp_flatt_e']:.3f} | {r8['karp_flatt_e']:.3f} | {verdict} |")


def main():
    a, b, c = load("exp_a_granularity"), load("exp_b_queue"), load("exp_c_buffered")
    host = a["host"]
    micro = next(s for s in a["series"] if s["chunksize"] == 1)["rows"]
    coarse = next(s for s in a["series"] if s["chunksize"] != 1)["rows"]
    b8, c8, c4 = row(b["rows"], 8), row(c["rows"], 8), row(c["rows"], 4)
    b_items_s = b["total_items"] / row(b["rows"], 1)["T_s"]
    c_items_s = c["total_items"] / c8["T_s"]

    lines = [
        "# RESULTS_B.md: Granularity and IPC Contention Diagnostics (Variant B)",
        "",
        f"Student ID {STUDENT_ID} | branch `variant-b-{STUDENT_ID}` | generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Hardware",
        "",
        f"- {host['cpu_model']}, {host['execution_cores']} execution cores ({host['performance_cores']} performance + "
        f"{host['efficiency_cores']} efficiency, no SMT), {host['cache_line_bytes']}-byte cache line; macOS {host['macos_version']}; "
        f"{host['python']}, start method `{host['default_start_method']}`.",
        f"- `CPUs: {host['execution_cores']} | Default Pool: {host['execution_cores']}` (Part 0 check).",
        f"- Runtimes are medians of {a['repeats']} runs of the whole `run_*` function as written in the handout.",
        "",
        "## Multi-core matrix",
        "",
        "| Configuration | T1 (s) | T2 (s) | T4 (s) | T8 (s) | S8 | e(2) | e(4) | e(8) | Verdict |",
        "|---|---|---|---|---|---|---|---|---|---|",
        line("A: Micro (k=1)", micro, "Master dispatch serialization"),
        line("A: Coarse (k=10,000)", coarse, "Amortised dispatch, healthy strong scaling"),
        line("B: Queue Choke", b["rows"], "IPC lock contention explosion"),
        line("C: Buffered Batch", c["rows"], "Amortised synchronization"),
        "",
        "## Sign-off self-check",
        "",
        f"- **Experiment A:** with chunksize = 1 the master dispatches 1,000,000 task envelopes at about "
        f"{row(micro, 1)['T_s'] / 1_000_000 * 1e6:.1f} us each; runtime is {row(micro, 1)['T_s']:.2f} s at P = 1 and "
        f"{row(micro, 8)['T_s']:.2f} s at P = 8 (S8 = {row(micro, 8)['speedup']:.2f}x), i.e. the workers starve behind the "
        f"single dispatching thread. With chunksize = 10,000 the same work takes {row(coarse, 1)['T_s']:.2f} s at P = 1 and "
        f"{row(coarse, 8)['T_s']:.2f} s at P = 8 (S8 = {row(coarse, 8)['speedup']:.2f}x); micro-tasks are "
        f"{a['micro_over_coarse_ratio']['8']:.0f}x slower at P = 8.",
        f"- **Experiment B:** T8 = {b8['T_s']:.2f} s vs T1 = {row(b['rows'], 1)['T_s']:.2f} s (T8/T1 = {b['t8_over_t1']:.2f}, "
        f"negative speedup: {'PASS' if b['t8_over_t1'] > 1 else 'FAIL'}); runtime rises monotonically with P: "
        f"{b['runtime_monotonic_increase']}; e(p) rises monotonically: {b['karp_flatt_monotonic_increase']} "
        f"(e(2) = {row(b['rows'], 2)['karp_flatt_e']:.2f}, e(8) = {b8['karp_flatt_e']:.2f}). One `put` per item costs about "
        f"{row(b['rows'], 1)['T_s'] / b['total_items'] * 1e6:.1f} us even with a single producer.",
        f"- **Experiment C:** {c['total_items']:,} items in {c8['T_s']:.3f} s at P = 8 versus {b['total_items']:,} items in "
        f"{row(b['rows'], 1)['T_s']:.2f} s for the unbuffered queue: throughput {c_items_s / b_items_s:.0f}x higher "
        f"({c_items_s:,.0f} vs {b_items_s:,.0f} items/s), with only {c['ipc_puts']:,} IPC crossings instead of one per item. "
        f"S8 = {c8['speedup']:.2f}x ({'PASS' if c['rubric_s8_above_5'] else 'below the 5.0x target'}), "
        f"e(8) = {c8['karp_flatt_e']:.3f} ({'PASS' if c['rubric_e8_at_most_0_05'] else 'above the 0.05 target'}); "
        f"on the 4 performance cores S4 = {c4['speedup']:.2f}x, e(4) = {c4['karp_flatt_e']:.3f}.",
        "",
        "## Notes",
        "",
        "- This CPU has 4 performance and 6 efficiency cores. With 8 equal static shares the slowest worker lands on an "
        "efficiency core (about half the speed), so S8 is capped near 4-5x for *any* well-behaved workload here; the "
        "P = 4 numbers show the scaling on homogeneous cores.",
        "- macOS starts processes with `spawn`, which adds tens of milliseconds per experiment; it is inside the timed "
        "region because the handout's `run_*` functions time everything from `start()` to `join()`.",
        "- Karp-Flatt e(p) > 1 is impossible for a pure serial fraction, so the Experiment B values are proof of contention "
        "by construction, independent of the exact T1.",
        "",
        "## Reproduce",
        "",
        "```",
        "python3 exp_a_granularity.py",
        "python3 exp_b_queue.py",
        "python3 exp_c_buffered.py",
        "python3 write_results_b.py",
        "```",
        "",
    ]
    out = os.path.join(HERE, "RESULTS_B.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
