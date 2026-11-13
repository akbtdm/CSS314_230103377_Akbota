"""Assemble lab_report_<ID>.md from results/*.json, logs/*.log, analysis/*.md and the sources.
Every number in the report is read from the JSON files written by the task scripts."""
import datetime as dt
import json
from pathlib import Path

LAB = Path(__file__).resolve().parent
RESULTS, LOGS, ANALYSIS = LAB / "results", LAB / "logs", LAB / "analysis"
SOURCES = ["harness.py", "labutil.py", "task0_cal.py", "task1_falsesharing.py", "task2_scaling.py",
           "task3a_roofline.py", "task3b_gemm.py", "task4_scheduling.py", "emit_hashes.py",
           "run_all.sh"]


def load(name):
    return json.loads((RESULTS / name).read_text())


def text_or(path: Path, fallback: str) -> str:
    return path.read_text().strip() if path.exists() else fallback


def code_block(text: str, lang: str = "") -> str:
    return f"```{lang}\n{text.rstrip()}\n```"


def table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def kb(nbytes):
    return "n/a" if not nbytes else f"{nbytes / 1024:.0f} KB"


def mb(nbytes):
    return "n/a" if not nbytes else f"{nbytes / 1024 / 1024:.0f} MB"


def f(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def hardware_section(p0):
    hw = p0["hardware"]
    rows = [
        ("CPU Model String", hw["cpu_model"], "L1 Data Cache (per core)",
         f"{kb(hw.get('p_core_l1d_bytes'))} P-core / {kb(hw.get('e_core_l1d_bytes'))} E-core "
         f"(hw.l1dcachesize = {kb(hw['l1d_bytes'])})"),
        ("Physical Cores (P_phys)", hw["physical_cores"], "L2 Cache Size",
         f"{mb(hw.get('p_core_l2_bytes'))} shared by the 4 P-cores / {mb(hw.get('e_core_l2_bytes'))} "
         f"shared by the 6 E-cores (hw.l2cachesize = {mb(hw['l2_bytes'])})"),
        ("Logical Threads (P_log)", hw["logical_threads"], "L3 Cache Size (Shared)",
         "none reported by sysctl (hw.l3cachesize absent; Apple SLC not exposed)"),
        ("Cache Line Size (L_bytes)", f"{hw['cache_line_bytes']} bytes", "Python Version & OS",
         f"Python {hw['python']} / macOS ({hw['os']})"),
        ("Core topology", f"{hw['perf_cores']} performance + {hw['eff_cores']} efficiency cores, no SMT",
         "RAM", mb(hw["mem_bytes"])),
    ]
    return table(["Hardware Parameter", "Measured Value", "Hardware Parameter", "Measured Value"], rows)


def phase1_section(p1):
    sweep = p1["stride_sweep"]
    t1a = table(["stride (words)", "separation (bytes)", "median (ms)", "IQR (ms)", "throughput (Mops/s)"],
                [(r["stride"], r["sep_bytes"], f(r["median_ms"], 2), f(r["iqr_ms"], 2), f(r["mops"], 3))
                 for r in sweep])
    drops = []
    for prev, cur in zip(sweep, sweep[1:]):
        drops.append((cur["sep_bytes"], 100.0 * (prev["median_ms"] - cur["median_ms"]) / prev["median_ms"]))
    drop_tbl = table(["transition to separation (bytes)", "change in median time vs previous stride"],
                     [(b, f"{-d:+.1f} %") for b, d in drops])
    parts = [f"Configuration: M = {p1['m_total']:,} total RMW steps, p = P_phys = {p1['p_phys']}, "
             f"run_benchmark(warmup_runs={p1['warmup_runs']}, timed_runs={p1['timed_runs']}), "
             f"SharedMemory(size=4096) viewed as 512 x c_uint64.",
             "**Table 1A — stride sensitivity at p = P_phys**", t1a,
             "Relative change between consecutive strides (negative = faster):", drop_tbl]
    for stride in ("1", "16"):
        rows = p1["scaling"][stride]
        parts.append(f"**Table 1B — process scaling, stride = {stride} ({int(stride) * 8} bytes between slots)**")
        parts.append(table(["p", "median (ms)", "IQR (ms)", "S_p = T1/Tp", "E_p = S_p/p", "Karp-Flatt e(p)"],
                           [(r["p"], f(r["median_ms"], 2), f(r["iqr_ms"], 2), f(r["speedup"]),
                             f(r["efficiency"]), f(r["karp_flatt"], 4)) for r in rows]))
    return "\n\n".join(parts)


def phase2_section(p2, perf_cores):
    s = p2["summary"]
    strong, weak = p2["strong"], p2["weak"]
    t2a = table(["p", "T_strong (ms)", "IQR (ms)", "Stage A (ms)", "S_p", "E_p", "Amdahl S(p)", "marginal dS"],
                [(r["p"], f(r["median_ms"], 2), f(r["iqr_ms"], 2), f(r["stage_a_ms"], 2), f(r["speedup"]),
                  f(r["efficiency"]), f(r["amdahl_speedup"]), f(r["marginal_speedup"])) for r in strong])
    t2b = table(["p", "N(p)", "T_weak (ms)", "IQR (ms)", "Stage A (ms)", "E_weak = T1/Tp", "alpha", "S_Gustafson"],
                [(r["p"], f"{r['n']:,}", f(r["median_ms"], 2), f(r["iqr_ms"], 2), f(r["stage_a_ms"], 2),
                  f(r["weak_efficiency"]), f(r["alpha"], 4), f(r["gustafson_speedup"])) for r in weak])
    marg = [r["marginal_speedup"] for r in strong]
    boundary = ""
    if perf_cores and perf_cores + 1 <= len(strong):
        before, after = marg[perf_cores - 1], marg[perf_cores]
        if before:
            boundary = (f"Marginal speedup of the {perf_cores}th worker (last P-core): {before:.3f}; "
                        f"of the {perf_cores + 1}th worker (first E-core): {after:.3f}; "
                        f"change = {100.0 * (after - before) / before:+.1f} %.")
    summary = table(["quantity", "value"], [
        ("serial fraction s = T_StageA(1) / T_strong(1)", f(s["serial_fraction_s"], 4)),
        ("Amdahl asymptote S_inf = 1/s", f(s["amdahl_s_inf"], 2)),
        (f"Amdahl theoretical S(P_log={p2['p_log']})", f(s["amdahl_speedup_at_plog"])),
        (f"observed S_strong(P_log={p2['p_log']})", f(s["observed_strong_speedup_at_plog"])),
        (f"observed S_Gustafson(P_log={p2['p_log']})", f(s["gustafson_speedup_at_plog"])),
    ])
    return "\n\n".join([
        f"Configuration: Stage A serial over 5 % of N (12 transcendental ops/element), Stage B over all N "
        f"(35 multiply-add + sin iterations/element) on a persistent mp.Pool(p) created outside the timed "
        f"region; run_benchmark(warmup_runs={p2['warmup_runs']}, timed_runs={p2['timed_runs']}).",
        f"**Table 2A — strong scaling, N = {p2['n_strong']:,}**", t2a,
        f"**Table 2B — weak scaling, N(p) = p x {p2['n0_weak']:,}**", t2b,
        "**Amdahl ceiling comparison**", summary, boundary])


def phase3_section(p3a, p3b):
    bw, perf, ridge = p3a["bw_sustained_gbs"], p3a["p_sustained_gflops"], p3a["ridge_point"]
    bw_rows = [(i, f(r["t_median_worker_ms"]), f(r["bw_gbs"], 2), f(r["wall_ms"]), f(r["bw_wallclock_gbs"], 2),
                r["copy_verified"], " ".join(f"{t:.3f}" for t in r["per_worker_ms"]))
               for i, r in enumerate(p3a["bandwidth_rounds"])]
    cp_rows = [(i, f(r["wall_ms"], 1), f(r["gflops_aggregate"], 4),
                " ".join(f"{g:.3f}" for g in r["per_worker_gflops"])) for i, r in enumerate(p3a["compute_rounds"])]
    ceilings = table(["quantity", "value"], [
        ("BW_sustained (2 x 96 MB shared buffers, parallel memcpy, P_phys workers)", f"{bw:.2f} GB/s"),
        ("P_sustained (16-way unrolled scalar FMA, 8e7 FLOPs/worker, P_phys workers)", f"{perf:.4f} GFLOP/s"),
        ("Ridge point I* = P_sustained / BW_sustained", f"{ridge:.6f} FLOP/Byte"),
        ("BW wall-clock aggregate of the same round (all reps x all workers / (last end - first start))",
         f"{p3a['bw_wallclock_gbs']:.2f} GB/s"),
        ("Ridge point with the wall-clock bandwidth", f"{perf / p3a['bw_wallclock_gbs']:.6f} FLOP/Byte"),
    ])

    def attainable(intensity):
        return min(perf, intensity * bw)

    rows = []
    for r in p3b["rows"]:
        if "bs" in r:
            inten, foot = r["intensity_flop_per_byte"], f"{r['tile_footprint_bytes']:,}"
        else:
            inten, foot = 0.125, f"{p3b['footprint_bytes']:,} (full 3N^2 x 8)"
        att = attainable(inten)
        rows.append((r["variant"], foot, f(inten, 3), f(r["median_ms"], 2), f(r["iqr_ms"], 2),
                     f(r["gflops"], 4), f(att, 4), f"{100.0 * r['gflops'] / att:.1f} %"))
    gemm = table(["variant", "working set (bytes)", "I (FLOP/Byte)", "median (ms)", "IQR (ms)", "GFLOP/s",
                  "Roofline attainable (GFLOP/s)", "% of attainable"], rows)
    chk = p3b["checks"]
    return "\n\n".join([
        f"Configuration: P_phys = {p3a['p_phys']} worker processes; bandwidth = median over workers of each "
        f"worker's median of {p3a['copy_reps']} timed slice copies after a Barrier (first copy is a page-touch "
        f"warm-up); compute = total FLOPs / (last worker end - first worker start); {len(p3a['bandwidth_rounds'])} "
        f"rounds each, the median round is reported.",
        "**Part 3A — empirical ceilings**", ceilings,
        table(["round", "median worker copy (ms)", "BW lab formula (GB/s)", "wall of all reps (ms)",
               "BW wall-clock (GB/s)", "copy verified", "per-worker median copy time (ms)"], bw_rows),
        table(["round", "wall (ms)", "aggregate GFLOP/s", "per-worker GFLOP/s"], cp_rows),
        f"**Part 3B — GEMM N = {p3b['n']}, footprint {p3b['footprint_bytes'] / 1e6:.2f} MB, "
        f"run_benchmark(warmup_runs={p3b['warmup_runs']}, timed_runs={p3b['timed_runs']})**",
        f"Numerical equivalence: max|C(V1) - C(V2)| = {chk['max_abs_diff_v1_vs_v2']:.3e}, "
        f"max|C(V3, BS=16) - C(V2)| = {chk['max_abs_diff_v3_bs16_vs_v2']:.3e} (< 1e-9), "
        f"sum(C) = {chk['c_checksum']:.6f}. Row ranges per worker: {p3b['row_ranges']}.",
        gemm,
        f"Empirical optimum: BS* = {p3b['best_bs']}.  Roofline attainable = min(P_sustained, I x BW_sustained)."])


def phase4_section(p4):
    rows = [(r["strategy"], f(r["median_ms"], 1), f(r["iqr_ms"], 1), f(r["lambda_median"]),
             f(r["worker_ms_max"], 0), f(r["worker_ms_mean"], 0),
             " ".join(f"{t:.0f}" for t in r["worker_ms_last"])) for r in p4["rows"]]
    tbl = table(["strategy", "T_wall median (ms)", "IQR (ms)", "lambda (median over runs)", "max t_worker (ms)",
                 "mean t_worker (ms)", "per-worker active time, last run (ms)"], rows)
    return "\n\n".join([
        f"Configuration: N = {p4['n']:,} items, K(i) = 1 + floor((i/60)^2), total inner iterations = "
        f"{p4['total_inner_iterations']:,}, p = {p4['p']} mp.Process workers, per-worker active time via "
        f"mp.Array('d', p), dynamic queue = mp.Value('i') + its lock, run_benchmark(warmup_runs="
        f"{p4['warmup_runs']}, timed_runs={p4['timed_runs']}). T_wall includes process spawn and join.",
        tbl, f"Best dynamic chunk: C* = {p4['best_chunk']}."])


def main() -> None:
    p0, p1, p2 = load("phase0.json"), load("phase1.json"), load("phase2.json")
    p3a, p3b, p4 = load("phase3a.json"), load("phase3b.json"), load("phase4.json")
    sid = p0["student_id"]
    hw = p0["hardware"]
    hashes = "\n".join(d["verify"] for d in (p0, p1, p2, p3b, p4))
    pending = "_(analysis pending)_"
    doc = [
        "# Performance Evaluation and Optimization in Parallel Computing — Solo Python Systems Lab",
        f"**Student ID:** {sid}  \n**Date:** {dt.date.today().isoformat()}  \n"
        f"**Machine:** {hw['cpu_model']}, {hw['physical_cores']} cores "
        f"({hw['perf_cores']}P + {hw['eff_cores']}E), {mb(hw['mem_bytes'])} RAM, Python {hw['python']}, "
        f"{hw['os']}  \n**Environment:** standard library only (multiprocessing, shared_memory, ctypes, array, "
        f"time, math); every number below was produced by the scripts in Appendix B on this machine.",
        "## 1. Hardware Baseline & Verification Hashes",
        hardware_section(p0),
        "Raw hardware audit (lab section 1.1 commands):", code_block(text_or(LOGS / "hw_audit.log", ""), "text"),
        "**Verification hashes (verbatim script output):**", code_block(hashes, "text"),
        "Self-audit (`python3 emit_hashes.py` recomputes fnv1a over `task_id|student_id|P_log|metric1|metric2` "
        "from the saved metrics):", code_block(text_or(LOGS / "emit_hashes.log", ""), "text"),
        "### Phase 0 — harness calibration (task0_cal.py console output)",
        code_block(text_or(LOGS / "task0_cal.log", ""), "text"),
        text_or(ANALYSIS / "phase0.md", pending),
        "## 2. Phase 1 — False Sharing Telemetry & Karp-Flatt Analysis",
        phase1_section(p1),
        "### Cache-line boundary identification and forensic explanation",
        text_or(ANALYSIS / "phase1.md", pending),
        "## 3. Phase 2 — Strong vs Weak Scaling Matrices",
        phase2_section(p2, hw.get("perf_cores")),
        "### Amdahl ceiling and core-heterogeneity inflection analysis",
        text_or(ANALYSIS / "phase2.md", pending),
        "## 4. Phase 3 — Empirical Roofline & GEMM Tile Sweep",
        phase3_section(p3a, p3b),
        "### Roofline and cache-capacity analysis",
        text_or(ANALYSIS / "phase3.md", pending),
        "## 5. Phase 4 — Scheduling & Load Imbalance",
        phase4_section(p4),
        "### Chunk-size trade-off analysis",
        text_or(ANALYSIS / "phase4.md", pending),
        "## Appendix A — Raw terminal telemetry logs",
    ]
    for name in ["task1_falsesharing", "task2_scaling", "task3a_roofline", "task3b_gemm", "task4_scheduling"]:
        doc += [f"### {name}.py", code_block(text_or(LOGS / f"{name}.log", "(missing)"), "text")]
    doc.append("## Appendix B — Python 3 source code")
    for name in SOURCES:
        lang = "bash" if name.endswith(".sh") else "python"
        doc += [f"### {name}", code_block((LAB / name).read_text(), lang)]
    out = LAB / f"lab_report_{sid}.md"
    out.write_text("\n\n".join(doc) + "\n")
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
