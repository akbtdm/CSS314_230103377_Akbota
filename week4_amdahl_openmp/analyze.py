"""Build Tables 1-3 and speedup_plot.png from the measured results.csv.

Student: Akbota Kutushova, ID 230103377.
Usage: python3 analyze.py   (reads results.csv, writes results/tables.md and speedup_plot.png)
Every number comes from results.csv; this script only does arithmetic on it.
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
N_WORK = 10_000_000 + 3377 * 1000
PHYSICAL_CORES = 10


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def amdahl(p, k):
    return 1.0 / ((1.0 - p) + p / k)


def fit_p_least_squares(points):
    """Grid search p in [0, 1] minimising squared error of S(k) (supplementary fit)."""
    best = min(
        (sum((amdahl(i / 100000, k) - s) ** 2 for k, s in points), i / 100000)
        for i in range(100001)
    )
    return best[1]


def table1(rows, t_seq):
    scale = [r for r in rows if r["experiment"] == "scale"]
    t = {int(r["threads"]): float(r["avg_run2_run3_s"]) for r in scale}
    s2 = t_seq / t[2]
    p = 2.0 * (1.0 - 1.0 / s2)
    lines = [
        "| Threads (k) | Run 1 cold (s) | Run 2 (s) | Run 3 (s) | Avg T_k (s) | S_emp(k) | S_theo(k) | Delta(k) | thread finish spread (s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    series = []
    for r in scale:
        k = int(r["threads"])
        s_emp = t_seq / float(r["avg_run2_run3_s"])
        s_theo = amdahl(p, k)
        series.append((k, s_emp, s_theo))
        lines.append(
            f"| {k} | {float(r['run1_cold_s']):.4f} | {float(r['run2_s']):.4f} | "
            f"{float(r['run3_s']):.4f} | {float(r['avg_run2_run3_s']):.4f} | {s_emp:.3f} | "
            f"{s_theo:.3f} | {s_theo - s_emp:+.3f} | {float(r['thread_finish_spread_s']):.4f} |"
        )
    p_fit = fit_p_least_squares([(k, s) for k, s, _ in series if k <= PHYSICAL_CORES])
    return lines, series, s2, p, p_fit


def table2(rows):
    fs = [r for r in rows if r["experiment"] == "false"]
    t_best = min(float(r["avg_run2_run3_s"]) for r in fs if r["variant"] == "reduction")
    lines = [
        "| Implementation variant | Threads | Run 1 cold (s) | Run 2 (s) | Run 3 (s) | Execution time T (s) | Throughput (iter/s) | Penalty ratio T / T_reduction | hits (>100 steps) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in fs:
        t = float(r["avg_run2_run3_s"])
        lines.append(
            f"| {r['variant']} | {r['threads']} | {float(r['run1_cold_s']):.4f} | "
            f"{float(r['run2_s']):.4f} | {float(r['run3_s']):.4f} | {t:.4f} | "
            f"{N_WORK / t:,.0f} | {t / t_best:.3f}x | {r['hits_gt100']} |"
        )
    return lines


def table3(rows):
    sc = [r for r in rows if r["experiment"] == "sched"]
    lines = [
        "| Schedule | Threads | Run 1 cold (s) | Run 2 (s) | Run 3 (s) | Execution time (s) | thread finish spread (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sc:
        lines.append(
            f"| {r['variant']} | {r['threads']} | {float(r['run1_cold_s']):.4f} | "
            f"{float(r['run2_s']):.4f} | {float(r['run3_s']):.4f} | "
            f"{float(r['avg_run2_run3_s']):.4f} | {float(r['thread_finish_spread_s']):.4f} |"
        )
    return lines


def plot(series, p, path):
    ks = [k for k, _, _ in series]
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=200)
    ax.plot(ks, ks, "--", color="0.55", label="Linear ideal S = k")
    ax.plot(ks, [s for _, _, s in series], "o-", color="#1f77b4",
            label=f"Amdahl S_theo(k), p = {p:.4f} (from k = 2)")
    ax.plot(ks, [s for _, s, _ in series], "s-", color="#d62728", label="Measured S_emp(k) = T_seq / T_k")
    ax.axvline(PHYSICAL_CORES, color="0.3", lw=0.8, ls=":")
    ax.text(PHYSICAL_CORES + 0.2, 1.0, "10 physical cores\n(4 P + 6 E, no SMT)", fontsize=8)
    for k, s_emp, s_theo in series:
        ax.vlines(k, s_emp, s_theo, color="#d62728", alpha=0.3, lw=3)
    ax.set_xlabel("OpenMP threads k")
    ax.set_ylabel("Speedup vs sequential baseline")
    ax.set_title(f"Collatz N = {N_WORK:,}, Apple M4, schedule(static) — Reality Gap (shaded)")
    ax.set_xticks(ks)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)


def main():
    rows = load_rows(HERE / "results.csv")
    seq = next(r for r in rows if r["experiment"] == "seq")
    t_seq = float(seq["avg_run2_run3_s"])
    t1, series, s2, p, p_fit = table1(rows, t_seq)
    out = [
        f"N = {N_WORK:,}; checksum (sum of stopping times mod 1e9+7) = {seq['checksum']}; "
        f"max stopping time = {seq['max_steps']}",
        "",
        f"Sequential baseline T_seq: run1 (cold, discarded) {float(seq['run1_cold_s']):.4f} s, "
        f"run2 {float(seq['run2_s']):.4f} s, run3 {float(seq['run3_s']):.4f} s, "
        f"T_seq = {t_seq:.4f} s",
        "",
        f"S_emp(2) = {s2:.4f}; p = 2 * (1 - 1/S_emp(2)) = {p:.4f}; serial fraction 1 - p = {1 - p:.4f}",
        f"Supplementary least-squares fit of p over k = 1..{PHYSICAL_CORES}: p_fit = {p_fit:.4f}",
        "",
        "## Table 1",
        *t1,
        "",
        "## Table 2",
        *table2(rows),
        "",
        "## Table 3",
        *table3(rows),
    ]
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "tables.md").write_text("\n".join(out) + "\n")
    plot(series, p, HERE / "speedup_plot.png")
    print("\n".join(out))


if __name__ == "__main__":
    main()
