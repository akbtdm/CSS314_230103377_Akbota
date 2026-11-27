"""Builds every figure in results/ from the raw CSV files (no numbers typed in)."""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
DPI = 200


def read(name):
    with open(os.path.join(RES, name), newline="") as fh:
        return list(csv.DictReader(fh))


def save(fig, name):
    path = os.path.join(RES, name)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"wrote {path}")


def lab1():
    rows = read("lab1_oversubscription.csv")
    p = [int(r["P"]) for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for key, lab in (("executor_create_join_us", "ThreadPoolExecutor create+join"),
                     ("thread_create_join_us", "threading.Thread start+join"),
                     ("numba_region_steady_us", "Numba OpenMP region (pool reused)")):
        a1.plot(p, [float(r[key]) for r in rows], "o-", label=lab)
    a1.set_xscale("log", base=2)
    a1.set_yscale("log")
    a1.set_xlabel("team size P")
    a1.set_ylabel("time per fork+join (us, log)")
    a1.set_title("Task 1.2: cost of forking and joining a team")
    a1.axvline(10, color="grey", ls=":", lw=1)
    a1.legend(fontsize=8)
    a2.plot(p, [float(r["numba_fixed_work_s"]) for r in rows], "o-",
            label="Numba: 2e9 sqrt split over P")
    a2.plot(p, [float(r["py_threads_fixed_work_s"]) for r in rows], "s-",
            label="Python threads: 4e6 sqrt split over P")
    a2.axvline(10, color="grey", ls=":", lw=1, label="10 physical cores")
    a2.set_xscale("log", base=2)
    a2.set_xlabel("team size P")
    a2.set_ylabel("wall time (s)")
    a2.set_title("Fixed total work vs team size")
    a2.legend(fontsize=8)
    save(fig, "lab1_oversubscription.png")

    rows = read("lab1_cpu_saturation_summary.csv")
    cores = [k for k in rows[0] if k.startswith("cpu")]
    x = np.arange(len(cores))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    w = 0.27
    for i, r in enumerate(rows):
        ax.bar(x + (i - 1) * w, [float(r[c]) for c in cores], w,
               label=f"{r['variant']} (sum {float(r['sum_pct']):.0f}%)")
    ax.set_xticks(x, [c.replace("_mean_pct", "") for c in cores])
    ax.set_ylabel("mean utilisation during run (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Task 1.3: per-core CPU utilisation (psutil, 0.1 s samples)")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "lab1_cpu_saturation.png")


def lab2():
    rows = read("lab2_scaling.csv")
    p = [int(r["P"]) for r in rows]
    s = [float(r["speedup"]) for r in rows]
    e = [float(r["efficiency"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(p, s, "o-", label="measured speedup S(P)")
    ax.plot(p, p, "k--", label="linear ideal S = P")
    ax.axvline(10, color="grey", ls=":", lw=1)
    ax.set_xlabel("threads P")
    ax.set_ylabel("speedup S(P) = T(1)/T(P)")
    ax.set_xticks(p)
    ax2 = ax.twinx()
    ax2.plot(p, e, "s-", color="tab:red", label="efficiency E(P)")
    ax2.set_ylabel("efficiency E(P) = S/P", color="tab:red")
    ax2.set_ylim(0, 1.15)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    ax.set_title("Task 2.4: Pi reduction strong scaling (N = 1e8, mean of 5)")
    save(fig, "lab2_speedup_efficiency.png")


def lab3():
    rows = read("lab3_schedule_matrix.csv")
    ps, cs = [2, 4, 8, 16], [1, 16, 64, 256]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    vmax = max(float(r["mean_s"]) for r in rows if r["policy"] != "block")
    for ax, pol in zip(axes, ("static", "dynamic", "guided")):
        m = np.zeros((len(ps), len(cs)))
        for r in rows:
            if r["policy"] == pol:
                m[ps.index(int(r["P"])), cs.index(int(r["chunk"]))] = float(r["mean_s"])
        im = ax.imshow(m, cmap="viridis_r", vmin=0, vmax=vmax)
        for i in range(len(ps)):
            for j in range(len(cs)):
                ax.text(j, i, f"{m[i, j]*1e3:.0f}", ha="center", va="center",
                        color="white" if m[i, j] > vmax * 0.5 else "black", fontsize=9)
        ax.set_xticks(range(len(cs)), cs)
        ax.set_yticks(range(len(ps)), ps)
        ax.set_xlabel("chunk size C")
        ax.set_ylabel("threads P")
        ax.set_title(f"schedule({pol}, C): mean ms")
    fig.colorbar(im, ax=axes, shrink=0.8, label="seconds")
    fig.suptitle("Task 3.3: Mandelbrot 1920x1080, mean of 3 runs")
    path = os.path.join(RES, "lab3_schedule_heatmap.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")

    pt = read("lab3_per_thread_work.csv")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for pol, c, lab in (("block", 0, "static block (prange default)"),
                        ("static", 16, "static, 16"), ("dynamic", 16, "dynamic, 16")):
        r = next(x for x in pt if x["policy"] == pol and x["P"] == "8"
                 and x["chunk"] == str(c) and x["run"] == "1")
        w = np.array([int(v) for v in r["work"].split(";")]) / 1e6
        ax.plot(range(8), w, "o-", label=lab)
    ax.set_xlabel("thread rank")
    ax.set_ylabel("escape iterations (millions)")
    ax.set_title("Task 3.4: work per thread, P = 8")
    ax.legend(fontsize=8)
    save(fig, "lab3_work_per_thread.png")


def lab4():
    rows = read("lab4_false_sharing.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = {"unpadded": "unpadded (stride 8 B)", "pad64": "padded 64 B (manual)",
              "pad128": "padded 128 B (M4 line)", "local_mem": "thread-local buffer",
              "local_reg": "thread-local register"}
    for v, lab in labels.items():
        sel = [r for r in rows if r["variant"] == v]
        ax.plot([int(r["P"]) for r in sel], [float(r["mean_s"]) for r in sel], "o-", label=lab)
    ax.axvline(10, color="grey", ls=":", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16], [1, 2, 4, 8, 16])
    ax.set_xlabel("threads P (each does 1e8 increments)")
    ax.set_ylabel("wall time (s)")
    ax.set_title("Tasks 4.1-4.3: false sharing vs padding")
    ax.legend(fontsize=8)
    save(fig, "lab4_false_sharing.png")


def lab5():
    rows = read("lab5_cutoff_sweep.csv")
    k = [int(r["K"]) for r in rows]
    t = [float(r["mean_s"]) for r in rows]
    dnf = [r["status"] != "ok" for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(k, t, "o-")
    for ki, ti, d in zip(k, t, dnf):
        if d:
            ax.annotate("DNF (stopped)", (ki, ti), textcoords="offset points", xytext=(5, 5))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sequential cutoff K (log scale)")
    ax.set_ylabel("wall time (s, log)")
    ax.set_title("Task 5.2: task merge sort, N = 5e6, 10 workers")
    save(fig, "lab5_cutoff_sweep.png")


def lab6():
    rows = read("lab6_pipeline.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [f"{r['passes']} pass{'es' if r['passes'] != '1' else ''}\nW={r['filter_workers']}"
              for r in rows]
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [float(r["serial_throughput_pkt_s"]) for r in rows], 0.4, label="serial")
    ax.bar(x + 0.2, [float(r["throughput_pkt_s"]) for r in rows], 0.4, label="pipeline")
    ax.set_xticks(x, labels)
    ax.set_yscale("log")
    ax.set_ylabel("throughput (packets/s, log)")
    ax.set_title("Lab 6: pipeline throughput by filter workload")
    ax.legend()
    save(fig, "lab6_throughput.png")


if __name__ == "__main__":
    for f in (lab1, lab2, lab3, lab4, lab5, lab6):
        f()
