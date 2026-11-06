#!/usr/bin/env python3
"""
Task 21 - Parametric Speedup Simulator (Amdahl vs. Gustafson).

Models fixed-size (Amdahl) and scaled-size (Gustafson) speedup for
1 <= P <= 65,536 and 0.001 <= alpha <= 0.20, optionally with a communication
penalty C(P) (Task 12). Exports CSV and an overlay plot.

Examples
--------
  python3 simulator.py                                  # default sweep -> CSV + PNG
  python3 simulator.py --alpha 0.04 --p 16 64 256 1024 4096 16384 --table
  python3 simulator.py --alpha 0.04 --comm-model log --comm-c 0.005 --p 16 256 1024 16384 --table
"""
import argparse
import csv
import math
import os
import sys

P_MIN, P_MAX = 1, 65_536
A_MIN, A_MAX = 0.001, 0.20


def amdahl(p: int, s: float) -> float:
    """Fixed-size speedup: 1 / (s + (1 - s) / P)."""
    return 1.0 / (s + (1.0 - s) / p)


def gustafson(p: int, a: float) -> float:
    """Scaled speedup: P - alpha * (P - 1)."""
    return p - a * (p - 1)


def comm_overhead(p: int, model: str, c: float) -> float:
    """Normalised communication cost C(P) with C(1) = 0 (Task 12)."""
    if model == "none" or p == 1:
        return 0.0
    if model == "log":
        return c * math.log2(p)
    if model == "sqrt":
        return c * (math.sqrt(p) - 1.0)
    if model == "linear":
        return c * (p - 1)
    raise ValueError(f"unknown comm model: {model}")


def gustafson_comm(p: int, a: float, model: str, c: float) -> float:
    """Modified scaled speedup: (P - alpha(P-1)) / (1 + C(P))."""
    return gustafson(p, a) / (1.0 + comm_overhead(p, model, c))


def default_p_values():
    return [2 ** i for i in range(0, 17)]  # 1 .. 65536


def default_alphas():
    return [0.001, 0.01, 0.04, 0.10, 0.20]


def simulate(p_values, alphas, model, c):
    rows = []
    for a in alphas:
        for p in p_values:
            sa, sg = amdahl(p, a), gustafson(p, a)
            sc = gustafson_comm(p, a, model, c)
            rows.append({
                "P": p, "alpha": a,
                "amdahl_S": sa, "amdahl_E": sa / p,
                "gustafson_S": sg, "gustafson_E": sg / p,
                "comm_model": model, "comm_c": c,
                "gustafson_comm_S": sc, "gustafson_comm_E": sc / p,
            })
    return rows


def write_csv(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot(rows, alphas, path, model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    colors = plt.cm.viridis([i / max(1, len(alphas) - 1) for i in range(len(alphas))])
    ps = sorted({r["P"] for r in rows})
    ax1.plot(ps, ps, "k:", lw=1.2, label="Ideal S = P")
    for a, col in zip(alphas, colors):
        sub = [r for r in rows if r["alpha"] == a]
        x = [r["P"] for r in sub]
        ax1.plot(x, [r["gustafson_S"] for r in sub], "-", color=col, lw=2, label=f"Gustafson α={a}")
        ax1.plot(x, [r["amdahl_S"] for r in sub], "--", color=col, lw=1.5, label=f"Amdahl s={a}")
        if model != "none":
            ax1.plot(x, [r["gustafson_comm_S"] for r in sub], "-.", color=col, lw=1, alpha=0.8)
        ax2.plot(x, [r["gustafson_E"] for r in sub], "-", color=col, lw=2)
        ax2.plot(x, [r["amdahl_E"] for r in sub], "--", color=col, lw=1.5)
    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Processors P")
        ax.grid(True, which="both", alpha=0.3)
    ax1.set_yscale("log")
    ax1.set_ylabel("Speedup S(P)")
    ax1.set_title("Strong (Amdahl, dashed) vs. weak (Gustafson, solid) scaling")
    ax2.set_ylabel("Efficiency E(P) = S/P")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Parallel efficiency")
    ax1.legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    if path.endswith(".png"):
        fig.savefig(path[:-4] + ".pdf")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p", type=int, nargs="+", help="core counts (default: powers of 2 from 1 to 65536)")
    ap.add_argument("--alpha", type=float, nargs="+", help="serial fractions (default: 0.001 0.01 0.04 0.10 0.20)")
    ap.add_argument("--comm-model", choices=["none", "log", "sqrt", "linear"], default="none")
    ap.add_argument("--comm-c", type=float, default=0.0, help="cost per communication stage (normalised)")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--csv", default=os.path.join(here, "simulation_results.csv"))
    ap.add_argument("--plot", default=os.path.join(here, "..", "plots", "simulator_overlay.png"))
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--table", action="store_true", help="print a table to stdout")
    args = ap.parse_args(argv)

    p_values = args.p or default_p_values()
    alphas = args.alpha or default_alphas()
    bad_p = [p for p in p_values if not P_MIN <= p <= P_MAX]
    bad_a = [a for a in alphas if not A_MIN <= a <= A_MAX]
    if bad_p or bad_a:
        sys.exit(f"Out of range: P must be in [{P_MIN},{P_MAX}], alpha in [{A_MIN},{A_MAX}]. Got P={bad_p}, alpha={bad_a}")

    rows = simulate(p_values, alphas, args.comm_model, args.comm_c)
    write_csv(rows, args.csv)
    print(f"[+] CSV written: {args.csv} ({len(rows)} rows)")
    if args.table:
        hdr = f"{'alpha':>6} {'P':>6} {'Amdahl S':>10} {'Gustafson S':>12} {'E_G':>9}"
        if args.comm_model != "none":
            hdr += f" {'S_comm':>11} {'E_comm':>8}"
        print(hdr)
        for r in rows:
            line = f"{r['alpha']:>6} {r['P']:>6} {r['amdahl_S']:>10.3f} {r['gustafson_S']:>12.2f} {r['gustafson_E']:>9.6f}"
            if args.comm_model != "none":
                line += f" {r['gustafson_comm_S']:>11.2f} {r['gustafson_comm_E']:>8.4f}"
            print(line)
    if not args.no_plot:
        os.makedirs(os.path.dirname(os.path.abspath(args.plot)), exist_ok=True)
        plot(rows, alphas, args.plot, args.comm_model)
        print(f"[+] Plot written: {args.plot}")


if __name__ == "__main__":
    main()
