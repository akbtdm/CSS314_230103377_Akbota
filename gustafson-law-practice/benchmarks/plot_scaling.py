#!/usr/bin/env python3
"""
Task 24 - Publication-quality scaling plot.

Plots (1) ideal linear speedup, (2) Gustafson curve with the fitted alpha,
(3) measured scaled speedup with standard-error bars, and (4) Amdahl's curve with
the same serial fraction. Writes plots/scaling_comparison.png and .pdf.
Run fit_gustafson.py first (or this script will run the fit itself).
"""
import argparse
import os

import numpy as np

import fit_gustafson


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=os.path.join(here, "results.csv"))
    ap.add_argument("--out", default=os.path.join(here, "..", "plots", "scaling_comparison"))
    ap.add_argument("--title", default=None)
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary, res = fit_gustafson.fit(fit_gustafson.load(args.csv))
    alpha = res["alpha_observed"]
    P = np.array([r["procs"] for r in summary], float)
    S = np.array([r["S_mean"] for r in summary])
    sem = np.array([r["S_sem"] for r in summary])
    E = S / P
    pp = np.linspace(1, P.max(), 200)
    gus = pp - alpha * (pp - 1)
    amd = 1.0 / (alpha + (1 - alpha) / pp)

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 5), gridspec_kw={"width_ratios": [1.4, 1]})
    ax.plot(pp, pp, ":", color="0.35", lw=1.5, label="(1) Ideal linear S = P")
    ax.plot(pp, gus, "-", color="#1f6fb4", lw=2.2,
            label=f"(2) Gustafson fit, α = {alpha:.4f} ± {res['alpha_std_error']:.4f}")
    ax.errorbar(P, S, yerr=sem, fmt="o", color="#d1495b", ms=6, capsize=4, lw=1.5,
                label="(3) Measured (mean ± SE, n = %d)" % summary[0]["n_runs"])
    ax.plot(pp, amd, "--", color="#edae49", lw=2, label=f"(4) Amdahl, s = {alpha:.4f}")
    ax.set_xlabel("Processes P")
    ax.set_ylabel("Speedup S(P)")
    ax.set_xticks(P)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left", frameon=False)
    ax.set_title(args.title or "Weak scaling: measured vs. theoretical")

    ax2.bar(P, E, width=0.6, color="#1f6fb4", alpha=0.8)
    ax2.axhline(1 - alpha, color="#d1495b", ls="--", lw=1.5, label=f"1 − α = {1 - alpha:.3f}")
    ax2.set_xlabel("Processes P")
    ax2.set_ylabel("Efficiency E = S/P")
    ax2.set_xticks(P)
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(frameon=False)
    ax2.set_title("Measured weak-scaling efficiency")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out + ".png", dpi=300)
    fig.savefig(args.out + ".pdf")
    print(f"[+] {args.out}.png\n[+] {args.out}.pdf")


if __name__ == "__main__":
    main()
