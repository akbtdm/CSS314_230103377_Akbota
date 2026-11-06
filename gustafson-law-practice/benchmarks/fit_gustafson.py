#!/usr/bin/env python3
"""
Task 23 - Statistics and least-squares fit of Gustafson's law.

Reads benchmarks/results.csv (from weak_scale_bench/bench.py) and computes, per P:
  mean and std of T(P) over runs, scaled speedup and efficiency.

Scaled speedup for weak scaling: the P-times larger problem would take
P * T(1) on one process, so   S(P) = P * mean(T(1)) / T(P).

alpha_observed is fitted by least squares on Gustafson's model
    S(P) = P - alpha (P - 1)   <=>   (P - S) = alpha (P - 1)
which is a regression through the origin:
    alpha = sum((P - S)(P - 1)) / sum((P - 1)^2)
using every individual run with P >= 2. A free linear fit S = a + b P is also
reported for comparison (b ~ 1 - alpha if the model holds).

Outputs: benchmarks/summary.csv and benchmarks/fit_results.json
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np


def load(path):
    data = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            data[int(r["procs"])].append(float(r["wall_s"]))
    return dict(sorted(data.items()))


def fit(data):
    if 1 not in data:
        raise SystemExit("results.csv must contain P = 1 runs (baseline).")
    t1 = float(np.mean(data[1]))
    P_all, S_all = [], []
    summary = []
    for p, times in data.items():
        t = np.array(times)
        s_runs = p * t1 / t
        n = len(t)
        summary.append({
            "procs": p, "n_runs": n,
            "T_mean_s": t.mean(), "T_std_s": t.std(ddof=1) if n > 1 else 0.0,
            "S_mean": s_runs.mean(), "S_std": s_runs.std(ddof=1) if n > 1 else 0.0,
            "S_sem": (s_runs.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "E_mean": (s_runs / p).mean(),
        })
        if p >= 2:
            P_all += [p] * n
            S_all += list(s_runs)

    P_arr, S_arr = np.array(P_all, float), np.array(S_all, float)
    x, y = P_arr - 1.0, P_arr - S_arr
    alpha = float(np.sum(x * y) / np.sum(x * x))
    resid = y - alpha * x
    dof = max(1, len(x) - 1)
    alpha_se = float(np.sqrt(np.sum(resid ** 2) / dof / np.sum(x * x)))
    ss_tot = np.sum((S_arr - S_arr.mean()) ** 2)
    pred = P_arr - alpha * (P_arr - 1)
    r2 = float(1 - np.sum((S_arr - pred) ** 2) / ss_tot) if ss_tot > 0 else float("nan")
    b, a = np.polyfit(P_arr, S_arr, 1)

    for row in summary:
        p = row["procs"]
        row["S_gustafson_fit"] = p - alpha * (p - 1)
        row["alpha_karp_flatt"] = ((1 / row["S_mean"]) - (1 / p)) / (1 - 1 / p) if p > 1 else 0.0
    result = {
        "T1_mean_s": t1,
        "alpha_observed": alpha,
        "alpha_std_error": alpha_se,
        "r2_gustafson_model": r2,
        "free_linear_fit": {"intercept": float(a), "slope": float(b), "implied_alpha": float(1 - b)},
        "n_points": int(len(x)),
    }
    return summary, result


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=os.path.join(here, "results.csv"))
    ap.add_argument("--summary", default=os.path.join(here, "summary.csv"))
    ap.add_argument("--json", default=os.path.join(here, "fit_results.json"))
    args = ap.parse_args(argv)

    summary, result = fit(load(args.csv))
    with open(args.summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows([{k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()} for r in summary])
    with open(args.json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"{'P':>4} {'n':>3} {'T mean (s)':>11} {'T std':>9} {'S mean':>8} {'S std':>7} {'E':>7} {'S fit':>7} {'e_KF':>8}")
    for r in summary:
        print(f"{r['procs']:>4} {r['n_runs']:>3} {r['T_mean_s']:>11.4f} {r['T_std_s']:>9.4f} {r['S_mean']:>8.3f} "
              f"{r['S_std']:>7.3f} {r['E_mean']:>7.3f} {r['S_gustafson_fit']:>7.3f} {r['alpha_karp_flatt']:>8.4f}")
    print(f"\nalpha_observed = {result['alpha_observed']:.5f} ± {result['alpha_std_error']:.5f} (SE), "
          f"R² = {result['r2_gustafson_model']:.4f}, n = {result['n_points']}")
    lf = result["free_linear_fit"]
    print(f"free linear fit: S = {lf['intercept']:.3f} + {lf['slope']:.4f}·P  (implied alpha = {lf['implied_alpha']:.4f})")
    print(f"[+] {args.summary}\n[+] {args.json}")


if __name__ == "__main__":
    main()
