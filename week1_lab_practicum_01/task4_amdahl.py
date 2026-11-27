"""Task 4: Amdahl vs Gustafson from the measured Task 2 T1 and T2.

Reads results/task2_scaling.json (written by task2_scaling.py).
Usage: python3 task4_amdahl.py | tee logs/task4_output.txt
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CORES = 64


def main():
    with open(os.path.join(HERE, "results", "task2_scaling.json")) as f:
        rows = {r["threads"]: r for r in json.load(f)["rows"]}
    t1, t2 = rows[1]["avg"], rows[2]["avg"]
    p_raw = 2 * (t1 - t2) / t1
    p = min(max(p_raw, 0.0), 1.0)
    serial = 1 - p
    s_max = float("inf") if serial == 0 else 1 / serial
    s64 = 1 / (serial + p / CORES)
    gust = serial + p * CORES
    out = {"T1": t1, "T2": t2, "p_raw": p_raw, "p": p, "serial_fraction": serial,
           "amdahl_max": s_max, "amdahl_64": s64, "gustafson_64": gust}
    print(f"T1 = {t1:.4f} s | T2 = {t2:.4f} s")
    print(f"p = 2*(T1-T2)/T1 = {p_raw:.4f}" + ("" if p == p_raw else f" (clamped to {p:.4f})"))
    print(f"sequential fraction (1-p) = {serial:.4f}")
    print(f"Amdahl S_max (N->inf) = 1/(1-p) = {s_max:.2f}x")
    print(f"Amdahl S_64 = 1/[(1-p)+p/64] = {s64:.2f}x")
    print(f"Gustafson S_64 = (1-p)+64p = {gust:.2f}x")
    print("\nAmdahl prediction vs measured Task 2 speedup:")
    for n, r in sorted(rows.items()):
        pred = 1 / (serial + p / n)
        print(f"  N={n:2d}: Amdahl {pred:5.2f}x  measured {t1 / r['avg']:5.2f}x")
    with open(os.path.join(HERE, "results", "task4_amdahl.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
