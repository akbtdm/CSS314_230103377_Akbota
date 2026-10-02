"""
Runs all three lab challenges and prints/saves everything needed for Table 1.

Usage:
    python run_lab.py            # full run
    python run_lab.py --quick    # smaller sizes, for a fast sanity check

Outputs:
    mandelbrot_output.png   (Challenge 2, deliverable D)
    results.csv             (all numbers for Table 1)
"""
import argparse
import csv
import platform
import time

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numba
import numpy as np
from numba import njit, prange

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true", help="use small sizes")
args = parser.parse_args()

MAX_T = numba.config.NUMBA_NUM_THREADS
SAMPLES = 5_000_000 if args.quick else 120_000_000
H = W = 600 if args.quick else 2500
MAX_IT = 300 if args.quick else 1000
GRID = 500 if args.quick else 1500
STEPS = 50 if args.quick else 300

rows_out = []  # for results.csv

print("=" * 60)
print(f"CPU:      {platform.processor() or platform.machine()} ({platform.system()})")
print(f"Threads:  {MAX_T}   Numba {numba.__version__}   NumPy {np.__version__}")
print(f"Threading layer will be chosen at first parallel call")
print("=" * 60)


# ----------------------------------------------------------------------
# Challenge 1: Monte Carlo pi
# ----------------------------------------------------------------------
@njit(parallel=True)
def monte_carlo_pi(n_samples):
    inside = 0
    for i in prange(n_samples):
        x = np.random.uniform(0.0, 1.0)
        y = np.random.uniform(0.0, 1.0)
        if x * x + y * y <= 1.0:
            inside += 1
    return (4.0 * inside) / n_samples


_ = monte_carlo_pi(10_000)  # warm-up (JIT compile)
print(f"Threading layer in use: {numba.threading_layer()}\n")

print("CHALLENGE 1: Monte Carlo pi")
counts = sorted({t for t in [1, 2, 4, 8, MAX_T] if t <= MAX_T})
print(f"{'Threads':<8} | {'Time (s)':<10} | {'Speedup':<9} | {'Efficiency %':<12} | pi")
print("-" * 62)
t1 = None
for t in counts:
    numba.set_num_threads(t)
    start = time.perf_counter()
    pi_est = monte_carlo_pi(SAMPLES)
    el = time.perf_counter() - start
    if t == 1:
        t1 = el
    sp = t1 / el
    eff = sp / t * 100
    print(f"{t:<8} | {el:<10.4f} | {sp:<8.2f}x | {eff:<12.1f} | {pi_est:.5f}")
    rows_out.append(["Monte Carlo", f"{t} threads", SAMPLES, f"{el:.4f}", f"{sp:.2f}", f"{eff:.1f}"])
numba.set_num_threads(MAX_T)


# ----------------------------------------------------------------------
# Challenge 2: Mandelbrot rows vs cols
# ----------------------------------------------------------------------
@njit(parallel=True)
def render_rows(h, w, max_iter):
    img = np.zeros((h, w), dtype=np.int32)
    for r in prange(h):
        cy = -1.2 + (r / h) * 2.4
        for c in range(w):
            cx = -2.0 + (c / w) * 2.5
            zr, zi = 0.0, 0.0
            it = 0
            while (zr * zr + zi * zi <= 4.0) and (it < max_iter):
                nr = zr * zr - zi * zi + cx
                zi = 2.0 * zr * zi + cy
                zr = nr
                it += 1
            img[r, c] = it
    return img


@njit(parallel=True)
def render_cols(h, w, max_iter):
    img = np.zeros((h, w), dtype=np.int32)
    for c in prange(w):
        cx = -2.0 + (c / w) * 2.5
        for r in range(h):
            cy = -1.2 + (r / h) * 2.4
            zr, zi = 0.0, 0.0
            it = 0
            while (zr * zr + zi * zi <= 4.0) and (it < max_iter):
                nr = zr * zr - zi * zi + cx
                zi = 2.0 * zr * zi + cy
                zr = nr
                it += 1
            img[r, c] = it
    return img


_ = render_rows(100, 100, 50)
_ = render_cols(100, 100, 50)

print("\nCHALLENGE 2: Mandelbrot")
t0 = time.perf_counter()
grid_rows = render_rows(H, W, MAX_IT)
t_rows = time.perf_counter() - t0
t0 = time.perf_counter()
grid_cols = render_cols(H, W, MAX_IT)
t_cols = time.perf_counter() - t0
print(f"Row-parallel render:    {t_rows:.3f} s")
print(f"Column-parallel render: {t_cols:.3f} s")
print(f"Faster axis: {'rows' if t_rows < t_cols else 'cols'} "
      f"({max(t_rows, t_cols) / min(t_rows, t_cols):.2f}x)")
print(f"Outputs identical: {np.array_equal(grid_rows, grid_cols)}")
rows_out.append(["Mandelbrot (Rows)", f"{MAX_T} threads", f"{H}x{W}", f"{t_rows:.3f}", "N/A", "N/A"])
rows_out.append(["Mandelbrot (Cols)", f"{MAX_T} threads", f"{H}x{W}", f"{t_cols:.3f}", "N/A", "N/A"])

plt.figure(figsize=(8, 8))
plt.imshow(grid_rows, cmap="magma", extent=[-2.0, 0.5, -1.2, 1.2])
plt.title(f"Mandelbrot {H}x{W} (Render: {t_rows:.2f}s)")
plt.axis("off")
plt.savefig("mandelbrot_output.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved image: mandelbrot_output.png")


# ----------------------------------------------------------------------
# Challenge 3: Heat stencil (float64 vs float32, plus thread scaling)
# ----------------------------------------------------------------------
@njit(parallel=True)
def heat_step(u, u_next, alpha):
    rows, cols = u.shape
    for i in prange(1, rows - 1):
        for j in range(1, cols - 1):
            u_next[i, j] = u[i, j] + alpha * (
                u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1] - 4.0 * u[i, j]
            )


def run_heat(dtype, steps=STEPS):
    u = np.zeros((GRID, GRID), dtype=dtype)
    un = np.zeros_like(u)
    u[0, :] = 100.0
    u[:, 0] = 100.0
    un[0, :] = 100.0
    un[:, 0] = 100.0
    alpha = dtype(0.20)  # keep arithmetic in the array's precision
    heat_step(u, un, alpha)  # warm-up (compiles a version per dtype)
    start = time.perf_counter()
    for _ in range(steps):
        heat_step(u, un, alpha)
        u, un = un, u
    el = time.perf_counter() - start
    return el, (GRID * GRID * steps) / el / 1e6


print("\nCHALLENGE 3: Heat stencil")
el64, mc64 = run_heat(np.float64)
print(f"float64 ({MAX_T} threads): {el64:.3f} s   {mc64:.2f} Megacells/s")
el32, mc32 = run_heat(np.float32)
print(f"float32 ({MAX_T} threads): {el32:.3f} s   {mc32:.2f} Megacells/s")
print(f"float32 speedup over float64: {el64 / el32:.2f}x")
rows_out.append(["Heat Stencil f64", f"{MAX_T} threads", f"{GRID}x{GRID}x{STEPS}", f"{el64:.3f}", "N/A", "N/A"])
rows_out.append(["Heat Stencil f32", f"{MAX_T} threads", f"{GRID}x{GRID}x{STEPS}", f"{el32:.3f}", "N/A", "N/A"])

print("\nExtra (for question 3C): stencil thread scaling, float64")
print(f"{'Threads':<8} | {'Time (s)':<10} | {'Speedup':<9} | Megacells/s")
print("-" * 46)
base = None
for t in counts:
    numba.set_num_threads(t)
    el, mc = run_heat(np.float64)
    if t == 1:
        base = el
    print(f"{t:<8} | {el:<10.3f} | {base / el:<8.2f}x | {mc:.2f}")
    rows_out.append(["Heat scaling f64", f"{t} threads", f"{GRID}x{GRID}x{STEPS}",
                     f"{el:.3f}", f"{base / el:.2f}", f"{base / el / t * 100:.1f}"])
numba.set_num_threads(MAX_T)

# ----------------------------------------------------------------------
with open("results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Benchmark", "Threads", "Size/Samples", "Time_s", "Speedup", "Efficiency_pct"])
    w.writerows(rows_out)
print("\nSaved results.csv  -- copy these numbers into Table 1.")