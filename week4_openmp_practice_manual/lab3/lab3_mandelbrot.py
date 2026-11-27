"""Lab 3: loop scheduling policies on the Mandelbrot set (Python edition).

One Numba parallel region forks P OpenMP threads (prange over ranks) and each
thread pulls rows according to the policy, exactly as OpenMP would:
  static(C)  : chunk k of C rows goes to thread k % P (round-robin, fixed)
  dynamic(C) : shared row counter, each idle thread does fetch-and-add(C)
  guided(C)  : chunk = max(C, ceil(remaining / P)) claimed with compare-and-swap
  block      : prange default (static, one contiguous block of H/P rows each)
Each thread counts the escape-time iterations it executed (Task 3.4).
"""
import csv
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
os.environ.setdefault("NUMBA_NUM_THREADS", "16")
from omp_numba import atomic_cas_i64, atomic_fetch_add_i64, banner  # noqa: E402

import numba  # noqa: E402
import numpy as np  # noqa: E402
from numba import njit, prange  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
WIDTH, HEIGHT, MAX_ITER = 1920, 1080, 1000
POLICIES = {"static": 0, "dynamic": 1, "guided": 2}
P_VALUES = [2, 4, 8, 16]
CHUNKS = [1, 16, 64, 256]
RUNS = 3


@njit
def compute_pixel(px, py, width, height, max_iter):
    x0 = (px - width / 2.0) * 4.0 / width
    y0 = (py - height / 2.0) * 4.0 / height
    x, y = 0.0, 0.0
    iteration = 0
    while x * x + y * y <= 4.0 and iteration < max_iter:
        xtemp = x * x - y * y + x0
        y = 2.0 * x * y + y0
        x = xtemp
        iteration += 1
    return iteration


@njit
def do_rows(img, start, end, width, height, max_iter):
    w = 0
    for y in range(start, end):
        for x in range(width):
            it = compute_pixel(x, y, width, height, max_iter)
            img[y, x] = it
            w += it
    return w


@njit
def render_serial(img, width, height, max_iter):
    return do_rows(img, 0, height, width, height, max_iter)


@njit(parallel=True)
def render_block_static(img, work, width, height, max_iter, p):
    per = (height + p - 1) // p
    for rank in prange(p):
        start = rank * per
        end = min(height, start + per)
        work[rank] = do_rows(img, start, end, width, height, max_iter)


@njit(parallel=True)
def render_scheduled(img, work, counter, width, height, max_iter, p, chunk, policy):
    for rank in prange(p):
        w = 0
        if policy == 0:                                   # static, chunk
            c = rank
            while c * chunk < height:
                s = c * chunk
                w += do_rows(img, s, min(height, s + chunk), width, height, max_iter)
                c += p
        elif policy == 1:                                 # dynamic, chunk
            while True:
                s = atomic_fetch_add_i64(counter, 0, chunk)
                if s >= height:
                    break
                w += do_rows(img, s, min(height, s + chunk), width, height, max_iter)
        else:                                             # guided, chunk
            while True:
                old = atomic_fetch_add_i64(counter, 0, 0)   # atomic read
                if old >= height:
                    break
                remaining = height - old
                size = max(chunk, (remaining + p - 1) // p)
                size = min(size, remaining)
                if atomic_cas_i64(counter, 0, old, old + size) == old:
                    w += do_rows(img, old, old + size, width, height, max_iter)
        work[rank] = w


def imbalance(work):
    avg = work.mean()
    return (work.max() - work.min()) / avg if avg > 0 else 0.0


def run_cell(policy, p, chunk, img):
    numba.set_num_threads(p)
    work = np.zeros(p, dtype=np.int64)
    counter = np.zeros(1, dtype=np.int64)
    t0 = time.perf_counter()
    if policy == "block":
        render_block_static(img, work, WIDTH, HEIGHT, MAX_ITER, p)
    else:
        render_scheduled(img, work, counter, WIDTH, HEIGHT, MAX_ITER, p, chunk,
                         POLICIES[policy])
    return time.perf_counter() - t0, work


def main():
    banner(f"Lab 3: Mandelbrot {WIDTH}x{HEIGHT}, MAX_ITER={MAX_ITER}, "
           f"{RUNS} runs per cell (after one warm-up render per cell)")
    img = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
    small = np.zeros((100, 100), dtype=np.int32)
    render_serial(small, 100, 100, 100)
    render_block_static(small, np.zeros(2, dtype=np.int64), 100, 100, 100, 2)
    render_scheduled(small, np.zeros(2, dtype=np.int64), np.zeros(1, dtype=np.int64),
                     100, 100, 100, 2, 4, 1)
    render_serial(img, WIDTH, HEIGHT, MAX_ITER)
    t0 = time.perf_counter()
    total_work = render_serial(img, WIDTH, HEIGHT, MAX_ITER)
    t_serial = time.perf_counter() - t0
    ref = img.copy()
    print(f"serial render: {t_serial:.4f} s | total escape iterations = {total_work:,}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.imsave(os.path.join(RESULTS, "lab3_mandelbrot.png"), np.log1p(ref), cmap="magma")
    except ImportError:
        pass
    rows, per_thread = [], []
    configs = [("block", p, 0) for p in P_VALUES]
    configs += [(pol, p, c) for pol in POLICIES for p in P_VALUES for c in CHUNKS]
    for pol, p, c in configs:
        run_cell(pol, p, c, img)                         # warm-up
        times, imbs = [], []
        for r in range(1, RUNS + 1):
            img[:] = 0
            t, work = run_cell(pol, p, c, img)
            assert work.sum() == total_work and np.array_equal(img, ref)
            times.append(t)
            imbs.append(imbalance(work))
            per_thread.append(dict(policy=pol, P=p, chunk=c, run=r,
                                   work=";".join(str(x) for x in work)))
        mean_t = statistics.mean(times)
        rows.append(dict(policy=pol, P=p, chunk=c, mean_s=f"{mean_t:.4f}",
                         runs=";".join(f"{x:.4f}" for x in times),
                         speedup_vs_serial=f"{t_serial/mean_t:.2f}",
                         imbalance_mean=f"{statistics.mean(imbs):.3f}",
                         imbalance_runs=";".join(f"{x:.3f}" for x in imbs)))
        print(f"{pol:<7} P={p:<2} C={c:<3}: mean {mean_t:.4f} s | runs "
              + " ".join(f"{x:.4f}" for x in times)
              + f" | S={t_serial/mean_t:5.2f} | imbalance {statistics.mean(imbs):.3f}",
              flush=True)
    for name, data in (("lab3_schedule_matrix.csv", rows),
                       ("lab3_per_thread_work.csv", per_thread)):
        path = os.path.join(RESULTS, name)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
        print(f"wrote {path}")
    with open(os.path.join(RESULTS, "lab3_serial.txt"), "w") as fh:
        fh.write(f"serial_s={t_serial:.4f}\ntotal_iterations={total_work}\n")


if __name__ == "__main__":
    main()
