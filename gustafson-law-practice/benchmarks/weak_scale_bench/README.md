# Weak-Scaling Benchmark Suite (Task 22)

Python `multiprocessing` benchmark. Every process integrates a fixed number of
points `K`, so the global problem grows as `P * K` and ideal wall-clock time
stays constant (Gustafson / weak scaling).

## Requirements
* Python 3.9+
* `numpy` (benchmark), `matplotlib` (plots)

```bash
python3 -m pip install numpy matplotlib
```

No compilation is needed.

## Run

From the repository root (`gustafson-law-practice/`):

```bash
# Full run: 5 interleaved repetitions, P = 1,2,4,6,8,10 (up to os.cpu_count())
python3 benchmarks/weak_scale_bench/bench.py

# Custom process counts / problem size per process
python3 benchmarks/weak_scale_bench/bench.py --procs 1 2 4 6 8 10 --k 20000000 --runs 5

# Quick smoke test (seconds)
python3 benchmarks/weak_scale_bench/bench.py --procs 1 2 3 4 5 --k 200000 --runs 2 --cooldown 0 --out /tmp/test.csv
```

Outputs:
* `benchmarks/results.csv`: raw timings (one row per run × P)
* `benchmarks/hardware.json`: CPU model, core topology, caches (via `sysctl` on macOS)

Then:

```bash
python3 benchmarks/fit_gustafson.py     # mean/std, alpha_observed (Task 23)
python3 benchmarks/plot_scaling.py      # plots/scaling_comparison.png + .pdf (Task 24)
```

## Measurement notes
* Pool spawn and the numpy import in each worker happen **before** the timer starts
  (macOS uses the `spawn` start method, which costs tens of ms per worker).
  Spawn time is still recorded in `pool_startup_s`.
* The timed region contains partitioning, dispatch (pickle + pipes), compute,
  gather and the final reduction on the master: the full parallel run.
* Close other applications and keep the laptop on power during the run.
  Default `K = 20,000,000` gives about 0.1–0.3 s per measurement on an Apple M4.
