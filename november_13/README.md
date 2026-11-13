# Solo Python Systems Lab — Performance Evaluation and Optimization in Parallel Computing (13 Nov 2026)

Submission file: **`lab_report_230103377.md`** (hardware baseline, five `VERIFY_HASH` lines, Tables 1A/1B, 2A/2B, Roofline + 10-point GEMM tile sweep, scheduling/λ table, analysis, raw logs, full source).

Everything was measured on an Apple M4 (4P + 6E cores, no SMT, 128-byte cache line) with CPython 3.13.5 and the standard library only.

| file | phase |
|---|---|
| `harness.py` | timing / median-IQR / `VERIFY_HASH` framework from the handout |
| `labutil.py` | shared helpers (student ID, `sysctl` probe, partitioning, JSON output) |
| `task0_cal.py` | Phase 0 — topology audit, timer calibration |
| `task1_falsesharing.py` | Phase 1 — shared-memory false-sharing stride sweep, Karp-Flatt |
| `task2_scaling.py` | Phase 2 — strong (Amdahl) vs weak (Gustafson) scaling on `mp.Pool` |
| `task3a_roofline.py` | Phase 3A — DRAM bandwidth and FMA ceiling, ridge point |
| `task3b_gemm.py` | Phase 3B — naive / i-k-j / 6-loop blocked GEMM, N = 384, tile sweep |
| `task4_scheduling.py` | Phase 4 — block / cyclic / dynamic-chunk scheduling, load imbalance λ |
| `run_all.sh` | runs every phase sequentially, writes `logs/` and `results/` |
| `emit_hashes.py` | recomputes the five hashes from `results/*.json` (self-audit) |
| `make_report.py` | assembles the report from `results/`, `logs/`, `analysis/` and the sources |
| `results_quick/` | output of the `--quick` smoke-test configuration (used once in the Phase 1 analysis to quantify spawn overhead) |

Reproduce:

```bash
./run_all.sh                 # ~5 minutes on the M4
python3 emit_hashes.py | tee logs/emit_hashes.log
python3 make_report.py
```
