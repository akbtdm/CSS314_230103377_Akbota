# Minimal Python Parallel Systems Lab (Ungraded) — Performance Evaluation and Optimization in Parallel Computing (13 Nov 2026)

Report file: **`lab_report_230103377.md`** (hardware table, five `VERIFY_HASH` lines, timing / speedup / Karp-Flatt / Roofline / λ tables, diagnostic answers, raw logs, full source).

Same machine and approach as the solo lab in `../november_13`: Apple M4 (4P + 6E, no SMT, 128-byte line), CPython 3.13.5, standard library only, every number produced by the scripts below.

| file | phase |
|---|---|
| `harness.py` | Listing 1 of the handout (BenchStats, fnv1a, run_benchmark(warmup=2, runs=7), emit_verify_tag) |
| `labutil.py` | shared helpers (student ID, `sysctl` probe, partitioning, JSON output) |
| `task0_cal.py` | Phase 0 — topology audit, 10⁶-call timer latency |
| `task1_falsesharing.py` | Phase 1 — M = 1.6·10⁷ false-sharing stride sweep, Karp-Flatt |
| `task2_scaling.py` | Phase 2 — strong (N = 1.2·10⁶) vs weak (N₀ = 2·10⁵) scaling |
| `task3_roofline_gemm.py` | Phase 3 — 2 × 64 MB bandwidth, 6·10⁷-FLOP FMA ceiling, GEMM N = 256 with BS ∈ {8…128} |
| `task4_scheduling.py` | Phase 4 — N = 8000, K(i) = 1 + ⌊(i/40)²⌋, block / cyclic / dynamic C ∈ {1, 8, 64, 512} |
| `run_all.sh`, `emit_hashes.py`, `make_report.py` | run everything, self-audit the hashes, assemble the report |

```bash
./run_all.sh
python3 emit_hashes.py | tee logs/emit_hashes.log
python3 make_report.py
```
