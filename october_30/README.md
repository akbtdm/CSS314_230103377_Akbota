# CSS314 Lab Practicum: Memory Hierarchy, Concurrency Contention and Scaling Limits

Student ID 230103377. Salted parameters: N = 1280, seed = 30385, stride buffer 2^24 doubles.
Hardware: Apple M4, 10 cores (4 performance + 6 efficiency, no SMT), 128-byte cache line,
L1d 128 KB (P) / 64 KB (E), L2 16 MB (P cluster) / 4 MB (E cluster), no classic L3, 16 GB LPDDR5X.
Software: macOS 15.5, CPython 3.13.5, NumPy 2.1.3, Numba 0.61 (used to compile the inner loops, as the
lab allows; the CPython interpreter costs ~65 ns per array access and hides every cache effect).

## Summary of bottleneck findings

On this machine every kernel in the practicum is limited by data movement, not by arithmetic. Reading a
134 MB array with stride 1 costs 0.15 ns per element (52 GB/s from one core), and the per-element cost
climbs 16x by stride 16 because that is where one access per 128-byte cache line is reached (M4 lines hold
16 doubles, not 8); from stride 1 to 16 the total sweep time barely changes because the same number of
lines is fetched either way. Swapping the matrix-multiply loop order from i-j-k (stride-N walks down B) to
i-k-j (stride-1 across rows of B) gives a 4.84x speedup at M = 640 with bit-identical results. In the
shared-memory counter experiment, eight processes incrementing adjacent 8-byte slots slow each increment
from 0.83 ns to 2.47 ns and push the Karp-Flatt metric from 1.10 (P = 2) to 3.27 (P = 8); padding by the
lab's 64 bytes only half-fixes it on a 128-byte line (e(4) = 1.81), padding by a full line keeps e(4) at
1.12, and privatised counters are fastest of all. The Jacobi stencil stops scaling at 2.08x on the four
performance cores (Amdahl fit s = 0.43, S_max = 2.3); weak scaling plateaus at ~20 GFLOP/s from P = 2
onward, which with write-allocate traffic is ~120 GB/s, i.e. the DRAM limit of the chip. With an
arithmetic intensity of 0.25 FLOP/byte against a ridge point of ~3.5 FLOP/byte the kernel sits far down the
memory-bandwidth roof. The practical lessons: lay data out so each cache line is used completely, keep
independently written hot variables on separate 128-byte lines, and expect speedup from extra cores only
until the memory system saturates, which on M4 happens at two to four performance cores.

## Results at a glance

| Experiment | Key numbers |
|---|---|
| Stride cliff (`stride_results.tsv`) | 0.153 ns/elem at stride 1, 2.47 at stride 16, 3.49 at stride 128 (22.8x) |
| Loop interchange (`loop_interchange.json`) | i-j-k 0.158 s vs i-k-j 0.033 s at M = 640, speedup 4.84x (5.33x at N = 1280), bitwise equal, rel. error vs BLAS 3.9e-15 |
| False sharing (`false_sharing_karp_flatt.tsv`) | unpadded 0.83 / 0.87 / 1.65 / 2.47 ns per increment for P = 1 / 2 / 4 / 8; e(P) = 1.10 / 2.33 / 3.27; 128-byte padding e(4) = 1.12; privatised 0.14 ns flat |
| Strong scaling (`scaling_metrics.tsv`) | S_P = 1.71 / 2.08 / 1.84 / 1.66 for P = 2 / 4 / 8 / 10; Amdahl s = 0.432, S_max = 2.31 |
| Weak scaling | N_P = ceil(1280 * sqrt(P)); aggregate throughput flat at ~20 GFLOP/s (~80 GB/s useful) for P = 2..10 |
| Roofline (`roofline_summary.json`) | AI = 0.25 FLOP/B, ridge ~3.5 FLOP/B, memory-bandwidth-bound; best measured 28.6 GFLOP/s vs 30 GFLOP/s DRAM roof |
| Verification (`verification_receipt.sig`) | all four milestones PASS, all counters exactly P * K, stencil checksums identical for every P |

## Layout

```
october_30/
├── src/
│   ├── profile_env.py      # M0: host telemetry -> host_profile.json, salted_params.json
│   ├── salt.py             # M0: student-ID salting (given in the assignment)
│   ├── common.py           # shared paths/loaders
│   ├── stride_bench.py     # M1: stride latency cliff
│   ├── dense_kernel.py     # M1: i-j-k vs i-k-j
│   ├── false_sharing.py    # M2: unpadded / 64 B / 128 B padding / privatised + Karp-Flatt
│   └── scaling_study.py    # M3: strong + weak scaling, Amdahl fit, Roofline
├── results/                # all empirical outputs listed above
├── verify_outputs.py       # M4: checks every artefact, emits verification_receipt.sig
└── README.md
```

## Reproduce

```
python3 src/profile_env.py --student-id 230103377
python3 src/stride_bench.py
python3 src/dense_kernel.py
python3 src/false_sharing.py        # ~1 min, both Python and Numba backends
python3 src/scaling_study.py        # ~30 s
python3 verify_outputs.py
```

## Methodology notes

- Timings are medians over repeated runs after warm-up; JIT compilation and process start-up are never
  inside a timed region. Multi-process experiments time only the hot loop, released by a Barrier, and
  report the slowest worker.
- The shared counter array is created with `lock=False`: each worker owns a distinct slot, so the
  measurement isolates cache-coherence traffic instead of lock contention.
- Jumps at P = 8 and P = 10 that appear in *every* variant (including privatised counters that touch no
  shared line) come from the heterogeneous cores: beyond four processes the scheduler uses the slower
  efficiency cores, and barriers wait for the slowest one.
- The lab's scaled-speedup formula reduces to T_1 / T_P when Work(P) = P * Work(1) and therefore measures
  weak-scaling efficiency (ideal 1.0); a Gustafson speedup column (ideal P) is reported alongside it.
- M4 peak compute and DRAM bandwidth used for the Roofline ceilings come from public specifications and
  are labelled as assumptions in `roofline_summary.json`; the measured single-core stream bandwidth from
  Milestone 1 is recorded next to them.
