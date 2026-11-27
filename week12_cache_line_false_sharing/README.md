# cache-line-false-sharing-akbtdm: Week 12 Caching Tasks

**Student:** Akbota Kutushova, ID 230103377, Git handle `akbtdm`
**Course:** Systems Programming (Instructor: Sufyan bin Uzayr). Lab: "Cache Line & False Sharing Mechanics in Multi-Core Python"
**Machine:** Apple M4: 10 cores (4 P + 6 E, no SMT), `hw.cachelinesize` = 128 B, L1d 128 KB (P) / 64 KB (E), L2 16 MB (P) / 4 MB (E), 16 GB RAM, macOS 15.5, Python 3.13.5, numpy 2.1.3, numba 0.61.0 (`logs/hw_info.log`).

> **Repository note.** The handout asks for a separate repository named `cache-line-false-sharing-akbtdm`. A GitHub repository cannot be created from this machine's session, so this folder of the course repository is laid out exactly as that repository would be, with the three required files at its root. To move it, copy the folder's contents into a new repository, or run `git subtree split -P week12_cache_line_false_sharing -b cache-line-false-sharing-akbtdm` and push that branch to the new repository.

Every number below was measured on this machine in one `./run_all.sh` run on 2026-11-27 15:37 (+05). The whole suite took **15.5 s**, with benchmarks running one at a time. The 1-minute load average before the run was 2.41 (desktop apps only). The full console output is in `logs/false_sharing_lab.log`, and the manifest is `cache_lab_manifest.json` with **`"verification_passed": true`**. Values I looked up instead of measuring are marked **(looked up)**.

## Files

| File | Purpose |
|---|---|
| `false_sharing_lab.py` | The handout's starter with every TODO completed: Tasks 1-5, plus the extras marked in the code |
| `cache_lab_manifest.json` | Generated telemetry manifest: context, every run, parity, LLVM audit, `verification_passed` |
| `ANALYSIS.md` | Answers to Q1-Q5, as the handout requires (also reproduced at the end of this README) |
| `run_all.sh` | Writes `logs/hw_info.log`, then runs the lab and tees the output into `logs/false_sharing_lab.log` |
| `results/llvm_*.ll` | Optimized LLVM IR of each kernel's core function, saved by `audit_llvm_and_parity` |
| `results/asm_worker_strided_accumulate.txt` | arm64 instructions of the Task 2 loop |

Reproduce with `./run_all.sh` (options: `--iters`, `--reps`, `--extra-reps`).

## Task 1: hardware topology and identity binding

| Field | Value |
|---|---|
| GIT_USERNAME | `akbtdm` |
| CPU signature (`machine::processor::cores`) | `arm64::arm::10` (brand: Apple M4, 4 P + 6 E) |
| Cache line size | 128 B (`sysctl hw.cachelinesize`; there is no Linux sysfs file on macOS) |
| SHA-256(`akbtdm::arm64::arm::10`) | `3534e2b2b188f2638c6946e2ce5333f2986cbd94ec5fb32f9f5d79d14752157b` |
| INSTANCE_SEED (first 16 hex digits) | `0x3534e2b2b188f263` = 3833938439909732963 |
| Thread seeds | `(INSTANCE_SEED + t * 0x9E3779B97F4A7C15) mod 2^64` |
| Core affinity | Not supported. macOS has no `sched_setaffinity`, and Apple Silicon ignores affinity tags. Each worker sets QoS `USER_INTERACTIVE` via `pthread_set_qos_class_self_np`, which steers threads towards the P-cores but does not pin them |
| Buffer | `uint64`, 1024 elements, base address asserted to be 128 B aligned (this also satisfies the handout's 64 B requirement) |

## Tasks 2-4: stride sweep (T = 4, 60,000,000 updates per thread, median of 15 interleaved runs)

The Task 2 kernel is `worker_strided_accumulate`, a `@njit(nogil=True)` in-place RMW: `buf[slot] = buf[slot] + (seed ^ i) mod 2^64`. The load and the store are relaxed atomics (Numba `@intrinsic`, `load_atomic`/`store_atomic` "monotonic"), so every iteration does one real `ldr` and one real `str` (Q4). Timing uses `time.perf_counter_ns()` around a barrier-released start and `join`, and context switches are the `resource.getrusage(RUSAGE_SELF)` delta.

| Layout | Spacing | Lines (64 B model) | Lines (128 B, M4) | Median ms | Min ms | ns/update per thread | vs stride 8 | vs stride 16 |
|---|---|---|---|---|---|---|---|---|
| A: stride = 1 | 8 B | 0,0,0,0 | 0,0,0,0 | **87.6** | 87.4 | 1.460 | **2.25x** | 2.02x |
| B: stride = 4 | 32 B | 0,0,1,1 | 0,0,0,0 | **87.5** | 87.4 | 1.459 | 2.25x | 2.02x |
| C: stride = 8 | 64 B | 0,1,2,3 | 0,0,1,1 | **39.0** | 29.0 | 0.649 | 1.00 | 0.90 |
| D: stride = 16 | 128 B | 0,2,4,6 | 0,1,2,3 | **43.3** | 30.9 | 0.722 | 1.11 | 1.00 |
| extra: stride = 32 | 256 B | 0,4,8,12 | 0,2,4,6 | 47.0 | 34.6 | 0.784 | 1.21 | 1.09 |
| Task 4: register-privatized | (64 B) | - | - | **8.2** | 8.1 | 0.136 | 0.21 | 0.19 |
| extra: plain NumPy indexing, stride 1 | 8 B | 0,0,0,0 | 0,0,0,0 | 8.2 | 8.2 | 0.137 | 0.21 | 0.19 |

Single-thread reference, no contention, median of 7 runs:
- Task 2 kernel: 30.2 ms = 0.503 ns/update. The runs are bimodal, 21.9-62.4 ms.
- Privatized kernel: 7.2 ms = 0.121 ns/update.

Voluntary context switches were 0 in every run. Involuntary switches had a median of 11-15 per run in every layout.

### Extra experiments (same kernel family, T = 4 unless stated, median of 7 runs)

**Latency-bound RMW** `x = x * 6364136223846793005 + (seed ^ i)`. One store per about 2.1 ns, and the next load is forwarded from the store buffer:

| lcg stride 1 | lcg stride 4 | lcg stride 8 | lcg stride 16 | lcg private |
|---|---|---|---|---|
| 126.2 ms (2.103 ns) | 126.2 ms (2.104 ns) | 130.2 ms (2.169 ns) | 126.7 ms (2.111 ns) | 61.7 ms (1.029 ns) |

**Store-throughput-bound** `buf[slot] = seed ^ i` (no load):

| store stride 1 | stride 4 | stride 8 | stride 16 | stride 32 |
|---|---|---|---|---|
| 40.2 ms (0.669 ns), **2.62x** | 18.5 ms (0.309 ns), 1.21x | 15.3 ms (0.256 ns) | 15.3 ms (0.256 ns) | 15.4 ms (0.256 ns) |

**Thread count** (Task 2 kernel):

| T | stride 1 | stride 8 | stride 16 | stride 1 / stride 16 | stride 8 / stride 16 |
|---|---|---|---|---|---|
| 2 | 79.3 ms (1.321 ns) | 38.5 ms (0.642 ns) | 31.2 ms (0.520 ns) | 2.54x | 1.24x |
| 4 | 87.6 ms (1.460 ns) | 39.0 ms (0.649 ns) | 43.3 ms (0.722 ns) | 2.02x | 0.90x |
| 8 (P + E cores) | 143.4 ms (2.389 ns) | 107.8 ms (1.797 ns) | 74.1 ms (1.235 ns) | 1.94x | 1.46x |

**GIL experiment for Q5** (stride 1, median of 3 runs): see the Q5 table below.

## Task 5: parity and LLVM IR verification

- **Parity.** Final counters match bit for bit across all 23 layout groups (7 main, 5 LCG, 5 store-only, 3 at T = 2, 3 at T = 8), against references computed single-threaded (privatized kernels; for store-only the closed form seed ^ (iters-1)). Expected T = 4 finals: `0x283e52b91e60bc80, 0x7b733e4e64acbc80, 0xcea855d9c328bc80, 0x21d0a6250174bc80`. Every repetition of a layout ended with identical values. The Task 2 kernel also matches a pure-Python big-int reference over 200,000 iterations.
- **LLVM IR**, from `worker_strided_accumulate.inspect_llvm()`, core function only, not the CPython wrapper:

| Kernel | i64 stores | inside loop | atomic monotonic load/store | Python/NRT calls |
|---|---|---|---|---|
| `worker_strided_accumulate` (Task 2) | 1 | **1** (`B30`, self-loop) | 1 / 1 | none |
| `worker_private_accumulate` (Task 4) | 1 | 0 (after the loop) | 0 / 0 | none |
| `worker_plain_accumulate` (extra) | 1 | **0**: promoted to a register and vectorized | 0 / 0 | none |
| `worker_strided_lcg` (extra) | 1 | 1 | 1 / 1 | none |
| `worker_strided_store` (extra) | 1 | 1 | 0 / 1 | none |

`verification_passed` = parity_all AND (Task 2 store is atomic, inside the loop, with no Python calls) AND (Python reference matches) = **true**.

## Deviations and notes

1. **The starter file in the handout is cut off** in the middle of the Task 1 TODO (page 4 of 4 ends there). Everything after it (seed, affinity, Tasks 2-5, manifest schema) is my own code written from the task text. The functions keep the handout's names.
2. **Language.** Python, as the handout requires, with Numba `@njit(nogil=True)` and `threading.Thread`. CPython's GIL would hide false sharing (Q5), and LLVM would promote a plain `buf[slot]` RMW into a register and remove the per-iteration store (measured: 8.2 ms, the same as the privatized kernel). So the Task 2 kernel uses relaxed-atomic `@intrinsic` load/store to force a real store every iteration. On arm64 these are plain `ldr`/`str`, with no fences.
3. **Affinity.** macOS cannot pin threads to cores, so only a QoS hint is used, and the manifest records this. Thread placement is up to the scheduler, which is the most likely source of the bimodal timings of the uncontended layouts (one thread alone ranged 21.9-62.4 ms). I could not verify this, because macOS has no per-thread core-ID query.
4. **Cache line.** The handout assumes 64 B. The M4 reports 128 B, so both line models are tabulated, and stride 32 (256 B) was added. See Q3 for which model the data supports.
5. **Run-to-run variation.** A preliminary full run of the same code with 7 repetitions gave stride 1 / stride 8 = 1.31x instead of 2.25x, because the padded layouts landed in their slow mode. That log was overwritten. Only the committed run is cited.
6. **Update function.** The handout does not fix the arithmetic. I chose `x + (seed ^ i)` so that the privatized baseline and every layout must produce identical, seed-dependent finals. The extra LCG and store-only kernels show how the store rate changes the visible cost of false sharing.
7. **Nothing in this lab needs a GPU, MPI or screenshots**, so nothing was skipped.

---

## Q1. MESI state transitions under stride = 1

Under stride = 1, `buf[0]` (Core 0) and `buf[1]` (Core 1) are 8 B apart in one line (line 0 under both the 64 B and the 128 B model). The textbook MESI sequence:

1. **Start.** Neither core has the line: Core 0 = I, Core 1 = I.
2. **Core 0 stores to buf[0].** The store misses, so Core 0 sends a Read-For-Ownership (RFO; on ARM fabrics, a "read unique"). Nobody else holds the line, so it arrives and Core 0 = **M** (it passes through E). The store commits to L1.
3. **Core 1 stores to buf[1].** Core 1 has the line in I, so it sends its own RFO. The RFO snoops Core 0. Core 0 supplies the dirty line, either cache-to-cache or through the shared L2 (writing it back to L2 on a plain MESI system), and drops to **I**. Core 1 = **M**. Only then does Core 1's store commit.
4. **Core 0's next store** finds its line in I and sends another RFO, so Core 1 goes M -> I and Core 0 goes I -> M. Each core's own iteration loop now repeats steps 3 and 4. The two cores never touch the same byte, but the line moves between them on almost every store. Under stride = 1 all four threads share the line, so it rotates among four cores.

A line that is read but not written would pass through S. Here every access is load-then-store from the same core, and the RFO takes the line straight to M. Apple does not document the M4's exact protocol (it is probably a MOESI-style variant), so the state names are the textbook model **(looked up)**.

**Why the RFO stalls Core 1's store buffer.** A store executes into the store buffer right away, but it can only *commit* to L1 when the core owns the line in M or E. Core 1's oldest store-buffer entry waits for the whole RFO round trip: request, snoop, invalidate at Core 0, data return. Stores commit in order, so every younger store queues behind that entry. When the buffer is full, the pipeline cannot issue more stores and the core stalls. The cost is measured: per thread, an update takes **1.460 ns** at stride 1 and **0.649 ns** at stride 8. That is **+0.81 ns per update**, or 87.6 ms vs 39.0 ms of wall time.

The extras give a measured nuance: the stall only appears when stores arrive faster than the RFOs can be served.
- In the latency-bound LCG kernel (`x = x*MULT + (seed^i)`), the next load always reads the core's own last store. That load is served by store-to-load forwarding from the store buffer, and the chain allows only one store per about 2.1 ns. False sharing then costs nothing measurable: **lcg stride 1 = 126.2 ms vs lcg stride 16 = 126.7 ms** (ratio 1.00).
- In the store-only kernel (one store per iteration, no dependency), **stride 1 = 40.2 ms vs stride 8 = 15.3 ms, 2.62x**.

So the ping-pong is real in all cases, but it only becomes a stall once the store buffer fills. This reading of the LCG result is my interpretation of the measured numbers. I did not observe the store buffer directly.

## Q2. Measured durations and slowdown

The table above gives the medians. Every individual run is in the manifest under `stride_benchmark.<layout>.wall_ns_runs`.

- **Slowdown of stride 1 relative to stride 8 = 87,571,167 / 38,958,458 = 2.25x** (manifest `slowdown_stride1_vs_stride8` = 2.2478). Relative to stride 16 it is **2.02x**.
- **Average ns per update**, as wall / 60 M per thread (aggregate in brackets):
  - stride 1: 1.460 (0.365)
  - stride 4: 1.459 (0.365)
  - stride 8: 0.649 (0.162)
  - stride 16: 0.722 (0.180)
  - privatized: 0.136 (0.034)
- Stride 1 is **10.7x** slower than the register-privatized baseline.
- **Variance.** The contended rows are very stable: stride 1 ranges 87.4-97.0 ms over 15 runs. The padded rows are bimodal: stride 8 ranges 29.0-63.7 ms and stride 16 ranges 30.9-57.6 ms. The same spread appears with **one thread and no contention at all** (21.9-62.4 ms over 7 runs), so it is a per-core effect, not coherence. I could not determine whether the cause is P-core vs E-core placement or clock changes: macOS reports no per-thread core ID. A preliminary 7-repetition run of the same code, whose log was overwritten, landed mostly in the slow mode for the padded layouts and gave 1.31x for stride 1 / stride 8. The 2.25x above is from the committed run.
- **Context switches.** Voluntary context switches were 0 in every run. Involuntary switches had a median of 11-15 per run for every layout, with occasional spikes up to about 1,480. `getrusage(RUSAGE_SELF)` counts the whole process, and the switch counts do not separate the layouts. The slowdown is in the memory system, not the scheduler.
- **Thread count (extra).** T = 2: stride 1 / stride 16 = 79.3 / 31.2 ms = **2.54x**. T = 8, which uses E-cores too: 143.4 / 74.1 ms = **1.94x**.

**Task 4: store forwarding vs coherence.** The privatized kernel costs 0.136 ns per update. It runs in registers, and LLVM also vectorizes it (14 `<2 x i64>` operations in `results/llvm_worker_private_accumulate.ll`). Stride 8 does a real L1 load and store per iteration that the core resolves locally, through L1 and store forwarding: **0.649 ns, so +0.51 ns** for the memory round trip. Stride 1 adds cross-core coherence on top: **1.460 ns, a further +0.81 ns**.

The LCG extra separates the two effects more cleanly, because its private version cannot be vectorized:
- private: 1.029 ns
- stride 8: 2.169 ns (+1.14 ns of load/store-forwarding latency per iteration)
- stride 1: 2.103 ns (no extra coherence cost)

## Q3. Stride 8 (64 B) vs stride 16 (128 B) and the spatial prefetcher

**Measured, T = 4:**

| Kernel | stride 8 (64 B) | stride 16 (128 B) | stride 32 (256 B) |
|---|---|---|---|
| Task 2 (add RMW) | 39.0 ms (min 29.0) | 43.3 ms (min 30.9) | 47.0 ms (min 34.6) |
| store-only | 15.3 ms | 15.3 ms | 15.4 ms |
| LCG RMW | 130.2 ms | 126.7 ms | not run |

At T = 4 I measured **no penalty for 64 B padding**. The stride 8 vs 16 difference is inside the bimodal noise of Q2, even though `hw.cachelinesize` says 128 B, under which threads 0/1 and 2/3 share a line at stride 8.

**Measured, other thread counts:**
- T = 8: stride 8 = 107.8 ms vs stride 16 = 74.1 ms, **1.46x**.
- T = 2: 38.5 vs 31.2 ms (1.24x). The run distributions overlap here (30.2-56.6 vs 22.0-39.2 ms).

On this machine, 128 B padding is the only spacing that was never worse. Stride 32 is never better than stride 16, so I found no evidence of a pairing effect beyond 128 B.

**Stride 4 (32 B).** The 64 B model predicts 2-way sharing. On the 128 B line, all four threads land in line 0. With the Task 2 kernel, stride 4 is exactly as slow as stride 1 (87.5 vs 87.6 ms), which fits the 128 B line. With the store-only kernel, stride 4 is only 1.21x slower than stride 8, against 2.62x for stride 1. That fits neither simple model, so I do not claim to know the M4's internal coherence granule. Apple does not document it.

**How adjacent-line prefetchers pair lines (looked up, Intel 64 and IA-32 Optimization Reference Manual):** Intel's L2 "spatial prefetcher" treats memory as 128 B-aligned pairs of 64 B lines. When it fetches one line of a pair into L2, it also fetches the *buddy* line (the even/odd partner). If core A writes line 2k and core B writes line 2k+1 (64 B padding), B's L2 miss on 2k+1 also prefetches 2k, so an L2 copy of A's line is requested even though B never touches it. That costs A a snoop or invalidation and starts a second, prefetch-driven ping-pong. These secondary invalidations happen when:
1. the prefetcher is enabled;
2. the per-thread slots are 64 B apart but in the same 128 B pair (stride 8 here);
3. the slots miss in L2 often enough to trigger the prefetcher.

This is why Intel code and libraries such as folly pad to 128 B, and why C++ `std::hardware_destructive_interference_size` is 128 on some targets **(looked up)**. On the M4 the native line is already 128 B, so stride 16 is both "line-padded" and "prefetch-safe".

## Q4. Register promotion vs real memory stores (LLVM IR)

**Why a compiler hoists `buf[slot]`.** In `worker_plain_accumulate`, written as ordinary NumPy indexing (`buf[slot] = buf[slot] + (seed ^ i)`), the address `&buf[slot]` is loop-invariant, and the loop contains no other memory access or call that could alias it. In LLVM's memory model, non-atomic accesses may assume that no other thread races on them. So LICM's scalar promotion loads the value once before the loop, carries it in a register, and stores it once after the loop. The loop is then pure register arithmetic, and the vectorizer turns it into NEON code.

The IR confirms this (`results/llvm_worker_plain_accumulate.ll`): there is exactly **one** i64 store, `store i64 %.177.lcssa, i64* %8, align 8`, in block `B26.common.ret_crit_edge`, **outside** the loop, plus 14 `<2 x i64>` vector operations. The timing matches: plain = 8.19 ms, the same as the hand-privatized kernel (8.16 ms), and **10.7x faster than stride 1, because it never false-shares**. A naive kernel would therefore "prove" that false sharing does not exist.

**What actually keeps the store in the loop.** The handout says the loop's *data dependency* prevents promotion. On this compiler that is not true: the recurrence `x = x + f(i)` is carried in a register just as easily (the plain kernel above has exactly that dependency). What prevents promotion in `worker_strided_accumulate` is that the load and store are **relaxed atomics**, created by two small Numba `@intrinsic`s that call `builder.load_atomic(ptr, "monotonic", 8)` and `builder.store_atomic(..., "monotonic", 8)`. LLVM does not promote, merge or eliminate atomic accesses. The optimized loop body (`results/llvm_worker_strided_accumulate.ll`, block `B30`, which branches back to itself) is:

```llvm
%.163 = load atomic i64, i64* %1 monotonic, align 8
%.165 = xor i64 %.48.04, %arg.seed
%.166 = add i64 %.163, %.165
store atomic i64 %.166, i64* %1 monotonic, align 8
br i1 %exitcond.not, label %common.ret, label %B30
```

The audit counted 1 atomic load and 1 atomic store, with the store inside the loop (block `B30` is on a cycle). It found **no Python object or runtime calls** (`@Py*`, `@NRT_*`, `@numba_*`) in the core function. The only Python API use, `PyArg_UnpackTuple`, is in the separate CPython call wrapper. Every operand is a native `i64`.

On arm64, monotonic atomics compile to plain instructions (`results/asm_worker_strided_accumulate.txt`): `ldr x12, [x6, x11]` / `eor` / `add` / `str x10, [x6, x11]` / `b.ne`. There is no `dmb`, `ldar` or `stlr`, so the kernel measures ordinary L1 stores, not fence cost. This is the Python equivalent of `volatile` in C.

## Q5. CPython GIL masking vs `nogil=True`

Measured with stride 1, the same update, and the median of 3 runs (`gil_experiment` in the manifest):

| Variant | T = 1 | T = 2 | T = 4 | Per-thread times at T = 4 (ms) | Involuntary csw T = 1/2/4 |
|---|---|---|---|---|---|
| Pure Python bytecode threads, 300,000 updates/thread | 33.4 ms (111.4 ns/upd) | 66.5 ms (110.8 ns/upd) | 132.8 ms (110.6 ns/upd) | 132.7, 111.4, 122.6, 92.8 | 2 / 37 / 111 |
| Numba kernel **without** nogil, 30 M/thread | 28.8 ms | 65.3 ms | 85.1 ms | 63.5, 77.4, 21.5, 39.0 (staggered: serialized) | 2 / 18 / 61 |
| Numba kernel **with** nogil=True, 30 M/thread | 23.3 ms | 39.6 ms | 43.8 ms | 43.6, 43.8, 43.7, 43.7 (equal: concurrent) | 2 / 3 / 13 |

- **Pure Python.** With threads, wall time grows in proportion to the work: x1.99 at T = 2 and x3.98 at T = 4. The aggregate cost stays at about **111 ns per update**, so there is no speedup at all. False sharing cannot show up for two reasons:
  1. Only the thread holding the GIL runs bytecode, so two cores never store to the line at the same time. The line stays in M in one core for a whole GIL slice (the default switch interval is 5 ms) and moves only at each GIL handoff, a few hundred times per second instead of tens of millions.
  2. An update costs about 111 ns of interpreter work (boxing ints, dispatching opcodes, `__setitem__`). A coherence miss costs about 0.8 ns per update here (Q2), so it would be invisible even if it happened.

  The rising involuntary switch count (2 -> 37 -> 111) is the GIL handoffs. In this run the per-update cost did **not** get worse with more threads (111.4 -> 110.6 ns), contrary to the handout's expectation. It just did not get better, so for a fixed total amount of work the threaded version is no faster than one thread, plus the handoff overhead.
- **Numba without nogil.** The native kernel holds the GIL for the whole call, so the threads run one after another. The per-thread times are staggered (21.5 ... 77.4 ms), and the wall time at T = 4 (85.1 ms) is about 4 single-thread runs (one run takes 21.9-28.8 ms). That is serialization, not cache contention.
- **Numba with nogil=True.** All four threads finish within 0.2 ms of each other, so they really ran at the same time. The cost per thread rises from 0.78 ns (T = 1) to 1.46 ns (T = 4) per update. That increase is the hardware cache-line contention that the GIL hides.

In short: GIL contention is a software lock that serializes threads and keeps the cache line on one core. Hardware false sharing needs truly simultaneous stores from different cores, which only `nogil=True` native code produces.
