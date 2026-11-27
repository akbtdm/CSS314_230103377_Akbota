# Lab Practicum 01: Empirical Parallel Computing

| Student name | Student ID | Host CPU / SoC model | Operating system & kernel |
|---|---|---|---|
| Akbota Kutushova | 230103377 | Apple M4 (arm64), 10 cores (4P + 6E) | macOS 15.5 (24F74), Darwin 24.5.0 (xnu-11417.121.6, T8132) |

I measured every number below on this machine (2026-11-27). The raw console output is in `logs/`. Code was written with AI help, which the handout allows.

## Files

| File | Purpose |
|---|---|
| `task1_hw_audit.sh` | Runs `sysctl -a \| grep machdep.cpu` plus the `hw.*` topology, cache and SIMD keys, then builds and runs `clock_probe.c` |
| `clock_probe.c` | Measures the clock: one core runs a chain of dependent `add` instructions (1 add per cycle) |
| `task2_scaling.py` | Task 2 benchmark: counts primes below 20,000,000 with N real threads (Numba `nogil` kernels). Extra modes: `--imbalance`, `--demo N seconds` |
| `task3_race.c` | Task 3: 10 pthreads x 1,000,000 increments, first unlocked, then with a mutex, 10 runs each |
| `task3_python_gil.py` | Task 3 cross-check: the same test with CPython threads, where the GIL hides the race |
| `task4_amdahl.py` | Task 4: computes the Amdahl and Gustafson numbers from the Task 2 results |
| `logs/*.txt` | Console output saved with tee, plus the `top` / `ps -M` thread snapshot |
| `results/*.json` | Task 2 and Task 4 numbers in machine-readable form |

How to reproduce:
```sh
./task1_hw_audit.sh
python3 task2_scaling.py | tee logs/task2_output.txt
python3 task2_scaling.py --imbalance | tee logs/task2_imbalance.txt
clang -O2 -pthread -o task3_race task3_race.c && ./task3_race | tee logs/task3_race_c.txt
python3 task3_python_gil.py | tee logs/task3_python_gil.txt
python3 task4_amdahl.py | tee logs/task4_output.txt
```

---

## Task 1: Host silicon audit and Flynn's taxonomy

Source: `logs/task1_hw_audit.txt`

| Hardware property | Local measured value | Hardware property | Local measured value |
|---|---|---|---|
| CPU model & architecture | Apple M4, ARM64 (`machdep.cpu.brand_string: Apple M4`) | Base & max boost clock | macOS does not report them (0 `hw.cpufrequency` keys). Measured with `clock_probe.c`: **4.33 GHz** sustained on a P-core, **1.13-1.17 GHz** on an E-core under background QoS |
| Physical hardware cores | **10** (4 Performance + 6 Efficiency) | Logical cores (SMT/threads) | **10**: no SMT, so logical = physical |
| L1 data / instruction cache | P-core: L1d **128 KB**, L1i **192 KB**. E-core: L1d **64 KB**, L1i **128 KB** (per core) | L2 cache (per core / total) | P-cluster **16 MB** shared by 4 cores, E-cluster **4 MB** shared by 6 cores. Total **20 MB** |
| L3 shared last-level cache | **Not reported**: macOS has no `l3cache` sysctl key (0 found). The Apple SoC's system-level cache is not visible through sysctl | SIMD vector extensions | **ARM NEON / AdvSIMD** (128-bit), FP16, BF16, DotProd, I8MM, **SME + SME2**. No AVX (that is x86) |

Other values: cache line is 128 bytes and RAM is 16 GB (`hw.memsize = 17179869184`).

About the clock: Apple Silicon does not show its frequency through `sysctl`, so I measured it. A chain of dependent integer adds retires one add per cycle, so adds per nanosecond equals GHz. The P-core result is steady at about 4.33 GHz. The E-core figure was taken under `QOS_CLASS_BACKGROUND`, the only way to pin a thread to E-cores from user space, and macOS also lowers the clock for background work. So 1.1-1.2 GHz is the E-core clock under background QoS, not its maximum.

### Q1.1: Flynn's taxonomy on this machine
- **SISD** (single instruction stream, single data stream). One thread runs scalar code, for example Task 2 with N = 1. One P-core runs one instruction stream (load, `udiv`/`msub`, compare, branch) on one number at a time. A single-threaded compiler pass works the same way.
- **SIMD** (single instruction stream, multiple data streams). Every core has 128-bit NEON registers (`hw.optional.neon: 1`), so one instruction can process, say, 16 bytes or 4 floats at once. The numpy sieve in `task2_scaling.py` (`s[i*i::i] = False`, `s.sum()`) and vectorized matrix math run in this mode. The M4 also has SME/SME2 matrix units for 2-D SIMD (outer products).
- **MIMD** (multiple instruction streams, multiple data streams). The 10 independent cores each run their own instruction stream on their own data. Examples: Task 2 with N >= 2, the 10 racing threads in Task 3, and the OS scheduling unrelated processes on different cores at the same time. `logs/task2_thread_activity.txt` shows 8 worker threads in state `R` (running) at the same moment, with the process at 421 % CPU.

---

## Task 2: Multi-thread scaling benchmark and the contention wall

- **Workload algorithm:** count primes below **20,000,000** by trial division (odd divisors up to sqrt n). Work is split into blocks of 2,000 numbers dealt round-robin to the threads. Every run is checked against a numpy sieve (1,270,607 primes). I raised the handout's example of 5,000,000 to 20,000,000 because at 5,000,000 the N = 1 time was only 0.20 s, and thread start-up would have distorted the results.
- **Language & runtime:** Python 3.13.5 + Numba 0.61. The kernel is `@njit(nogil=True)` and runs inside N `threading.Thread` objects. These are real OS threads running in parallel, because the kernel releases the GIL.

Source: `logs/task2_output.txt`

| Threads (N) | Run 1 (s) | Run 2 (s) | Run 3 (s) | Avg time T_N (s) | Speedup S_N = T1/T_N | Efficiency E_N = S_N/N |
|---|---|---|---|---|---|---|
| N = 1 (baseline) | 1.4293 | 1.4444 | 1.4413 | **1.4383** | 1.00x | 100.0 % |
| N = 2 | 0.7385 | 0.7852 | 0.7194 | **0.7477** | 1.92x | 96.2 % |
| N = 4 | 0.3994 | 0.4056 | 0.4096 | **0.4048** | 3.55x | 88.8 % |
| N = 8 | 0.3415 | 0.3563 | 0.3774 | **0.3584** | 4.01x | 50.2 % |
| N = 16 | 0.3299 | 0.3412 | 0.3301 | **0.3337** | 4.31x | 26.9 % |
| N = 32 | 0.3438 | 0.3332 | 0.3282 | **0.3351** | 4.29x | 13.4 % |

Terminal evidence: the handout asks for a screenshot. I captured `top` and `ps -M` as text instead (`logs/task2_thread_activity.txt`), recorded while `task2_scaling.py --demo 8 6` was running. A screen capture would have recorded the student's whole desktop. The log shows `python3.13  %CPU 421.3  #TH 18/8 running` and 8 OS threads in state `R` at about 33 % CPU each. If the portal needs an image, take a screenshot of `top` while running `python3 task2_scaling.py --demo 8 20`.

### Q2.1: Scaling saturation and hardware contention
- **Inflection point: N = 4 -> 8.** Up to 4 threads the scaling is almost linear (1.92x at N = 2, 3.55x at N = 4). Going from 4 to 8 threads doubles the thread count but cuts the time by only 11 % (0.405 s -> 0.358 s), and efficiency falls from 88.8 % to 50.2 %. Speedup **peaks at N = 16 (4.31x)** and **reverses slightly at N = 32 (4.29x)**. The 16 and 32 results are equal within run-to-run noise (about +/- 0.01 s).
- **Why at 4: physical vs. logical cores.** The M4 has only **4 performance cores**. The other 6 are efficiency cores, and there is no SMT (10 logical = 10 physical). Threads 5-10 therefore run on E-cores. Those cores are narrower and run at a lower clock (the P-core is 4.33 GHz in Task 1), so each one does much less of this division-heavy loop per second. The data show this: with all 10 cores busy the slowest thread finishes in 0.326 s (`logs/task2_imbalance.txt`), only about 4.4x faster than T1. So the 6 E-cores together added less than one P-core's worth of throughput.
- **Beyond 10 threads: OS scheduling overhead.** For N = 16 and 32 there are more threads than cores, so macOS time-slices them. There is no extra hardware to use, so speedup stays flat at about 4.3x. Creating, switching and joining the extra threads costs a little, which explains the small drop at N = 32. The imbalance log shows that the fastest and slowest threads finish within 0.244-0.327 s at N = 16, so threads that finish early wait about 11 % of the wall time.
- **Cache thrashing and false sharing: not a factor.** Each thread works on a few integers in registers and writes one result at the end. There is no shared array, and the working set is tiny next to the 64-128 KB L1d. So the saturation does not come from memory or cache contention.
- **Thermal throttling: not observed.** The three N = 1 runs agree within 1 % (1.429-1.444 s). The clock probe gave the same 4.33 GHz in all 3 repeats of two separate audit runs. The runs are short (under 1.5 s each, about 15 s total), so the chip did not have time to heat-soak.

---

## Task 3: The unsynchronized shared counter (race condition)

There are 10 pthreads. Each runs `counter++` 1,000,000 times on one shared `volatile int`, with no lock or atomic. Expected total: 10,000,000. I implemented this in **C** (`task3_race.c`, `clang -O2`) because CPython's GIL makes the race disappear (see the cross-check below). `logs/task3_unlocked_worker_asm.txt` confirms the compiled loop is a separate load, add and store on every iteration:
```
ldr w10, [x9, _counter@PAGEOFF]   ; Load
add w10, w10, #1                  ; Add
str w10, [x9, _counter@PAGEOFF]   ; Store
```

Source: `logs/task3_race_c.txt`

| Run | Measured output | Error (10^7 - Act) | Run | Measured output | Error (10^7 - Act) |
|---|---|---|---|---|---|
| #1 | 1,022,730 | 8,977,270 | #6 | 1,252,187 | 8,747,813 |
| #2 | 1,269,803 | 8,730,197 | #7 | 1,218,063 | 8,781,937 |
| #3 | 1,295,213 | 8,704,787 | #8 | 1,373,477 | 8,626,523 |
| #4 | 1,302,439 | 8,697,561 | #9 | 1,561,162 | 8,438,838 |
| #5 | 1,341,819 | 8,658,181 | #10 | 1,474,321 | 8,525,679 |

Every run is wrong, and by a different amount each time. Between 84 % and 90 % of the increments were lost.

**Python cross-check** (`logs/task3_python_gil.txt`): the same test in CPython 3.13.5, where the GIL is enabled, gave exactly **10,000,000 in all 10 runs**, with an error of 0. With the GIL only one thread runs Python bytecode at a time. CPython switches threads only at loop back-edges or calls, never between the load and the store of `counter += 1`. So in Python the bug exists in principle but never shows up.

### Q3.1: The read-modify-write (RMW) window
`counter++` is not one machine operation. It is three: **Load** the value from memory into a register (`ldr`), **Add** 1 in the register (`add`), and **Store** it back (`str`). Between the load and the store, the value lives only in that core's private register. Suppose core A loads 500 and core B also loads 500 before A stores. Both compute 501 and both store 501, so two increments produced one, and one update is **lost**. Each core works on its own copy of the 128-byte cache line. The coherence protocol keeps the line consistent, but it cannot merge two stale read-modify-write sequences. Whichever store lands last overwrites the other.

With 10 cores hammering one line in a tight loop, a core usually stores a value computed from an old load. Often it overwrites hundreds of other cores' increments at once. That is why the result was only about 1.0-1.6 million instead of 10 million. The amount lost depends on exact timing, so every run gives a different number.

### Q3.2: The synchronization penalty
I wrapped the counter in a mutex (`pthread_mutex_lock` / `unlock`). Results are from `logs/task3_race_c.txt`, mean of 10 runs:

**Unlocked = 3.07 ms vs. Locked = 141.19 ms** (about **46x slower**, but correct: 10,000,000 in all 10 runs).
Python cross-check: unlocked 186.4 ms vs. `threading.Lock` 707.2 ms (3.8x slower).

Why the correct version is so much slower:
1. **Serialization.** Only one thread at a time may be inside the critical section. The 10 cores are back to doing the 10 million increments one after another, plus the time to hand the lock from core to core.
2. **Atomic instructions and memory barriers.** Each lock and unlock is an atomic read-modify-write (an LSE `cas`/`ldadd` or `ldaxr`/`stlxr` on ARM) with acquire/release ordering. That costs far more than a plain `add`, and it runs 20 million times (10 million locks plus 10 million unlocks).
3. **Cache-line ping-pong.** The lock word and the counter keep moving between cores' L1 caches. Each transfer goes through the cluster L2, or crosses between the P and E clusters, taking tens of nanoseconds instead of the roughly 1 ns of an L1 hit.
4. **Contention in the kernel.** When the lock is taken, waiting threads spin and then may sleep and be woken by the OS, which adds system-call and scheduling cost.

The unlocked version is fast only because it is broken: each core mostly updates its own stale copy and throws away the others' work. The right fix in real code is to cut contention, not just add locks. Each thread sums into a private local variable and adds it to the total once at the end, or uses an `atomic_fetch_add`.

---

## Task 4: Empirical Amdahl verification vs. Gustafson scaling

Source: `logs/task4_output.txt`, using the T1 and T2 averages from the Task 2 table.

1. **Parallel fraction:** p = 2 x (T1 - T2) / T1, with **T1 = 1.4383 s** and **T2 = 0.7477 s**. Calculated **p = 0.9603**.
2. **Amdahl ceiling:** sequential fraction (1 - p) = **0.0397**. Max theoretical speedup (N -> infinity) = 1 / (1 - p) = **25.19x**.
3. **Amdahl 64-core projection:** S64 = 1 / [(1 - p) + p/64] = **18.28x**.
4. **Gustafson scaled speedup:** S = (1 - p) + 64p = **61.50x**.

How well the Amdahl model with p = 0.9603 matches the measured speedup:

| N | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| Amdahl prediction | 1.00x | 1.92x | 3.57x | 6.26x | 10.03x | 14.35x |
| Measured (Task 2) | 1.00x | 1.92x | 3.55x | 4.01x | 4.31x | 4.29x |

The model matches at N = 4 (3.57x predicted vs. 3.55x measured) but overshoots from N = 8 upward. The prime count is almost completely parallel. Its real limit on this laptop is hardware: only 4 fast cores and 10 cores in total, which the formula ignores. The roughly 4 % "serial fraction" estimated from T2 also includes thread start-up and the uneven split of work, not just serial code.

### Q4.1: Strong vs. weak scaling
- **Amdahl (strong scaling)** keeps the problem size fixed: primes below 20,000,000. As N grows, the parallel part p/N shrinks toward zero, but the serial 3.97 % does not shrink. With 64 cores the parallel part shrinks to 0.9603/64 = 0.015 of T1, but the serial part stays at 0.0397 of T1. Serial code is then about 73 % of the remaining run time, so it dominates. The speedup flattens at 18.3x and can never pass 25.2x.
- **Gustafson (weak scaling)** grows the problem with the machine: 64 cores get 64 times as much work (for example, a much larger prime range). The serial part (start-up, combining results) stays about the same size, while the parallel part grows 64-fold. The serial share becomes tiny, so scaled speedup = 0.0397 + 64 x 0.9603 = 61.5x, close to linear.
- **Same algorithm, different questions.** Amdahl asks "how much faster can I finish *this* job?" and the fixed serial part sets a hard ceiling. Gustafson asks "how much *more* work can I finish in the same time?" and extra cores keep buying more throughput. On a real machine both are capped by the hardware. My Task 2 data stops at about 4.3x on 4 P-cores + 6 E-cores, well below both projections, so a 64-core node would need 64 equally fast cores for either prediction to hold.

---

## Deviations and notes
- **Problem size:** primes below 20,000,000 instead of the 5,000,000 example, so that T1 (1.44 s) is not dominated by thread start-up. The handout's "e.g." allows this.
- **Task 3 in C:** the handout allows any language. A pure-Python version cannot show the hardware race because of the GIL. Its results are reported as a cross-check.
- **Screenshot:** a text log of `top` / `ps -M` replaces the image (see Task 2).
- **Clock and L3:** macOS on Apple Silicon reports neither. The clock was measured empirically, and L3 is marked "not reported" rather than filled with a spec-sheet number.
- **Submission:** the handout asks for a .docx/PDF uploaded to the course portal. This README holds all the content; exporting and uploading it is left to the student.
