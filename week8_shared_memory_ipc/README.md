# Week 8: Zero-Copy Shared Memory & Concurrent IPC

**Student:** Akbota Kutushova, ID 230103377
**Course:** Systems Programming (Sufyan Mustafa Bin Uzayr)
**Machine:** Apple M4: 10 cores (4 P + 6 E, no SMT), `hw.cachelinesize` = 128 B, L1d 128 KB (P) / 64 KB (E), L2 16 MB (P) / 4 MB (E), 16 GB RAM, 16 KiB pages, macOS 15.5, Python 3.13.5, numpy 2.1.3, numba 0.61.0 (`logs/hw_info.log`).

Every number in this report was measured on this machine in a single `./run_all.sh` run on 2026-11-27 (about 2.5 minutes, benchmarks run one at a time). The raw console output for each milestone is in `logs/`. `results/report.txt` holds all of that output plus the CPU model and the verification token, and `results/shm_lab_230103377.zip` is the submission archive.

## Files

| File | Purpose |
|---|---|
| `lab_constants.py` | Constants derived from the student ID (section 1) and the segment names |
| `slab_allocator.py` | Milestone 1: 512 B slab, `struct.pack_into` on a zero-copy memoryview, XOR-16 checksum, a validating worker |
| `ring_buffer.py` | Milestone 2: lock-free SPSC ring in `shm_ring_230103377`, throughput and drop statistics, negative test (`--inject-gap`) |
| `race_inducer.py` | Milestone 3, experiment 1: race matrix for N = 1..16, unsynchronized vs `multiprocessing.Lock` |
| `cache_collision.py` | Milestone 3, experiment 2: false sharing, cases A and B from the handout plus padded cases at 64 B and 128 B |
| `matrix_sifter.py` | Milestone 4: 200 MB zero-copy parallel reduction with `ru_maxrss` logging, and a copying counter-example |
| `verify_submission.py` | Section 6 self-audit harness (reconstructed, see Deviations) |
| `run_all.sh` | Runs everything, tees output to `logs/`, builds `results/report.txt` and the ZIP |

Reproduce with `./run_all.sh`.

## 1. Student constants

UID = 230103377:

| Constant | Formula | Value |
|---|---|---|
| OFFSET_PAD | (UID % 7) * 8 | 230103377 % 7 = 0, so **0 bytes** |
| RING_CAPACITY | 8 + (UID % 5) * 4 | % 5 = 2, so **16 slots** |
| MAGIC_BYTE | 0xA0 + UID % 16 | % 16 = 1, so **0xA1** |
| TARGET_SEED | UID ^ 0x5A5A5A5A | **1475166987** (0x57ED430B) |

Because OFFSET_PAD is 0 for this ID, the record starts at byte 0 of the slab. The code still applies the pad everywhere.

## 2. Milestone 1: raw memory slab (`logs/m1_slab_allocator.log`)

The layout is big-endian with no padding (`>BIQ4d` followed by `>H`): magic@0, writer_pid@1, timestamp_ns@5, payload_floats@13, checksum_crc@45, for a 47 B record. The master creates `shm_slab_230103377` with `size=512` and writes into `shm.buf` directly with `struct.pack_into`. No `bytes` copies are made: the checksum is computed by iterating a memoryview slice, and a memoryview slice is a view, not a copy. A spawned worker attaches by name (`track=False`), validates the magic byte and the checksum, and prints the record.

Measured output (valid record): magic 0xA1 OK; checksum 0xAD2C stored, 0xAD2C recomputed, OK; writer_pid = the master's PID; payload 3.14159265359, 2.71828182846, 1.41421356237, 230.103377. The record was 26 ms old when the worker read it, which is mostly the time to spawn the worker.

Two negative tests also passed:
- Flipping one payload byte gave recomputed checksum 0x522C, and the worker exited with code 3.
- Writing a wrong magic byte made the worker exit with code 2.

The segment is unlinked in `finally`.

**Checksum definition.** The handout asks for a uint16 "XOR sum of all preceding bytes". A plain XOR of bytes only produces 8 bits, so the 45 preceding bytes are XORed as big-endian 16-bit words, with the odd last byte padded with 0x00.

**macOS note.** The kernel rounds the mapping up to one 16 KiB page (`shm.size` = 16384). The logical slab is still the first 512 B.

## 3. Milestone 2: lock-free SPSC ring buffer (`logs/m2_ring_buffer*.log`)

The segment `shm_ring_230103377` is 524 B: a 12 B header (head, tail, dropped_count) followed by 16 slots of 32 B (`<QQQQ`: sequence_id, timestamp_ns, 16 B of data). The three cursors are read and written through `shm.buf[:12].cast('I')`. Each access is one aligned native 32-bit load or store, which is single-copy atomic on arm64. Each cursor has exactly one writer: head is written only by the consumer, tail and dropped_count only by the producer. No lock is used.

The producer writes the slot first and stores `tail` second (publish order). It treats the ring as full when `(tail+1) % C == head`. When full, it increments `dropped_count` and calls `time.sleep(0.0001)`, then retries. The consumer treats the ring as empty when `head == tail` and spins with `os.sched_yield()` until a record is published. It asserts that every `sequence_id` is exactly the expected next value and that the 16 data bytes match.

| Trial | Wall-clock (record 0 to last record consumed) | Throughput | Final dropped_count | Records that met a full ring | Latency p50 / p99 | Total incl. spawning 2 processes |
|---|---|---|---|---|---|---|
| 1 | 239.20 ms | 418,052 rec/s | 1661 | 4578 | 2.58 / 7.67 µs | 276 ms |
| 2 | 216.31 ms | 462,296 rec/s | 1490 | 4055 | 2.29 / 6.25 µs | 249 ms |
| 3 | 226.15 ms | 442,190 rec/s | 1556 | 4214 | 2.58 / 6.83 µs | 259 ms |
| **median** | **226.15 ms** | **442,190 rec/s** | **1556** | | | |

All 3 × 100,000 records arrived strictly in order.

- **What dropped_count means here.** The consumer must see a gap-free sequence, so a "drop" is a full-buffer stall: one backoff sleep. The record is retried, not lost. The producer is faster than the consumer, so the 15 usable slots are often full.
- **Negative test.** `--inject-gap` skips sequence id 50000. The consumer aborted with `AssertionError: MEMORY CORRUPTION: expected sequence_id 50000, read 50001`. The master then stopped the producer, which was blocked on a full ring.

**Memory ordering.** Python has no memory-fence primitive. arm64 is weakly ordered, so in principle a plain store to `tail` could become visible before the slot stores. In CPython, hundreds of instructions separate the two stores, and the consumer re-checks both the sequence id and the payload. No mis-ordering appeared in 300,000 records. A C implementation would use a store-release on `tail` and a load-acquire on the consumer side. A second effect is that `head` and `tail` share one 128 B line, so every publish and every release moves that line between the two cores. This is false sharing on the control words themselves.

## 4. Milestone 3, experiment 1: race inducer matrix (`logs/m3_race_inducer.log`)

The counter is `multiprocessing.Array('i', 1)`. N spawned workers each run 1,000,000 iterations of `val = a[0]; val += 1; a[0] = val`. In the unsynchronized mode they do this on the raw buffer (`arr.get_obj()`), with no lock at all. In the Lock mode the same three statements are wrapped in an explicit `multiprocessing.Lock()`. All workers start together at a `Barrier`, and runtime is measured from the barrier release to the last `join`. The unsynchronized mode ran 3 times and the count shown is the median; the Lock mode ran once.

| Workers (N) | Expected Total | Observed Count (Unsynchronized) | Update Loss Rate (%) | Runtime: Unsynchronized | Runtime: multiprocessing.Lock |
|---|---|---|---|---|---|
| 1 | 1,000,000 | 1,000,000 | 0.00% | 58.4 ms | 720.8 ms |
| 2 | 2,000,000 | 1,019,036 | 49.05% | 63.7 ms | 7,563.0 ms |
| 4 | 4,000,000 | 1,018,131 | 74.55% | 69.8 ms | 15,124.0 ms |
| 8 | 8,000,000 | 1,430,135 | 82.12% | 127.3 ms | 31,036.3 ms |
| 16 | 16,000,000 | 1,878,874 | 88.26% | 232.9 ms | 66,326.2 ms |

The three unsynchronized counts were close each time: N=2: 1,014,397 / 1,021,622 / 1,019,036; N=16: 1,987,529 / 1,846,911 / 1,878,874. Every Lock run produced exactly N × 1,000,000.

**Analysis**
- **Loss grows with N.** It approaches 1 − 1/N: 49% for 2 workers and 75% for 4, where all workers fit on separate cores. When workers overlap fully, almost every increment is overwritten by another worker's stale `val`, so the final count is close to one worker's worth (about 1.0 M for N = 2 and 4).
- **Beyond 10 cores.** At N = 8 and 16 the count rises to 1.4 M and 1.9 M. Workers no longer all run at the same moment (16 processes on 10 cores, with time slicing), so some of their increments land without interference.
- **Cost of the Lock.** It makes the result correct, but it costs 0.72 µs per increment uncontended (N = 1, 12× the unsynchronized loop). Under contention it costs about 3.8 to 4.1 µs per increment of total work: 7.56 s / 2 M, and 66.3 s / 16 M. On macOS, `multiprocessing.Lock` is a named POSIX semaphore, so each acquire/release is a system call, and contended handoffs are serialized through the kernel. Lock runtime therefore scales linearly with total work, and adding workers gives no speed-up. At N = 16 the correct version is 285× slower than the incorrect one.

## 5. Milestone 3, experiment 2: false sharing (`logs/m3_cache_collision.log`)

The array is `multiprocessing.RawArray('q', ...)`. Two processes each increment their own slot. The array base was 128-byte aligned in all 45 runs (`base addr mod128=0`). The table shows the median of 3 runs of the slowest worker's loop time.

**A note on the handout's Case B.** `arr[7]` occupies bytes 56 to 63, so it sits in the same line as `arr[0]` for both 64 B and 128 B lines. Case B is therefore not isolated. To test the handout's 64 B assumption and the M4's 128 B line, I added Case C (`arr[0]` / `arr[8]`, 64 B apart) and Case D (`arr[0]` / `arr[16]`, 128 B apart).

| Engine | Solo (1 process) | **A** 0/1 (8 B apart) | **B** 0/7 (56 B, handout) | C 0/8 (64 B) | D 0/16 (128 B) | A vs D |
|---|---|---|---|---|---|---|
| python, handout loop `arr[i] += 1`, 10 M | 643.4 ms | **677.0 ms** | 673.7 ms | 656.0 ms | 654.6 ms | 1.03× |
| numba, plain load+store, 100 M | 90.4 ms | 94.6 ms | 93.5 ms | 93.9 ms | 92.1 ms | 1.03× |
| numba, atomic fetch-add (LDADD), 10 M | 15.9 ms | **64.2 ms** | **66.2 ms** | 16.3 ms | 16.2 ms | **3.96×** |

**Case A vs Case B, the handout's deliverable, with the literal Python loop:** A = 677.0 ms, B = 673.7 ms. Both are 5% slower than a single process, and about 3% slower than the truly isolated 64 B and 128 B cases. The degradation is measurable but small, for three reasons:

1. **Interpreter overhead hides it.** CPython spends about 64 ns per increment on bytecode and ctypes boxing. That is far more than the time to move a cache line, so the extra coherence traffic is only a few percent of the loop.
2. **Plain stores barely notice it, even compiled.** Each JIT-compiled plain store still costs about 0.9 ns. The M4 core keeps stores in its store buffer and feeds the following load directly from it (store-to-load forwarding). It merges consecutive stores to the same line, so the loop almost never waits for line ownership (A is only 1.03× D).
3. **Atomics show the real coherence cost.** A hardware atomic read-modify-write, which is what `std::atomic` counters and lock words do, must own the line every time. Same-line cases A and B become **4.0 to 4.2× slower** (about 6.4 to 6.6 ns per op versus 1.6 ns). This is the MESI ping-pong the handout describes.

**64 B vs 128 B.** Case C (64 B apart) is already as fast as Case D and as Solo (16.3 vs 16.2 vs 15.9 ms). So on this M4, placing two hot variables 64 B apart was enough to remove the contention, even though `hw.cachelinesize` reports 128 B. The measurement shows that coherence was resolved at 64-byte granularity for this L1 workload. Padding to 128 B (`hw.cachelinesize`, and the line size Apple documents for the L2) costs nothing extra here and is the portable choice on Apple silicon.

The two processes are not pinned to cores (macOS has no CPU affinity API), so a pair may run on P cores, E cores, or one of each. This explains the run-to-run spread for case A (55.0 to 72.8 ms).

## 6. Milestone 4: parallel zero-copy 200 MB sifter (`logs/m4_matrix_sifter.log`)

`shm_sift_230103377` holds 200,000,000 B, filled from `np.random.default_rng(1475166987).integers(0, 256, dtype=uint8)` in 8 MiB chunks (123 ms). There are K = `os.cpu_count()` = 10 workers. Each one attaches with `SharedMemory(name=...)` and wraps only its own 20 MB slice with `np.ndarray(buffer=shm.buf, offset=lo)`, which makes no copy. It sums the elements with `val % 7 == 0` using preallocated 1 MiB scratch buffers. Results go back through a second shared memory segment (`shm_sres_...`). `ru_maxrss` is reported in bytes on macOS.

| | Value |
|---|---|
| Total sum of val with val % 7 == 0 | **3,641,372,777**, equal to the serial reference |
| Count of matching elements | 28,894,654 (14.45% of 200 M; the expected fraction for uniform bytes is 37/256 = 14.45%) |
| Serial scan in the master | 529.5 ms |
| Parallel, 10 processes (barrier to last join) | 138.8 ms, a **3.82×** speed-up |
| Worker `ru_maxrss` before touching data | about 36.5 to 37.7 MB each (interpreter + numpy), 369.9 MB summed |
| Worker `ru_maxrss` increase during the scan | **22.2 to 22.5 MB each**, 222.9 MB summed (20 MB slice + 2 MB scratch) |
| Master peak `ru_maxrss` | 274.6 MB (200 MB segment + about 31 MB interpreter + scan buffers) |
| Counter-example: `bytes(shm.buf[a:b])` copy in every worker | increase ≈ 42 MB each, **421.3 MB** summed |

**RSS invariant.** The 200 MB of data are counted once in the master and once in total across the workers. Each worker's RSS grew only by its own 20 MB slice plus 2 MB of scratch. The sum of increases (222.9 MB) therefore stays about 200 MB, and the pages are the same physical pages as the master's. `ru_maxrss` counts a shared page in every process that touches it, so adding up raw RSS (592.8 MB for the workers, plus the master's 274.6 MB) double-counts the shared data. It also counts 10 × 37 MB of Python runtime that every process has regardless of the data.

The copying variant shows what serialization would add: roughly twice the increase (421.3 MB), because each worker keeps a private 20 MB duplicate next to the shared pages it read.

**Why the speed-up is only 3.82×.** Each worker's scan took 81 to 131 ms, against about 53 ms per 20 MB in the serial run. The causes are the 6 E cores (slower), page faults on the first touch of 1,221 16-KiB pages per worker, and shared memory bandwidth.

## 7. Self-verification and deliverables

`verify_submission.py` passed 31/31 checks:
- constants
- record offsets 0/1/5/13/45 and the 47 B record length
- XOR-16 checksum round trip on a real shared memory slab
- detection of a single-bit flip
- ring size 524 B = 12 + 32·16
- all segment names under macOS's 31-character limit
- **no leaked segments**: attaching to each name raises `FileNotFoundError`
- no `Queue` / `Pipe` / `Manager` / `pickle` in the deliverables

**Verification token: `AUTH-CAE348D06F2C`** (reproducible; `logs/verify_submission.log`).

After the run, `pgrep -f multiprocessing` found 0 leftover processes (`logs/cleanup_check.log`).

## Deviations and notes

1. **Verification harness reconstructed.** The handout's `verify_submission.py` listing (and the ring-buffer "structural snippet") is cut off after its first line in the PDF itself; the text is clipped in the source document, not just in the extraction. The original algorithm is therefore unknown. I wrote `run_self_audit()` from the checks the handout names. The `AUTH-…` token comes from this reconstruction (sha256 over the UID constants and the deliverable sources). **It is not the instructor's official signature.**
2. **What the IPC ban covers here.** With the spawn start method, `multiprocessing` itself pickles the `Process(args=...)` that start each child. In this lab those arguments are only small control values (worker id, mode) and the handles for `mp.Array`, `Lock` and `Barrier` that the handout requires. Every data path (slab record, ring records, counters, the 200 MB array, worker results) goes through shared memory. Neither `Queue`, `Pipe`, `Manager` nor `pickle` is used.
3. **dropped_count** counts full-buffer stalls (backoff sleeps). Records are retried rather than discarded, because the consumer must see every sequence id in order.
4. **Extra cases.** Experiment 2 adds Cases C and D (64 B and 128 B padding) and two numba engines. The numba plain loop uses 100 M increments so that the two processes overlap in time. The handout's Python loop and its 10 M increments are run unchanged.
5. **No CPU pinning** on macOS, so P-core and E-core placement adds noise. Medians of 3 runs are reported.
6. The `multiprocessing.Lock` column was measured once per N (66 s at N = 16) to keep total runtime under 10 minutes.
