# Monte Carlo π in Java: Phantom Bug, Synchronization Trap, Reduction

Student: Akbota Kutushova (230103377). Machine: Apple M4, 10 cores (4 performance + 6 efficiency, no SMT), OpenJDK 27.0.1.1.
Code: [PiLab.java](PiLab.java). Full console output: [output.txt](output.txt). Run with:

```bash
javac PiLab.java && java PiLab
```

## Part 1: The Phantom Bug (4 threads, `totalHits++`, 50M points)

| Run | π | Hits counted (expected ≈ 39,269,908) |
|---:|---:|---:|
| 1 | 0.964462 | 12,055,781 |
| 2 | 0.965238 | 12,065,476 |
| 3 | 0.964309 | 12,053,868 |
| 4 | 1.024033 | 12,800,408 |
| 5 | 0.983358 | 12,291,972 |

The handout predicts π ≈ 1.8–2.4; on this machine it is even worse, about 0.96–1.02, so roughly 69% of all hits are lost. The reason is the JIT. `totalHits` is a plain (non-volatile) static, so HotSpot is allowed to keep it in a register for the whole loop and write it back to memory only at the end. Each thread then writes its own count (about 9.8 million) over the others, and whoever writes last wins. A race does not just lose a few updates: without synchronization the compiler may legally make the shared variable effectively private.

## Part 2: The Synchronization Trap (50M points)

| Version | π | Time (ms) | vs. single thread |
|---|---:|---:|---:|
| Single-threaded plain loop | 3.141617 | 60 | 1.00x |
| 4 threads, `AtomicLong.incrementAndGet()` | 3.141588 | 463 | 7.72x slower |
| 4 threads, `synchronized` | 3.141588 | 1306 | 21.77x slower |

## Part 3: OpenMP-style reduction (100M points, median of 3 runs)

| Threads (T) | Runtime (ms) | Speedup vs. 1 thread | Efficiency |
|---:|---:|---:|---:|
| 1 (baseline) | 129.2 | 1.00x | 100.0% |
| 2 | 63.3 | 2.04x | 102.1% |
| 4 | 34.4 | 3.75x | 93.8% |
| 8 | 30.4 | 4.25x | 53.1% |
| 16 | 28.3 | 4.56x | 28.5% |
| 32 | 25.3 | 5.10x | 16.0% |

Each thread counts into a local `long` and writes it once into its own slot of a partial-sums array. The slots are 16 longs (128 bytes, one M4 cache line) apart, so the threads never share a line.

## Questions

**1. Why didn't 16 threads run twice as fast as 8?**
This machine has 10 cores and no hyper-threading, so it can run at most 10 threads at once. 16 threads just take turns on the same 10 cores, and the extra threads add scheduling and context-switch work instead of compute. On top of that, the cores are not equal. 4 are performance cores, and 6 are slower efficiency cores. That is why scaling is close to ideal up to 4 threads (3.75x), then flattens: 8 threads reach only 4.25x, and 16 threads add just 7% more (4.56x).

**2. Why was the synchronized version slower than one core?**
In the single-threaded loop the counter sits in a register, and an increment costs about one cycle. In Part 2 every hit, about 39 million of them, needs exclusive ownership of the cache line holding the counter. With `synchronized` it also needs a lock acquire and release, which become contended and inflated when 4 threads fight over the lock. The line keeps moving between the cores, and the threads spend almost all their time waiting for it instead of generating random points. Measured: `AtomicLong` is 7.7x slower than one thread, and `synchronized` is 21.8x slower. Correctness came back (π = 3.1416), but the parallelism did not.
