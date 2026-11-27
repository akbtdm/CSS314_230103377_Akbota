# Parallel Computing: Class 0 Diagnostic

**Student:** Akbota Kutushova, ID 230103377
**Language:** Python 3.13.5
**Machine:** Apple M4, 10 cores (4 performance + 6 efficiency), 16 GB RAM, macOS (Darwin 24.5.0)

All code answers are in [`diagnostic.py`](diagnostic.py). Run it with `python3 diagnostic.py`. The whole script takes about 0.13 s. The real output is shown here:

```
Task 1:
  x = 5, y = 2   -> x=5, y=2
  x = x + y      -> x=7, y=2
  y = x * 2      -> x=7, y=14
  x = y - x      -> x=7, y=14
Task 2: buggy=50 fixed=150
Task 3: 10x10=100 1000x1000=1000000 (10 x 1,000,000 = 10,000,000 by formula)
Task 6: linear (index, comparisons)=(6, 7) binary=(6, 2)
Task 7: chunked max of 200,000 ints = 999991014 (matches full max: True)
Task 8: {'A': 4, 'B': 3, 'C': 2, 'D': 1} (Counter agrees: True)
Task 9: A+B=[[6, 8], [10, 12]] AxB=[[19, 22], [43, 50]] multiplications=8
Task 15: A,B,D,E 10x -> 59s to 50.9s, speedup 1.159x; C 10x instead -> 14.0s, speedup 4.21x
Task 16: {'n': 100001, 'min': 0, 'max': 100, 'mean': 50.07835921640783, 'median': 50.0, 'ge90': 10828, 'lt40': 39452}
         matches sort-based reference: True
python3 diagnostic.py  0.13s user 0.00s system 99% cpu 0.134 total
```

---

## Part A: Programming

### Task 1: Trace the code

| Statement | x | y |
|---|---|---|
| `x = 5` | 5 | – |
| `y = 2` | 5 | 2 |
| `x = x + y` | 7 | 2 |
| `y = x * 2` | 7 | 14 |
| `x = y - x` | 7 | 14 (14 − 7 = 7) |

**Output:** `7 14`

### Task 2: Find the bug

- **What it calculates:** `total = numbers[i]` replaces `total` on every pass, so the loop only keeps the last element. It prints **50**.
- **What was intended:** the sum of the list, which is **150**.
- **Fix:** accumulate instead of assigning. Write `total += numbers[i]`, or loop over the values directly:

```python
total = 0
for value in numbers:
    total += value
print(total)   # 150
```

The built-in `sum(numbers)` also works. The test run above prints `buggy=50 fixed=150`.

### Task 3: Nested loops

`print()` runs once per (i, j) pair, so the count is outer × inner.

- (a) 10 × 10 = **100** times
- (b) 1000 × 1000 = **1,000,000** times
- (c) 10 × 1,000,000 = **10,000,000** times

The script counts (a) and (b) for real and gets 100 and 1,000,000. Case (c) is computed from the formula so the run stays short. Printing is very slow because every call is an I/O system call to the terminal. At 10 million lines, the output would take far longer than the arithmetic.

## Part B: Algorithms and complexity

### Task 4: Which is faster?

- **Algorithm A** (one scan): about **n** operations, which is O(n).
- **Algorithm B** (every element against every element): about **n²** operations, or n(n−1)/2 if each pair is compared only once. That is O(n²).
- **B grows much faster.** When n doubles, A's work doubles and B's work quadruples.
- **n = 1,000,000:** A needs about 10⁶ steps, which takes milliseconds even in Python. B needs about 10¹² steps (or 5 × 10¹¹ unique pairs). Even at 10⁹ simple operations per second, that is roughly 1000 s, or about 17 minutes, in optimised C. In pure Python, at about 10⁷–10⁸ operations per second, it would take hours to days. A is about a million times less work.

### Task 5: Complexity ranking

From fastest-growing to slowest-growing:

**2ⁿ > n² > n log₂ n > n > log₂ n > 1**

At n = 1,000,000: log₂ n ≈ 20, n log₂ n ≈ 2 × 10⁷, n² = 10¹², and 2ⁿ has about 301,030 decimal digits.

**Problematic for large n:** 2ⁿ is unusable once n is more than about 60–100. n² becomes a problem around n ≈ 10⁵–10⁶, where it needs 10¹⁰–10¹² operations. n log n and anything slower stay practical.

### Task 6: Searching for 45 in `[3, 8, 12, 17, 24, 31, 45, 51, 63]`

1. **List not known to be sorted: linear search.** Check the elements one by one: 3, 8, 12, 17, 24, 31, 45. The target is found at index 6 after **7 comparisons**. The worst case is n comparisons, which is O(n).
2. **List known to be sorted: binary search.** Set low = 0 and high = 8. The middle is index 4, which holds 24. Since 24 < 45, move to the right half, so low = 5. The new middle is index 6, which holds 45. Found after **2 comparisons**. The worst case is about log₂ n comparisons, which is O(log n).

**Why the second is faster:** sorted order means one comparison tells you which half cannot contain the target, so half of the remaining candidates are discarded each step. For 1,000,000 elements, that is about 20 comparisons instead of up to 1,000,000. The script output `linear=(6, 7) binary=(6, 2)` confirms this.

## Part C: Data and problem solving

### Task 7: Largest of 500 million integers

**Algorithm:**
1. Set `best` to the first value.
2. For each remaining value, if it is greater than `best`, set `best` to it.

This is O(n) time and O(1) extra memory.

- **How many times is each number examined?** Exactly **once**, which means n − 1 = 499,999,999 comparisons. You cannot do better, because any number you skip could be the maximum.
- **Do you need to store all 500 million?** **No.** You only need the current best. Stream the numbers from the file or network. Stored as 4-byte ints, the data would take about 2 GB, and as a Python list of int objects it would take more than 10 GB. Neither is needed.
- **Could you process the data in chunks?** **Yes.** Max is associative and commutative, so max(all) = max(max(chunk₁), max(chunk₂), ...). Each chunk can be read separately, and on a multicore machine each chunk can be given to a different core or process. The partial maxima are then combined in a tiny final step. This is a classic parallel reduction. On my 10-core M4, that could be 10 workers.

`chunked_max()` in the script processes 200,000 ints in 50,000-int chunks produced by a generator, so the full list is never stored. It returns the same result as a full `max()`.

### Task 8: Counting `A B A C B A D C A B`

**Data structure:** a **hash map** (a Python `dict`) that maps each letter to its count.

**Algorithm:**
1. Make one pass over the input.
2. For each letter, set `counts[letter] = counts.get(letter, 0) + 1`.

Each update takes O(1) on average, so the whole job is O(n) time and O(k) memory for k distinct keys.

The result is `{'A': 4, 'B': 3, 'C': 2, 'D': 1}`. It matches `collections.Counter`.

For a small known alphabet, a plain array indexed by `ord(letter) - ord('A')` works too. For parallel counting, each worker counts its own chunk into a local map, and the maps are merged at the end.

### Task 9: Matrix operations

A = [[1,2],[3,4]], B = [[5,6],[7,8]]

**Addition** (element by element):

A + B = [[1+5, 2+6], [3+7, 4+8]] = **[[6, 8], [10, 12]]**

**Multiplication** (row × column):

- c₁₁ = 1·5 + 2·7 = 19
- c₁₂ = 1·6 + 2·8 = 22
- c₂₁ = 3·5 + 4·7 = 43
- c₂₂ = 3·6 + 4·8 = 50

A × B = **[[19, 22], [43, 50]]**

**Number of multiplications:** there are 4 output cells, and each needs 2 multiplications, so **8** in total. There are also 4 additions. In general, multiplying two n×n matrices takes n³ multiplications, so 1000×1000 matrices need 10⁹. Strassen's method can do a 2×2 product with 7 multiplications, but the standard method uses 8. The script's counter reports `multiplications=8`.

## Part D: Systems intuition

### Task 10: Fast CPU, slow array scan

```python
for x in huge_array:
    total += x
```

1. **Memory bandwidth and latency, not the CPU, are the limit.** Each element does one addition but must be brought from RAM. RAM is about 100× slower to reach than a register. If the array is much bigger than the caches, the CPU spends most of its time waiting for memory. On my M4, the cache line is 128 bytes, the L1 data cache is 64 KB, and the shared L2 is 16 MB on the performance cores and 4 MB on the efficiency cores. An array larger than about 16 MB has to stream from the 16 GB of main memory.
2. **Poor locality or pointer chasing.** A Python list stores pointers to int objects that are scattered around the heap, not the numbers themselves next to each other. Every element can therefore cause a cache miss or a TLB miss, and the hardware prefetcher cannot help much.
3. **Interpreter overhead.** In CPython, each iteration involves bytecode dispatch, type checks, reference counting, and creating a new int object for `total`. That is tens of machine instructions for one useful addition, so the "extremely fast" CPU is mostly doing bookkeeping. NumPy's `np.sum` or C code avoids this.

Other possible reasons:
- The array does not fit in RAM, so the program is swapping to disk or reading from disk.
- Only one of the 10 cores is used.
- The code is not vectorised with SIMD.
- Other programs are competing for the CPU or memory bandwidth.

### Task 11: What happens when you run `./program`

1. **The shell reads the command.** It parses the line, sees a path (`./`), and checks that the file exists and has execute permission.
2. **A new process is created.** The shell calls `fork()` (or `posix_spawn`), which makes a child process that is a copy of the shell.
3. **The program is loaded.** The child calls `execve("./program")`. The kernel reads the executable format (Mach-O on macOS, ELF on Linux) and sets up a fresh virtual address space with code, data, heap, and stack. It passes in `argv` and the environment variables.
4. **Libraries are linked.** The dynamic linker (`dyld` on macOS) loads shared libraries such as libc and resolves symbols. If the file is a script starting with `#!`, the kernel starts the interpreter instead, such as `python3`.
5. **The program is scheduled.** The OS scheduler puts the process on a CPU core. Code pages are loaded lazily from disk into RAM through page faults, then pulled into the caches as they are executed.
6. **The program runs.** The CPU fetches, decodes, and executes instructions. When the program needs the OS, it makes **system calls**. For output, it calls `write()` on file descriptor 1 (stdout), which is often buffered by the C library first.
7. **The output is displayed.** The kernel sends the bytes through the terminal (pty) to the terminal application, which draws the characters on screen.
8. **The program exits.** It calls `exit()`. Buffers are flushed, the kernel frees the process's memory and files, and the shell, which was waiting in `wait()`, gets the exit status (`$?`) and prints the next prompt.

### Task 12: Four programs at once

| Program | Likely bottleneck |
|---|---|
| A: lots of CPU | **CPU** compute (core speed, number of cores) |
| B: lots of memory | **Memory** capacity and bandwidth. If RAM runs out, swapping to disk makes it much slower. |
| C: reads a large file | **Disk / storage I/O** (SSD throughput, file-system cache) |
| D: waits for user input | **The user.** It is I/O-bound on the keyboard and is blocked or sleeping almost all the time, using about 0% CPU. |

- **Can all four run at the same time?** Yes. My M4 has 10 cores, so they can run truly in parallel on different cores. They also use different resources, which overlap well: while C waits for the disk and D waits for the keyboard, the OS gives the CPU to A and B. They only slow each other down where they share something, such as memory bandwidth, the caches, or total RAM.
- **What determines CPU time?** The number of instructions the program executes, times the cycles per instruction (which depends on cache misses, branch mispredictions, and so on), divided by the clock speed. Only time spent actually running on a core counts. Time blocked on I/O or waiting for input does not count. How much the OS scheduler gives each process depends on priorities and how many processes are ready to run.
- **With one CPU core:** they still all run *concurrently* but no longer in *parallel*. The OS time-slices the single core and switches between processes many times per second, which is a context switch. A and B now compete directly for the core, so each gets roughly half and takes about twice as long in wall-clock time. C and D barely suffer because they mostly wait. While they are blocked, the core runs A or B, so the system still feels responsive.

## Part E: Reasoning challenges

### Task 13: "Twice as fast means 5 hours"

**I do not fully agree.** It is only true if the whole 10 hours is limited by the part that gets twice as fast. "Twice as fast" usually means the CPU, through clock speed or more instructions per cycle. However:

- If part of the time is spent on disk or network I/O, or waiting on memory (1 billion records is a lot of data), that part does not speed up. For example, if 4 hours are I/O and 6 hours are CPU, the new time is 4 + 6/2 = **7 hours**, not 5. This is Amdahl's law.
- If the new computer has twice as many *cores* but the program is single-threaded, there is almost **no** improvement until the program is parallelised.
- Fixed overheads, such as startup, database locks, and network latency, do not shrink either.

5 hours is a best case. The real answer depends on what the bottleneck is, so you should profile first.

### Task 14: Ways to make a 60-second program faster (same answer)

**Algorithm and data:**
- Use a better algorithm, for example O(n log n) instead of O(n²), or a hash lookup instead of a linear search.
- Avoid repeated work with caching or memoisation, and move work out of loops.
- Use better data structures, such as dict or set for lookups, or contiguous arrays instead of linked structures.
- Process only the data you need with filtering or early exit, and stream it instead of loading everything.

**Implementation:**
- Use vectorised libraries such as NumPy, or a compiled language or extension (C, C++, Rust, Cython, Numba).
- Use a faster interpreter such as PyPy.
- Turn on compiler optimisations (`-O2`/`-O3`) and SIMD.
- Improve memory locality: lay out data for sequential access, use smaller data types, and block or tile loops to fit in the cache.
- Reduce I/O: buffer reads and writes, use binary formats instead of text, compress, and avoid printing inside loops.
- Remove debug logging and unnecessary memory allocation and copying.

**Parallelism and concurrency:**
- Use several cores with multiprocessing, threads, or OpenMP. That is up to 10 cores on my M4, but parallelism only helps where the work can be split.
- Use the GPU for data-parallel maths.
- Overlap I/O with computation using async I/O or prefetching.
- Distribute the work across several machines.

**Hardware and environment:**
- Use a faster CPU, more RAM (to avoid swapping), or an SSD instead of an HDD.
- Close competing programs.

**Most important:** profile first and optimise the part that actually takes the time.

### Task 15: Bottleneck identification

A = 2 s, B = 2 s, C = 50 s, D = 3 s, E = 2 s. The total is 59 s.

Making A, B, D, and E 10× faster changes their combined time from 9 s to 0.9 s.

New total = 0.2 + 0.2 + 50 + 0.3 + 0.2 = **50.9 s**

Speedup = 59 / 50.9 ≈ **1.16×**, which is only about 14% less time, even though four of the five parts got 10× faster.

**Where to focus:** on **C**, which takes 50 of the 59 seconds (85%). Making only C 10× faster gives 2 + 2 + 5 + 3 + 2 = 14 s, which is a **4.2× speedup**. Even if C took zero time, the total could not go below 9 s, so the maximum is 6.6×. This is Amdahl's law: the part you do not improve limits the overall speedup. The script reports `1.159x` for the first option and `4.21x` for the second.

## Final challenge

### Task 16: 10 million student marks (0–100)

**Main idea:** marks are integers in a tiny range of only 101 possible values. Instead of sorting 10 million numbers, build a **histogram (counting array) of 101 counters** in a single pass. Every requested statistic can then be read from those 101 numbers.

**Algorithm:**
1. Create `hist = [0] * 101`.
2. For each mark m, check that 0 ≤ m ≤ 100 (invalid data is rejected with a clear error), then do `hist[m] += 1`.
3. Read the results from `hist`:
   - **n** = sum(hist)
   - **min** = the first index with a non-zero count. **max** = the last index with a non-zero count.
   - **average** = Σ(v × hist[v]) / n
   - **median** = walk the cumulative counts until you pass position n/2. If n is even, average the two middle values.
   - **count ≥ 90** = sum(hist[90:101]). **count < 40** = sum(hist[0:40]).

If the marks can have decimals (for example 87.5), use a 1001-bucket histogram with a step of 0.1. If any value is possible, the median needs a selection algorithm such as quickselect, which is O(n) on average, or a sort, which is O(n log n).

**Data structures:**
- A fixed-size integer array of 101 counters, which is a counting sort or histogram.
- Optionally, the input is kept as a compact byte array (`array('B')`, `bytes`, or NumPy `uint8`) instead of a Python list.

**How many times is the data examined?** **Exactly once.** All six results come from the histogram afterwards, which costs 101 steps and is nothing compared with 10⁷. The naive approach is: one pass each for min, max, sum, ≥90, and <40, plus a sort for the median. That is about 5 passes plus O(n log n), which is roughly 10⁷ × 23 ≈ 2.3 × 10⁸ comparisons.

**Memory estimate:**
- **Streaming with a histogram:** 101 counters, so under 1 KB. The 10 million marks never have to be in memory at once.
- **If all marks are held in memory:**
  - As `uint8`: 10⁷ × 1 byte = **10 MB**.
  - As 4-byte ints: 40 MB.
  - As a Python list: about **80 MB** of 8-byte pointers. CPython caches small ints, so values 0–100 are shared objects and add almost nothing extra.
  - A sorted copy for the median would double that.

My machine has 16 GB of RAM, so any of these fit, but the histogram needs almost nothing.

**Likely bottleneck:**
- Reading the data, through disk or network I/O and parsing text into integers, is probably the biggest cost if the marks come from a file.
- In pure Python, the per-element interpreter loop also matters. 10⁷ iterations take on the order of a second.
- The arithmetic itself is trivial. The histogram fits in the L1 cache, so the data stream from memory or I/O is what matters, not the computation.
- With the naive approach, the sort for the median would be the bottleneck.

**Making it faster without changing the answer:**
1. Use the histogram (one pass, no sort) instead of five passes plus a sort.
2. Store and read the data in a compact binary format (uint8) instead of text, so there is less I/O and no parsing.
3. Vectorise, for example `np.bincount(marks, minlength=101)` in NumPy, which runs in C with SIMD.
4. **Parallelise:** split the 10 million marks into chunks, one per core (10 cores on the M4), and have each worker build its own 101-counter histogram. Then add the histograms together element by element. Addition is associative, so the result is identical to the serial one. The merge costs only 10 × 101 additions, and each worker uses its own histogram, so there is no lock contention.
5. Overlap reading the next chunk with processing the current one.

`mark_stats()` in the script implements the one-pass histogram. It was tested on 100,001 random marks against a sort-based reference and printed `matches sort-based reference: True`.
