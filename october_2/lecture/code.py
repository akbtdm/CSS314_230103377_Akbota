"""
================================================================================
CSS 314 OPENMP & MULTI-CORE LAB EXAM - SECTION 2 SUBMISSION HEADER
================================================================================
STUDENT NAME : [Kutushova Akbota]
STUDENT ID   : 230103377

--- AUTO-GENERATED MACHINE TELEMETRY (Paste from script terminal output) ---
CPU / Processor : arm (Apple Silicon)
OS Platform     : Darwin 24.5.0
Detected Cores  : 10 Logical Cores

--- TASK 3: EMPIRICAL BENCHMARK TABLE (Paste from script terminal output) ---
Threads (p) | Run 1 (s) | Run 2 (s) | Run 3 (s) | Avg Time (s) | Speedup (T1/Tp)
1           | 1.21      | 1.16      | 1.18      | 1.18         | 1.00x
2           | 0.62      | 0.62      | 0.62      | 0.62         | 1.91x
4           | 0.37      | 0.35      | 0.36      | 0.36         | 3.30x
8           | 0.30      | 0.29      | 0.28      | 0.29         | 4.09x

--- HARDWARE & THEORETICAL REFLECTIONS (Answer in 2-3 sentences each) ---
1. Physical vs Logical Cores: Compare your CPU's physical core count vs detected logical
   cores. Did scaling from 4 to 8 workers achieve a 2x speedup? Why does SMT /
   Hyper-Threading share execution units rather than doubling raw compute hardware?
   Answer: 
     The apple silicon CPU has 10 physical cores and 10 detected logical cores because 
     it doesnt use SMT . Increasing workers from 4 to 8 didnt give a 2x speedup; the time
     only decreased from 0.36s to 0.29 , because some workers use slower efficiency cores 
     and there is extra proccess overhead. ON CPUs with SMR Hyper-Threading, 2 treads share the saME 
     cote and its resources so the number of logical cores doesnt mean twice the real computing power

2. Load Imbalance & Amdahl's Law: Explain why naive contiguous chunking failed on the
   inverted quadratic workload, and how cyclic allocation restored thread utilization.
   If 4% of the execution is strictly serial (IPC synchronization, process setup),
   what is the theoretical maximum speedup possible as p -> infinity?
   Answer:
     With contiguous chunks worker 0 received the lowest indices, where
     steps (remaining/20)^2 is largest, so it did 58 percent of all work (1 - 0.75^3) while
     the other workers sat idle; the run took 0.70s. Cyclic allocation i % p interleaves
     heavy and light items so every worker gets 1/p of the work, cutting time to 0.36s
     (1.95x faster). With a 4% serial fraction, Amdahl's Law gives
     S_max = 1 / 0.04 = 25x as p -> infinity.

3. Thermal & Power Throttling: Examine your Run 1, Run 2, and Run 3 times for 8 workers.
   Did execution times drift upward across runs? Explain the role of CPU thermal
   throttling and Turbo Boost power limit caps (PL1 / PL2) during sustained workloads.
   Answer:
     No, the 8-worker times didnt drift upward (0.30s, 0.29s, 0.28s); they
     slightly improved due to warm caches and the process pool startup cost settling.
     Under long sustained all-core loads Intel CPUs first boost up to the short-term
     PL2 power limit, then drop to the sustained PL1 limit after the turbo window
     expires and thermal throttling near Tjmax lowers clocks further, making later runs
     slower. My runs were too short well under a second and Apple Silicon is very
     power-efficient, so no throttling appeared.
================================================================================
"""

import sys
import os
import time
import platform
from concurrent.futures import ProcessPoolExecutor

# ==============================================================================
# 1. STUDENT CONFIGURATION
# ==============================================================================
STUDENT_ID = 230103377  # <--- ENTER YOUR NUMERIC STUDENT ID HERE (e.g. 20210045)

if STUDENT_ID == 0:
    print("[ERROR] You must set your numeric STUDENT_ID on line 42 before running.")
    sys.exit(1)

# Personalized bounds to ensure unique dataset per student ID
SEED = (STUDENT_ID * 7 + 13) % 1000
N_ELEMENTS = 6_000_000 + (SEED * 120)
NUM_ITEMS = 2_400 + (SEED % 60) * 10


# ==============================================================================
# TOP-LEVEL WORKERS (DO NOT REMOVE FROM MODULE LEVEL)
# ==============================================================================

def task1_worker_chunk(start, end, seed):
    """
    TODO (TASK 1 - 30 MARKS):
    1. Iterate through index i from start to end (exclusive).
    2. Check if: (((i * 5) ^ seed) % 11) == 0
    3. If true, increment a local counter.
    4. Return the local counter value.
    """
    local_count = 0
    for i in range(start, end):
        if (((i * 5) ^ seed) % 11) == 0:
            local_count += 1
    return local_count




def heavy_kernel(idx, seed, total_items):
    """
    Simulates irregular computational workload with inverted quadratic skew.
    Early indices (idx -> 0) incur massive compute costs, while late indices
    (idx -> total_items) execute almost instantaneously.
    """
    remaining = total_items - 1 - idx
    steps = int((remaining / 20) ** 2) + 15
    acc = 0
    for k in range(steps):
        acc += ((idx * 37) ^ (k + seed)) % 1000
    return acc


def task2_worker_bucket(bucket_indices, seed, total_items):
    """Worker processing an assigned bucket of items."""
    subtotal = 0
    for idx in bucket_indices:
        subtotal += heavy_kernel(idx, seed, total_items)
    return subtotal


# ==============================================================================
# TASK 1: PARALLEL REDUCTION DISPATCHER
# ==============================================================================
def solve_task1_parallel(n, seed, num_workers=4):
    """
    Divides [0, n) among num_workers, submits chunks to task1_worker_chunk,
    and aggregates partial results without race conditions.
    """
    chunk_size = (n + num_workers - 1) // num_workers
    chunks = []
    for w in range(num_workers):
        start = w * chunk_size
        end = min((w + 1) * chunk_size, n)
        if start < end:
            chunks.append((start, end, seed))

    total = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(task1_worker_chunk, c[0], c[1], c[2]) for c in chunks]
        for f in futures:
            total += f.result()

    return total


# ==============================================================================
# TASK 2: LOAD BALANCING (30 MARKS)
# ==============================================================================
def solve_task2_balanced(num_items, seed, num_workers=4):
    """
    TODO (TASK 2 - 30 MARKS):
    The starter code below divides num_items into large contiguous slices (Naive Static).
    Because heavy_kernel(idx) executes quadratically more steps when idx is small,
    Worker 0 is severely bottlenecked while Worker (num_workers - 1) finishes immediately.

    YOUR TASK:
    Rewrite the bucket assignment below to use CYCLIC / INTERLEAVED distribution:
    Assign item 'i' to worker bucket: (i % num_workers).
    """
    # ---------------------------------------
    buckets = [[] for _ in range(num_workers)]
    for i in range(num_items):
        buckets[i % num_workers].append(i)
    # ---------------------------------------

    total = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(task2_worker_bucket, b, seed, num_items) for b in buckets]
        for f in futures:
            total += f.result()

    return total


def _run_naive_reference(num_items, seed, num_workers=4):
    """Internal benchmark reference using naive contiguous chunks."""
    chunk_size = (num_items + num_workers - 1) // num_workers
    buckets = []
    for w in range(num_workers):
        start = w * chunk_size
        end = min((w + 1) * chunk_size, num_items)
        buckets.append(list(range(start, end)))
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(task2_worker_bucket, b, seed, num_items) for b in buckets]
        return sum(f.result() for f in futures)


# ==============================================================================
# MAIN TEST & TELEMETRY HARNESS
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("SECTION 2: MACHINE HARDWARE TELEMETRY")
    print(f"Processor : {platform.processor() or platform.machine()}")
    print(f"OS        : {platform.system()} {platform.release()}")
    print(f"CPU Count : {os.cpu_count()} Logical Cores")
    print(f"Student ID: {STUDENT_ID} (Seed: {SEED})")
    print("=" * 70)

    # 1. VERIFY TASK 1
    print("\n[Running Task 1 Verification]...")
    expected_p1 = sum(1 for i in range(N_ELEMENTS) if (((i * 5) ^ SEED) % 11) == 0)
    actual_p1 = solve_task1_parallel(N_ELEMENTS, SEED, num_workers=4)

    if actual_p1 == expected_p1:
        print(f"-> TASK 1: [PASSED] (Checksum: {actual_p1})")
    else:
        print(f"-> TASK 1: [FAILED] (Expected: {expected_p1}, Got: {actual_p1})")
        print("   Hint: Implement filtering and counting loop inside 'task1_worker_chunk'.")

    # 2. VERIFY TASK 2 LOAD BALANCING
    print("\n[Running Task 2 Load Balancing Test] (4 workers)...")
    t_start = time.perf_counter()
    res_user = solve_task2_balanced(NUM_ITEMS, SEED, num_workers=4)
    t_user = time.perf_counter() - t_start

    t_start_ref = time.perf_counter()
    res_naive = _run_naive_reference(NUM_ITEMS, SEED, num_workers=4)
    t_naive = time.perf_counter() - t_start_ref

    if res_user == res_naive:
        print(f"-> TASK 2: [PASSED] (Checksum: {res_user})")
        print(f"   Your 4-Worker Time: {t_user:.2f}s | Naive 4-Worker Time: {t_naive:.2f}s")
        if t_user < (t_naive * 0.75):
            speedup = t_naive / t_user
            print(f"   Optimization Confirmed: {speedup:.2f}x faster than naive static allocation.")
        else:
            print("   Warning: Naive static allocation detected. Switch to cyclic (i % num_workers) to pass optimization threshold.")
    else:
        print(f"-> TASK 2: [FAILED] Checksum mismatch! All items must be processed exactly once.")

    # 3. RUN TASK 3 BENCHMARK SUITE
    print("\n[Running Task 3 Benchmark Suite] (1, 2, 4, 8 workers)...")
    print("-" * 70)
    print(f"{'Threads':<10} | {'Run 1':<9} | {'Run 2':<9} | {'Run 3':<9} | {'Avg Time':<10} | {'Speedup':<8}")
    print("-" * 70)

    t1_mean = None
    for p in [1, 2, 4, 8]:
        runs = []
        for _ in range(3):
            t0 = time.perf_counter()
            _ = solve_task2_balanced(NUM_ITEMS, SEED, num_workers=p)
            runs.append(time.perf_counter() - t0)
        avg_t = sum(runs) / len(runs)
        if p == 1:
            t1_mean = avg_t
            sp_str = "1.00x"
        else:
            sp = (t1_mean / avg_t) if avg_t > 0 else 1.0
            sp_str = f"{sp:.2f}x"
        print(f"{p:<10} | {runs[0]:<8.2f}s | {runs[1]:<8.2f}s | {runs[2]:<8.2f}s | {avg_t:<8.2f}s  | {sp_str:<8}")

    print("-" * 70)
    print("\n[EXAM COMPLETE] Copy telemetry and benchmark rows into top docstring header, answer the 3 questions, and submit to Moodle.")
