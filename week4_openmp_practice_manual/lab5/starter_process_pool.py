# Lab 5 Python starter from the manual (process-pool recursive merge sort),
# unchanged except: this header, the sortedness assertion (Task 5.1) and
# printing the number of the input elements. Pure-Python lists, spawn start
# method (macOS default), nested ProcessPoolExecutor per recursion level.
# --- Python 3: Task-Based Recursive Merge Sort ---
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor


def sequential_merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = sequential_merge_sort(arr[:mid])
    right = sequential_merge_sort(arr[mid:])
    return merge(left, right)


def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def parallel_merge_sort_entry(arr, cutoff=50000):
    if len(arr) <= cutoff:
        return sequential_merge_sort(arr)
    mid = len(arr) // 2

    # Process-based asynchronous tasks
    with ProcessPoolExecutor(max_workers=2) as executor:
        f_left = executor.submit(parallel_merge_sort_entry, arr[:mid], cutoff)
        f_right = executor.submit(parallel_merge_sort_entry, arr[mid:], cutoff)
        left = f_left.result()
        right = f_right.result()
    return merge(left, right)


if __name__ == "__main__":
    N = 1_000_000
    data = np.random.randint(0, 10_000_000, size=N).tolist()

    t0 = time.perf_counter()
    sorted_arr = parallel_merge_sort_entry(data, cutoff=50_000)
    t1 = time.perf_counter()
    assert sorted_arr == sorted(data), "output not sorted"
    print(f"Parallel Sort Completed (N={N}): {t1 - t0:.4f}s  (assert sorted: OK)")

    t0 = time.perf_counter()
    seq = sequential_merge_sort(data)
    t1 = time.perf_counter()
    print(f"Sequential pure-Python merge sort (same N, same process): {t1 - t0:.4f}s")
