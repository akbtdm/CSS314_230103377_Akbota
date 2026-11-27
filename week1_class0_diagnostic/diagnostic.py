"""Class 0 diagnostic - runnable answers for the code tasks.

Student: Akbota Kutushova (230103377). Run: python3 diagnostic.py
All demos use small inputs so the whole script finishes in well under a second.
"""

import random
from collections import Counter


# Task 1 - trace the code
def task1_trace():
    x, y = 5, 2
    steps = [("x = 5, y = 2", x, y)]
    x = x + y
    steps.append(("x = x + y", x, y))
    y = x * 2
    steps.append(("y = x * 2", x, y))
    x = y - x
    steps.append(("x = y - x", x, y))
    return steps


# Task 2 - buggy vs fixed sum
def buggy_total(numbers):
    total = 0
    for i in range(len(numbers)):
        total = numbers[i]  # bug: overwrites instead of accumulating
    return total


def fixed_total(numbers):
    total = 0
    for value in numbers:
        total += value
    return total


# Task 3 - count iterations of nested loops (counted, not printed)
def nested_count(outer, inner):
    count = 0
    for _ in range(outer):
        for _ in range(inner):
            count += 1
    return count


# Task 6 - linear search vs binary search, counting comparisons
def linear_search(items, target):
    comparisons = 0
    for index, value in enumerate(items):
        comparisons += 1
        if value == target:
            return index, comparisons
    return -1, comparisons


def binary_search(items, target):
    low, high, comparisons = 0, len(items) - 1, 0
    while low <= high:
        mid = (low + high) // 2
        comparisons += 1
        if items[mid] == target:
            return mid, comparisons
        if items[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1, comparisons


# Task 7 - streaming / chunked maximum (never stores the whole dataset)
def chunked_max(chunks):
    best = None
    for chunk in chunks:
        local = max(chunk)  # each chunk could be handled by a different worker
        if best is None or local > best:
            best = local
    return best


def generate_chunks(total, chunk_size, seed=1):
    rng = random.Random(seed)
    produced = 0
    while produced < total:
        size = min(chunk_size, total - produced)
        yield [rng.randint(-10**9, 10**9) for _ in range(size)]
        produced += size


# Task 8 - counting with a dictionary (hash map)
def count_letters(letters):
    counts = {}
    for letter in letters:
        counts[letter] = counts.get(letter, 0) + 1
    return counts


# Task 9 - matrix addition and multiplication with a multiplication counter
def mat_add(a, b):
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def mat_mul(a, b):
    n, m, p = len(a), len(b), len(b[0])
    mults = 0
    result = [[0] * p for _ in range(n)]
    for i in range(n):
        for j in range(p):
            s = 0
            for k in range(m):
                s += a[i][k] * b[k][j]
                mults += 1
            result[i][j] = s
    return result, mults


# Task 15 - Amdahl-style speedup when some parts are accelerated
def overall_speedup(times, factor, accelerated):
    before = sum(times.values())
    after = sum(t / factor if name in accelerated else t for name, t in times.items())
    return before, after, before / after


# Task 16 - one-pass statistics for marks 0..100 using a 101-bucket histogram
MAX_MARK = 100


def mark_stats(marks):
    hist = [0] * (MAX_MARK + 1)
    for m in marks:
        if not 0 <= m <= MAX_MARK:
            raise ValueError(f"mark out of range: {m}")
        hist[m] += 1
    n = sum(hist)
    if n == 0:
        raise ValueError("no marks given")
    lo = next(v for v in range(MAX_MARK + 1) if hist[v])
    hi = next(v for v in range(MAX_MARK, -1, -1) if hist[v])
    mean = sum(v * c for v, c in enumerate(hist)) / n
    median = _median_from_hist(hist, n)
    ge90 = sum(hist[90:])
    lt40 = sum(hist[:40])
    return {"n": n, "min": lo, "max": hi, "mean": mean, "median": median,
            "ge90": ge90, "lt40": lt40}


def _kth_from_hist(hist, k):
    """Return the k-th smallest value (0-based) from a histogram."""
    seen = 0
    for value, count in enumerate(hist):
        seen += count
        if seen > k:
            return value
    raise IndexError(k)


def _median_from_hist(hist, n):
    if n % 2:
        return float(_kth_from_hist(hist, n // 2))
    return (_kth_from_hist(hist, n // 2 - 1) + _kth_from_hist(hist, n // 2)) / 2


def reference_stats(marks):
    s = sorted(marks)
    n = len(s)
    median = float(s[n // 2]) if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    return {"n": n, "min": s[0], "max": s[-1], "mean": sum(s) / n, "median": median,
            "ge90": sum(m >= 90 for m in s), "lt40": sum(m < 40 for m in s)}


def main():
    print("Task 1:")
    for label, x, y in task1_trace():
        print(f"  {label:<14} -> x={x}, y={y}")

    nums = [10, 20, 30, 40, 50]
    print(f"Task 2: buggy={buggy_total(nums)} fixed={fixed_total(nums)}")

    print(f"Task 3: 10x10={nested_count(10, 10)} 1000x1000={nested_count(1000, 1000)}"
          f" (10 x 1,000,000 = {10 * 1_000_000:,} by formula)")

    data = [3, 8, 12, 17, 24, 31, 45, 51, 63]
    print(f"Task 6: linear (index, comparisons)={linear_search(data, 45)}"
          f" binary={binary_search(data, 45)}")

    total, size = 200_000, 50_000
    stream_max = chunked_max(generate_chunks(total, size))
    check = max(v for c in generate_chunks(total, size) for v in c)
    print(f"Task 7: chunked max of {total:,} ints = {stream_max} (matches full max: {stream_max == check})")

    letters = "A B A C B A D C A B".split()
    counts = count_letters(letters)
    print(f"Task 8: {counts} (Counter agrees: {counts == dict(Counter(letters))})")

    a, b = [[1, 2], [3, 4]], [[5, 6], [7, 8]]
    prod, mults = mat_mul(a, b)
    print(f"Task 9: A+B={mat_add(a, b)} AxB={prod} multiplications={mults}")

    times = {"A": 2, "B": 2, "C": 50, "D": 3, "E": 2}
    before, after, sp = overall_speedup(times, 10, {"A", "B", "D", "E"})
    _, after_c, sp_c = overall_speedup(times, 10, {"C"})
    print(f"Task 15: A,B,D,E 10x -> {before}s to {after:.1f}s, speedup {sp:.3f}x;"
          f" C 10x instead -> {after_c:.1f}s, speedup {sp_c:.2f}x")

    rng = random.Random(42)
    marks = [rng.randint(0, 100) for _ in range(100_001)]
    got, ref = mark_stats(marks), reference_stats(marks)
    print(f"Task 16: {got}")
    print(f"         matches sort-based reference: {got == ref}")


if __name__ == "__main__":
    main()
