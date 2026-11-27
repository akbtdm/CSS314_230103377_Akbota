"""Milestone 4: parallel zero-copy 200 MB array sifter.

The master allocates a 200,000,000-byte POSIX shared memory segment, fills it with
pseudo-random uint8 values from numpy.random.default_rng(TARGET_SEED), and spawns
K = os.cpu_count() workers. Each worker attaches by name, wraps ONLY its own slice in
a NumPy array over shm.buf (no copy), and sums the elements with val % 7 == 0.
Results (sum, count, RSS figures, time) are written back into a second small shared
memory segment -- no Queue/Pipe/Manager/pickle data path.

RSS: resource.getrusage(RUSAGE_SELF).ru_maxrss is reported in BYTES on macOS (KiB on
Linux). Each worker reports its peak RSS after attaching (before touching data) and
after the scan; the delta is what the scan added. Shared pages are counted in the
RSS of every process that touches them, so the sum of raw RSS double-counts them.

--compare-copy additionally runs a deliberately copying variant (bytes(shm.buf[a:b]),
i.e. what serialization would do) to show what duplicated pages look like.

Usage: python3 matrix_sifter.py [--compare-copy]
"""
import argparse
import os
import resource
import sys
import time
from multiprocessing import Barrier, Process, shared_memory

import numpy as np

from lab_constants import SIFT_NAME, SIFT_RESULT_NAME, TARGET_SEED, describe

TOTAL_BYTES = 200_000_000
FILL_CHUNK = 8 * 1024 * 1024
SCAN_CHUNK = 1024 * 1024   # bounded temporaries (mask) per step
RES_COLS = 6               # sum, count, maxrss_before, maxrss_after, elapsed_ns, bytes_scanned
MB = 1e6


def maxrss_bytes() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024


def sift(view: np.ndarray):
    """Sum and count of elements with val % 7 == 0, using preallocated scratch buffers so
    the only memory the scan adds is the pages of `view` itself (no per-chunk temporaries)."""
    rem = np.empty(SCAN_CHUNK, dtype=np.uint8)
    mask = np.empty(SCAN_CHUNK, dtype=np.bool_)
    total = 0
    count = 0
    for s in range(0, view.size, SCAN_CHUNK):
        c = view[s:s + SCAN_CHUNK]          # a view, not a copy
        n = c.size
        np.remainder(c, 7, out=rem[:n])
        np.equal(rem[:n], 0, out=mask[:n])
        total += int(np.sum(c, where=mask[:n], dtype=np.uint64))
        count += int(np.count_nonzero(mask[:n]))
    return total, count


def bounds(wid: int, k: int):
    step = TOTAL_BYTES // k
    lo = wid * step
    return lo, (TOTAL_BYTES if wid == k - 1 else lo + step)


def worker(wid: int, k: int, mode: str, barrier) -> None:
    shm = shared_memory.SharedMemory(name=SIFT_NAME, track=False)
    res_shm = shared_memory.SharedMemory(name=SIFT_RESULT_NAME, track=False)
    res = np.ndarray((k, RES_COLS), dtype=np.int64, buffer=res_shm.buf)
    lo, hi = bounds(wid, k)
    sift(np.zeros(2 * SCAN_CHUNK, dtype=np.uint8))  # warm-up: allocator/ufunc temporaries go in the baseline
    rss0 = maxrss_bytes()
    barrier.wait()
    t0 = time.perf_counter_ns()
    if mode == "zero-copy":
        view = np.ndarray((hi - lo,), dtype=np.uint8, buffer=shm.buf, offset=lo)
    else:  # "copy": materialise a private bytes object, like pickling would
        view = np.frombuffer(bytes(shm.buf[lo:hi]), dtype=np.uint8)
    total, count = sift(view)
    elapsed = time.perf_counter_ns() - t0
    res[wid] = (total, count, rss0, maxrss_bytes(), elapsed, hi - lo)
    del view, res
    shm.close()
    res_shm.close()


def run_parallel(k: int, mode: str):
    res_shm = shared_memory.SharedMemory(name=SIFT_RESULT_NAME, create=True, size=k * RES_COLS * 8)
    try:
        barrier = Barrier(k + 1)
        procs = [Process(target=worker, args=(w, k, mode, barrier)) for w in range(k)]
        for p in procs:
            p.start()
        barrier.wait()
        t0 = time.perf_counter()
        for p in procs:
            p.join()
        wall_ms = (time.perf_counter() - t0) * 1e3
        if any(p.exitcode != 0 for p in procs):
            raise RuntimeError(f"worker failed ({mode})")
        res = np.ndarray((k, RES_COLS), dtype=np.int64, buffer=res_shm.buf).copy()
    finally:
        res_shm.close()
        res_shm.unlink()
    return wall_ms, res


def report(mode: str, wall_ms: float, res: np.ndarray, ref_sum: int, ref_count: int, master_rss: int) -> None:
    print(f"\n=== {mode}: K={len(res)} workers, wall (barrier -> last join) {wall_ms:.1f} ms ===")
    print("| worker | slice MB | sum (val%7==0) | count | ru_maxrss before MB | ru_maxrss after MB | delta MB | scan ms |")
    print("|---|---|---|---|---|---|---|---|")
    for w, (s, c, r0, r1, ns, nb) in enumerate(res):
        print(f"| {w} | {nb / MB:.1f} | {s:,} | {c:,} | {r0 / MB:.1f} | {r1 / MB:.1f} | {(r1 - r0) / MB:.1f} | {ns / 1e6:.1f} |")
    tot_sum, tot_cnt = int(res[:, 0].sum()), int(res[:, 1].sum())
    deltas = (res[:, 3] - res[:, 2]).sum()
    base = res[:, 2].sum()
    print(f"total sum = {tot_sum:,} (serial reference {ref_sum:,}) -> {'MATCH' if tot_sum == ref_sum else 'MISMATCH'}; "
          f"count = {tot_cnt:,} (ref {ref_count:,})")
    print(f"sum of worker RSS deltas = {deltas / MB:.1f} MB for {TOTAL_BYTES / MB:.0f} MB of data; "
          f"sum of worker baselines (interpreter+numpy, before touching data) = {base / MB:.1f} MB; "
          f"sum of raw worker peak RSS = {res[:, 3].sum() / MB:.1f} MB; master peak RSS = {master_rss / MB:.1f} MB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare-copy", action="store_true")
    a = ap.parse_args()
    k = os.cpu_count()
    print(describe())
    print(f"K = os.cpu_count() = {k}; segment '{SIFT_NAME}' {TOTAL_BYTES:,} B; ru_maxrss unit: "
          f"{'bytes (macOS)' if sys.platform == 'darwin' else 'KiB'}")
    print(f"master ru_maxrss before allocation: {maxrss_bytes() / MB:.1f} MB")
    shm = shared_memory.SharedMemory(name=SIFT_NAME, create=True, size=TOTAL_BYTES)
    try:
        data = np.ndarray((TOTAL_BYTES,), dtype=np.uint8, buffer=shm.buf)
        t0 = time.perf_counter()
        rng = np.random.default_rng(TARGET_SEED)
        for s in range(0, TOTAL_BYTES, FILL_CHUNK):
            e = min(s + FILL_CHUNK, TOTAL_BYTES)
            data[s:e] = rng.integers(0, 256, size=e - s, dtype=np.uint8)
        print(f"populated in {(time.perf_counter() - t0) * 1e3:.0f} ms; first bytes {data[:8].tolist()}; "
              f"master ru_maxrss now {maxrss_bytes() / MB:.1f} MB")
        t0 = time.perf_counter()
        ref_sum, ref_count = sift(data)
        serial_ms = (time.perf_counter() - t0) * 1e3
        print(f"serial reference in master: sum={ref_sum:,} count={ref_count:,} in {serial_ms:.1f} ms")
        master_rss = maxrss_bytes()

        wall_ms, res = run_parallel(k, "zero-copy")
        report("zero-copy", wall_ms, res, ref_sum, ref_count, master_rss)
        print(f"speed-up vs serial scan: {serial_ms / wall_ms:.2f}x on {k} processes")
        ok = int(res[:, 0].sum()) == ref_sum
        if a.compare_copy:
            wall_c, res_c = run_parallel(k, "copy")
            report("copy (bytes(shm.buf[a:b]) -- counter-example)", wall_c, res_c, ref_sum, ref_count, master_rss)
            ok = ok and int(res_c[:, 0].sum()) == ref_sum
        del data
    finally:
        shm.close()
        shm.unlink()
        print(f"\nclosed and unlinked '{SIFT_NAME}'")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
