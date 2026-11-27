"""Milestone 3 / Experiment 2: false sharing in cache lines.

Two processes increment two *different* 64-bit slots of one shared
multiprocessing.RawArray('q', ...) 10,000,000 times each.

  Case A (handout, adjacent) : arr[0] vs arr[1]   ->  8 B apart
  Case B (handout, "padded") : arr[0] vs arr[7]   -> 56 B apart
  Case C (extra)             : arr[0] vs arr[8]   -> 64 B apart  (separate 64 B lines)
  Case D (extra)             : arr[0] vs arr[16]  -> 128 B apart (separate 128 B lines = M4 line size)
  Solo  (baseline)           : one process alone on arr[0]

Note: with a 64/128-byte aligned base, arr[7] occupies bytes 56..63, i.e. the SAME
line as arr[0] for both 64 B and 128 B lines; case B is therefore not isolated. Cases
C and D are added to test real 64 B and 128 B (Apple M4, hw.cachelinesize=128) padding.

Two engines:
  python : the handout's literal loop `arr[idx] += 1` in CPython (~tens of ns per op)
  numba  : same loop JIT-compiled (one load + one store per iteration, ~1 ns per op),
           so that coherence traffic is not hidden behind interpreter overhead
           (100,000,000 increments per process so the two loops overlap).
  atomic : JIT loop doing a hardware atomic fetch-add per increment (LLVM atomicrmw,
           arm64 LDADD). Plain stores are absorbed by the store buffer + store-to-load
           forwarding; an atomic RMW must own the line in the L1 every time, which is
           what C/C++ counters (std::atomic) and lock words actually do.

Usage: python3 cache_collision.py [--reps 3] [--engines python numba]
"""
import argparse
import ctypes
import multiprocessing as mp
import statistics
import sys
import time

from lab_constants import describe

INCREMENTS = 10_000_000  # handout value, used for the python engine
# numba does 10M ops in ~9 ms, shorter than process start-up jitter, so it gets 10x more
# work to guarantee the two loops actually overlap in time
INCREMENTS_FOR = {"python": INCREMENTS, "numba": 10 * INCREMENTS, "atomic": INCREMENTS}
CASES = [  # name, array length, index for process 0, index for process 1 (None = solo)
    ("Solo", 8, 0, None),
    ("A adjacent 0/1 (8 B)", 8, 0, 1),
    ("B handout 0/7 (56 B)", 8, 0, 7),
    ("C 0/8 (64 B)", 32, 0, 8),
    ("D 0/16 (128 B)", 32, 0, 16),
]


def _python_loop(arr, idx: int, n: int) -> None:
    for _ in range(n):
        arr[idx] += 1


def _make_numba_kernel(engine: str):
    """Build the JIT kernel inside the worker (imports kept out of the python engine)."""
    from numba import njit, types
    from numba.core import cgutils
    from numba.extending import intrinsic

    if engine == "numba":
        @njit(cache=False)
        def bump(a, ij, n):
            # The index is re-read from memory every iteration: the store to a[...] may alias
            # ij, so LLVM can neither hoist the index nor keep the counter in a register
            # -> one real load + store per iteration (checked: ~0.85 ns/op solo, while a
            # scalar-index version got loop-versioned into a register at ~0.2 ns/op).
            for _ in range(n):
                a[ij[0]] = a[ij[1]] + 1
            return a[ij[0]]
        return bump

    @intrinsic
    def atomic_add(typingctx, arr, idx, val):
        """LLVM `atomicrmw add monotonic` = C11 atomic_fetch_add(relaxed) (LDADD on arm64)."""
        sig = types.int64(arr, idx, val)

        def codegen(context, builder, signature, args):
            aryty = signature.args[0]
            ary = context.make_array(aryty)(context, builder, args[0])
            ptr = cgutils.get_item_pointer(context, builder, aryty, ary, [args[1]])
            return builder.atomic_rmw("add", ptr, args[2], "monotonic")
        return sig, codegen

    @njit(cache=False)
    def bump_atomic(a, ij, n):
        i = ij[0]
        for _ in range(n):
            atomic_add(a, i, 1)
        return a[i]
    return bump_atomic


def worker(arr, idx: int, engine: str, barrier, out, slot: int) -> None:
    n_inc = INCREMENTS_FOR[engine]
    if engine in ("numba", "atomic"):
        import numpy as np

        bump = _make_numba_kernel(engine)
        view = np.frombuffer(arr, dtype=np.int64)  # zero-copy view of the shared RawArray
        ij = np.array([idx, idx], dtype=np.int64)
        bump(np.zeros(32, dtype=np.int64), np.zeros(2, dtype=np.int64), 10)  # compile first
        barrier.wait()
        t0 = time.perf_counter()
        bump(view, ij, n_inc)
    else:
        barrier.wait()
        t0 = time.perf_counter()
        _python_loop(arr, idx, n_inc)
    out[slot] = time.perf_counter() - t0


def run_case(length: int, i0: int, i1, engine: str):
    arr = mp.RawArray("q", length)
    idxs = [i0] if i1 is None else [i0, i1]
    barrier = mp.Barrier(len(idxs) + 1)
    out = mp.RawArray("d", len(idxs))
    procs = [mp.Process(target=worker, args=(arr, ix, engine, barrier, out, k)) for k, ix in enumerate(idxs)]
    for p in procs:
        p.start()
    barrier.wait()
    t0 = time.perf_counter()
    for p in procs:
        p.join()
    wall_ms = (time.perf_counter() - t0) * 1e3
    assert all(p.exitcode == 0 for p in procs), "worker failed"
    assert all(arr[ix] == INCREMENTS_FOR[engine] for ix in idxs), "lost increments (should be impossible: one writer per slot)"
    base = ctypes.addressof(arr)
    return wall_ms, max(out) * 1e3, min(out) * 1e3, base % 64, base % 128


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--engines", nargs="+", default=["python", "numba", "atomic"])
    a = ap.parse_args()
    print(describe())
    print(f"increments per process = {INCREMENTS_FOR}; reps per case = {a.reps}")
    summary = []
    for engine in a.engines:
        for name, length, i0, i1 in CASES:
            loops = []
            for r in range(a.reps):
                wall, loop, fast, m64, m128 = run_case(length, i0, i1, engine)
                loops.append(loop)
                print(f"[{engine:6s}] {name:22s} rep{r + 1}: slowest loop {loop:8.1f} ms, fastest {fast:8.1f} ms, "
                      f"wall {wall:8.1f} ms, "
                      f"base addr mod64={m64} mod128={m128}", flush=True)
            summary.append((engine, name, statistics.median(loops), min(loops), max(loops)))
    print("\n| Engine | Case | median loop ms | min | max | ns per increment | vs Solo | vs Case D |")
    print("|---|---|---|---|---|---|---|---|")
    for engine in a.engines:
        rows = [s for s in summary if s[0] == engine]
        solo = rows[0][2]
        d = [s for s in rows if s[1].startswith("D")][0][2]
        for _, name, med, lo, hi in rows:
            print(f"| {engine} | {name} | {med:.1f} | {lo:.1f} | {hi:.1f} | {med * 1e6 / INCREMENTS_FOR[engine]:.2f} | "
                  f"{med / solo:.2f}x | {med / d:.2f}x |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
