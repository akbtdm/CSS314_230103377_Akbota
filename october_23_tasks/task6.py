"""TASK 6 - Synchronization Cost & Data Loss.

Part 1 uses Python threads (threading). CPython 3.13 still has the GIL, so it
shows how much (or how little) a plain `counter += 1` loses there, and the cost
of a threading.Lock.

Part 2 repeats the experiment with real native threads (Numba + OpenMP, no
GIL): the unsafe version does a plain load / add / store on the shared counter,
the correct one uses a hardware atomic fetch-and-add (LLVM atomicrmw, the same
instruction Java's AtomicInteger.incrementAndGet uses).
"""
import os

os.environ.setdefault("NUMBA_THREADING_LAYER", "omp")

import platform
import statistics
import threading
import time

import numpy as np
from numba import get_thread_id, njit, prange, set_num_threads
from numba.core import cgutils, types
from numba.extending import intrinsic

N_THREADS = 4
INCREMENTS = 2_000_000
EXPECTED = N_THREADS * INCREMENTS
REPEATS = 5


# ---------------------------------------------------------------- Python threads
class Counter:
    def __init__(self):
        self.value = 0


def run_python_threads(worker):
    counter, lock = Counter(), threading.Lock()
    threads = [threading.Thread(target=worker, args=(counter, lock)) for _ in range(N_THREADS)]
    t0 = time.perf_counter()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return time.perf_counter() - t0, counter.value


def unsafe_worker(counter, _lock):
    for _ in range(INCREMENTS):
        counter.value += 1  # read, add, write: not atomic


def locked_worker(counter, lock):
    for _ in range(INCREMENTS):
        with lock:
            counter.value += 1


# ---------------------------------------------------------------- native threads
def _element_ptr(context, builder, sig, args):
    arr_ty = sig.args[0]
    ary = context.make_array(arr_ty)(context, builder, args[0])
    return cgutils.get_item_pointer(context, builder, arr_ty, ary, [context.get_constant(types.intp, 0)])


@intrinsic
def racy_increment(typingctx, arr):
    """counter[0] = counter[0] + 1 as a separate load and store (no lock, no atomic RMW)."""
    sig = types.void(arr)

    def codegen(context, builder, signature, args):
        ptr = _element_ptr(context, builder, signature, args)
        old = builder.load_atomic(ptr, "monotonic", 8)  # forces a real memory load every time
        builder.store_atomic(builder.add(old, old.type(1)), ptr, "monotonic", 8)
        return context.get_dummy_value()

    return sig, codegen


@intrinsic
def atomic_increment(typingctx, arr):
    """Hardware atomic fetch-and-add (ldadd on arm64, lock xadd on x86)."""
    sig = types.void(arr)

    def codegen(context, builder, signature, args):
        ptr = _element_ptr(context, builder, signature, args)
        builder.atomic_rmw("add", ptr, ptr.type.pointee(1), "seq_cst")
        return context.get_dummy_value()

    return sig, codegen


@njit(parallel=True)
def native_unsafe(counter, n_threads, increments):
    for t in prange(n_threads):
        for _ in range(increments):
            racy_increment(counter)
    return get_thread_id()


@njit(parallel=True)
def native_atomic(counter, n_threads, increments):
    for t in prange(n_threads):
        for _ in range(increments):
            atomic_increment(counter)
    return get_thread_id()


def run_native(kernel):
    counter = np.zeros(1, dtype=np.int64)
    t0 = time.perf_counter()
    kernel(counter, N_THREADS, INCREMENTS)
    return time.perf_counter() - t0, int(counter[0])


# ---------------------------------------------------------------- reporting
def repeat(fn, *args):
    results = [fn(*args) for _ in range(REPEATS)]
    return statistics.median(r[0] for r in results), [r[1] for r in results]


def report(title, unsafe, safe, safe_name):
    (t_unsafe, vals_unsafe), (t_safe, vals_safe) = unsafe, safe
    worst = min(vals_unsafe)
    print(f"\n{title}")
    print(f"  Expected final value         : {EXPECTED:,}")
    print(f"  Actual final value (5 runs)  : {', '.join(f'{v:,}' for v in vals_unsafe)}")
    print(f"  Updates lost (per run)       : "
          + ", ".join(f"{100 * (EXPECTED - v) / EXPECTED:.2f}%" for v in vals_unsafe))
    print(f"  Worst run lost               : {EXPECTED - worst:,} updates = {100 * (EXPECTED - worst) / EXPECTED:.2f}%")
    print(f"  {safe_name:<29}: {', '.join(f'{v:,}' for v in vals_safe)} (all correct: {all(v == EXPECTED for v in vals_safe)})")
    print(f"  Wall-clock, no sync (median) : {t_unsafe * 1e3:.1f} ms")
    print(f"  Wall-clock, {safe_name.split()[0]:<17}: {t_safe * 1e3:.1f} ms")
    print(f"  Synchronization penalty      : +{(t_safe - t_unsafe) * 1e3:.1f} ms ({t_safe / t_unsafe:.2f}x slower)")
    assert all(v == EXPECTED for v in vals_safe), "synchronized version lost updates"


def main():
    print("TASK 6 - Synchronization Cost & Data Loss")
    print(f"Machine: {platform.machine()} {platform.system()}, {os.cpu_count()} cores, "
          f"Python {platform.python_version()}, {N_THREADS} threads x {INCREMENTS:,} increments")

    report("PART 1 - Python threading.Thread (CPython with GIL)",
           repeat(run_python_threads, unsafe_worker), repeat(run_python_threads, locked_worker),
           "threading.Lock result")

    set_num_threads(N_THREADS)
    run_native(native_unsafe), run_native(native_atomic)  # JIT warm-up
    report("PART 2 - native threads (Numba/OpenMP, no GIL)",
           repeat(run_native, native_unsafe), repeat(run_native, native_atomic),
           "atomic fetch-add result")


if __name__ == "__main__":
    main()
