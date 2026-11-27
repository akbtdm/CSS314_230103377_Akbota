"""Shared helpers: Numba OpenMP setup and low-level memory primitives.

Numba's nopython mode has no `#pragma omp atomic` / `critical`, so these
intrinsics emit the LLVM instructions directly:

* racy_add_f64 / racy_add_i64 : relaxed LOAD, ADD, relaxed STORE (three separate
  steps, NOT atomic). Relaxed (monotonic) atomics compile to plain ldr/str on
  ARM64 but stop LLVM from keeping the value in a register for the whole loop,
  so every iteration really touches memory (like `volatile` in C).
* atomic_fadd_f64 : one atomic read-modify-write (`#pragma omp atomic`).
* atomic_fetch_add_i64 : atomic fetch-and-add (dynamic-schedule work counter).
* atomic_cas_i64 : compare-and-swap (spin lock, guided schedule).
"""
import os

os.environ.setdefault("NUMBA_THREADING_LAYER", "omp")
os.environ.setdefault("KMP_WARNINGS", "0")  # hide libomp deprecation notice

from numba import types  # noqa: E402
from numba.core import cgutils  # noqa: E402
from numba.extending import intrinsic  # noqa: E402


def _ptr(context, builder, arr_t, arr, idx):
    ary = context.make_array(arr_t)(context, builder, arr)
    return cgutils.get_item_pointer(context, builder, arr_t, ary, [idx])


def _racy_add(typingctx, arr, idx, val):
    sig = types.void(arr, idx, val)

    def codegen(context, builder, signature, args):
        ptr = _ptr(context, builder, signature.args[0], args[0], args[1])
        ty = context.get_data_type(signature.args[0].dtype)
        align = context.get_abi_alignment(ty)
        old = builder.load_atomic(ptr, "monotonic", align)
        if isinstance(signature.args[0].dtype, types.Float):
            new = builder.fadd(old, args[2])
        else:
            new = builder.add(old, args[2])
        builder.store_atomic(new, ptr, "monotonic", align)
        return context.get_dummy_value()

    return sig, codegen


racy_add = intrinsic(_racy_add)


@intrinsic
def atomic_fadd_f64(typingctx, arr, idx, val):
    sig = types.float64(arr, idx, val)

    def codegen(context, builder, signature, args):
        ptr = _ptr(context, builder, signature.args[0], args[0], args[1])
        return builder.atomic_rmw("fadd", ptr, args[2], "seq_cst")

    return sig, codegen


@intrinsic
def atomic_fetch_add_i64(typingctx, arr, idx, val):
    sig = types.int64(arr, idx, val)

    def codegen(context, builder, signature, args):
        ptr = _ptr(context, builder, signature.args[0], args[0], args[1])
        return builder.atomic_rmw("add", ptr, args[2], "seq_cst")

    return sig, codegen


@intrinsic
def atomic_cas_i64(typingctx, arr, idx, expected, desired):
    """Returns the old value; the swap happened iff old == expected."""
    sig = types.int64(arr, idx, expected, desired)

    def codegen(context, builder, signature, args):
        ptr = _ptr(context, builder, signature.args[0], args[0], args[1])
        res = builder.cmpxchg(ptr, args[2], args[3], "seq_cst", "seq_cst")
        return builder.extract_value(res, 0)

    return sig, codegen


@intrinsic
def atomic_store_i64(typingctx, arr, idx, val):
    sig = types.void(arr, idx, val)

    def codegen(context, builder, signature, args):
        ptr = _ptr(context, builder, signature.args[0], args[0], args[1])
        builder.store_atomic(args[2], ptr, "release", 8)
        return context.get_dummy_value()

    return sig, codegen


def banner(title):
    import datetime
    import platform

    import numba

    line = "=" * 72
    print(line)
    print(title)
    print(f"date: {datetime.datetime.now().isoformat(timespec='seconds')}")
    print(f"python {platform.python_version()} | numba {numba.__version__} | "
          f"threading layer requested: {os.environ.get('NUMBA_THREADING_LAYER')} | "
          f"NUMBA_NUM_THREADS={numba.config.NUMBA_NUM_THREADS} | os.cpu_count()={os.cpu_count()}")
    print(line, flush=True)
