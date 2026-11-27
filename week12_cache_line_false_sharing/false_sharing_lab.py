"""Cache Line & False Sharing Mechanics in Multi-Core Python (Week 12, Caching Tasks).

Student: Akbota Kutushova (230103377), Git handle akbtdm.
Run: python3 false_sharing_lab.py [--reps 3] [--iters 60000000]

Tasks 1-5 of the handout are the functions init_system_context, worker_strided_accumulate,
run_stride_benchmark, worker_private_accumulate and audit_llvm_and_parity. The handout's
starter file is cut off in the middle of Task 1, so everything after that point is my own
code following the task text. Extra experiments (stride 32, a plain non-atomic kernel, the
GIL experiments for Q5) are marked as extras.
"""
import os
import re
import sys
import json
import time
import hashlib
import argparse
import platform
import threading
import subprocess

import numpy as np
import numba
from numba import njit, uint64
from numba.core import types, cgutils
from numba.extending import intrinsic

try:
    import resource
except ImportError:
    resource = None

GIT_USERNAME = "akbtdm"
STUDENT_ID = "230103377"
NUM_THREADS = 4
ITERATIONS_PER_THREAD = 60_000_000
REQUIRED_STRIDES = (1, 4, 8, 16)
EXTRA_STRIDES = (32,)                # 256 B spacing: extra, two 128 B lines apart
BUFFER_ALIGN = 128                   # >= 64 required by the handout, = M4 line size
BUFFER_ELEMS = 1024                  # 8 KiB, room for every stride
GOLDEN = 0x9E3779B97F4A7C15
MASK64 = (1 << 64) - 1
LCG_MULT = np.uint64(6364136223846793005)   # Knuth MMIX multiplier
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
MANIFEST_PATH = os.path.join(HERE, "cache_lab_manifest.json")


def detect_cache_line_size() -> int:
    """Query L1 data cache coherency line size in bytes from OS sysfs/sysctl."""
    linux_path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(linux_path):
        with open(linux_path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    if sys.platform == "darwin":
        out = subprocess.check_output(["sysctl", "-n", "hw.cachelinesize"], encoding="utf-8")
        return int(out.strip())
    return 64


def _sysctl(name: str) -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", name], encoding="utf-8").strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


# ==============================================================================
# TASK 1: Hardware Topology & Student Identity Context
# ==============================================================================
def init_system_context(git_user: str) -> dict:
    assert git_user != "REPLACE_WITH_YOUR_GIT_USERNAME" and len(git_user.strip()) > 0, (
        "Set GIT_USERNAME to your real Git handle before running."
    )
    logical_cores = os.cpu_count() or 1
    if logical_cores < NUM_THREADS:
        raise RuntimeError(f"Requires at least {NUM_THREADS} logical cores; detected {logical_cores}.")
    line_size = detect_cache_line_size()
    cpu_sig = f"{platform.machine()}::{platform.processor()}::{logical_cores}"
    digest = hashlib.sha256(f"{git_user}::{cpu_sig}".encode("utf-8")).hexdigest()
    instance_seed = int(digest[:16], 16)

    affinity_supported = hasattr(os, "sched_setaffinity")
    affinity_mask = sorted(os.sched_getaffinity(0)) if affinity_supported else None
    return {
        "git_username": git_user,
        "student_id": STUDENT_ID,
        "logical_cores": logical_cores,
        "cpu_signature": cpu_sig,
        "cpu_brand": _sysctl("machdep.cpu.brand_string"),
        "perf_cores": _sysctl("hw.perflevel0.physicalcpu"),
        "eff_cores": _sysctl("hw.perflevel1.physicalcpu"),
        "cache_line_size_bytes": line_size,
        "l1d_p_bytes": _sysctl("hw.perflevel0.l1dcachesize"),
        "l2_p_bytes": _sysctl("hw.perflevel0.l2cachesize"),
        "sha256_digest": digest,
        "instance_seed": instance_seed,
        "instance_seed_hex": f"0x{instance_seed:016x}",
        "affinity_supported": affinity_supported,
        "affinity_mask": affinity_mask,
        "affinity_note": (
            "os.sched_setaffinity pinning applied" if affinity_supported else
            "macOS exposes no hard thread-to-core affinity (no sched_setaffinity; "
            "thread_policy_set affinity tags are ignored on Apple Silicon), so each worker "
            "thread only sets QoS USER_INTERACTIVE (pthread_set_qos_class_self_np), which "
            "steers it towards the P-cores; placement is still decided by the scheduler"
        ),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "numba": numba.__version__,
        "platform": platform.platform(),
    }


def thread_seeds(instance_seed: int, n: int) -> list:
    return [(instance_seed + t * GOLDEN) & MASK64 for t in range(n)]


def aligned_buffer(n_elems: int, align: int) -> np.ndarray:
    """uint64 array whose first element sits on an `align`-byte boundary."""
    raw = np.zeros(n_elems * 8 + align, dtype=np.uint8)
    offset = (-raw.ctypes.data) % align
    buf = raw[offset:offset + n_elems * 8].view(np.uint64)
    assert buf.ctypes.data % align == 0
    return buf


# ==============================================================================
# TASK 2: GIL-Free Native Counter Kernel
# ==============================================================================
def _u64_array_ptr(context, builder, arr_ty, arr_val, idx_ty, idx_val):
    ary = context.make_array(arr_ty)(context, builder, arr_val)
    idx = context.cast(builder, idx_val, idx_ty, types.intp)
    return cgutils.get_item_pointer(context, builder, arr_ty, ary, [idx])


def _is_u64_vector(arr) -> bool:
    return isinstance(arr, types.Array) and arr.dtype == types.uint64 and arr.ndim == 1


@intrinsic
def load_relaxed(typingctx, arr, idx):
    """arr[idx] as an LLVM `load atomic i64 ... monotonic` (a plain ldr on arm64)."""
    if not (_is_u64_vector(arr) and isinstance(idx, types.Integer)):
        return None

    def codegen(context, builder, sig, args):
        ptr = _u64_array_ptr(context, builder, sig.args[0], args[0], sig.args[1], args[1])
        return builder.load_atomic(ptr, "monotonic", 8)

    return types.uint64(arr, idx), codegen


@intrinsic
def store_relaxed(typingctx, arr, idx, val):
    """arr[idx] = val as an LLVM `store atomic i64 ... monotonic` (a plain str on arm64)."""
    if not (_is_u64_vector(arr) and isinstance(idx, types.Integer) and val == types.uint64):
        return None

    def codegen(context, builder, sig, args):
        ptr = _u64_array_ptr(context, builder, sig.args[0], args[0], sig.args[1], args[1])
        builder.store_atomic(args[2], ptr, "monotonic", 8)
        return context.get_dummy_value()

    return types.void(arr, idx, val), codegen


@njit(nogil=True)
def worker_strided_accumulate(buf, slot, iters, seed):
    """In-place RMW counter on buf[slot] every iteration: x = x + (seed ^ i) mod 2^64.

    The load and the store are relaxed atomics, so LLVM must emit one real load and one
    real store per iteration and may not keep buf[slot] in a register (see Q4)."""
    for i in range(iters):
        old = load_relaxed(buf, slot)
        store_relaxed(buf, slot, old + (seed ^ uint64(i)))


@njit(nogil=True)
def worker_plain_accumulate(buf, slot, iters, seed):
    """Extra (Q4): the same update written as ordinary NumPy indexing."""
    for i in range(iters):
        buf[slot] = buf[slot] + (seed ^ uint64(i))


@njit(nogil=False)
def worker_strided_accumulate_gil(buf, slot, iters, seed):
    """Extra (Q5): identical to the Task 2 kernel but compiled WITHOUT nogil=True."""
    for i in range(iters):
        old = load_relaxed(buf, slot)
        store_relaxed(buf, slot, old + (seed ^ uint64(i)))


@njit(nogil=True)
def worker_strided_lcg(buf, slot, iters, seed):
    """Extra: latency-bound RMW, x = x * MULT + (seed ^ i); one store per ~4 dependent cycles."""
    for i in range(iters):
        old = load_relaxed(buf, slot)
        store_relaxed(buf, slot, old * LCG_MULT + (seed ^ uint64(i)))


@njit(nogil=True)
def worker_strided_store(buf, slot, iters, seed):
    """Extra: store-throughput-bound, buf[slot] = seed ^ i (no load, no dependency chain)."""
    for i in range(iters):
        store_relaxed(buf, slot, seed ^ uint64(i))


@njit(nogil=True)
def private_lcg(buf, slot, iters, seed):
    acc = buf[slot]
    for i in range(iters):
        acc = acc * LCG_MULT + (seed ^ uint64(i))
    buf[slot] = acc


# ==============================================================================
# TASK 4: Thread-Local Register Privatization Baseline
# ==============================================================================
@njit(nogil=True)
def worker_private_accumulate(buf, slot, iters, seed):
    """Same recurrence in a local scalar; shared memory is written exactly once."""
    acc = buf[slot]
    for i in range(iters):
        acc = acc + (seed ^ uint64(i))
    buf[slot] = acc


def python_bytecode_accumulate(buf, slot, iters, seed):
    """Extra (Q5): the same update as pure CPython bytecode (holds the GIL)."""
    seed = int(seed)
    for i in range(iters):
        buf[slot] = (int(buf[slot]) + (seed ^ i)) & MASK64


# ==============================================================================
# TASK 3: Multi-Threaded Stride Sweep
# ==============================================================================
QOS_CLASS_USER_INTERACTIVE = 0x21


def _load_libsystem():
    if sys.platform != "darwin":
        return None
    import ctypes
    return ctypes.CDLL("/usr/lib/libSystem.B.dylib")


LIBSYSTEM = _load_libsystem()


def apply_placement_hint(core: int) -> str:
    """Pin to `core` where the OS allows it; on macOS only a QoS hint is possible."""
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {core})
        return f"pinned:{core}"
    if LIBSYSTEM is not None:
        rc = LIBSYSTEM.pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)
        if rc != 0:
            raise OSError(rc, "pthread_set_qos_class_self_np failed")
        return "qos:user_interactive"
    return "none"


def _rusage():
    if resource is None:
        return (0, 0)
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return (ru.ru_nvcsw, ru.ru_nivcsw)


def run_threads(kernel, stride: int, seeds: list, iters: int, n_threads: int) -> dict:
    """Run `kernel` on n_threads threads, thread t owning buf[t * stride]."""
    buf = aligned_buffer(BUFFER_ELEMS, BUFFER_ALIGN)
    slots = [t * stride for t in range(n_threads)]
    barrier = threading.Barrier(n_threads + 1)
    per_thread_ns = [0] * n_threads
    errors = []

    def body(t):
        try:
            apply_placement_hint(t)
            barrier.wait()
            t0 = time.perf_counter_ns()
            kernel(buf, slots[t], iters, np.uint64(seeds[t]))
            per_thread_ns[t] = time.perf_counter_ns() - t0
        except Exception as exc:  # surfaced below, never swallowed
            errors.append(repr(exc))

    threads = [threading.Thread(target=body, args=(t,)) for t in range(n_threads)]
    for th in threads:
        th.start()
    cs0 = _rusage()
    t0 = time.perf_counter_ns()
    barrier.wait()
    for th in threads:
        th.join()
    wall_ns = time.perf_counter_ns() - t0
    cs1 = _rusage()
    if errors:
        raise RuntimeError(f"worker failed: {errors}")
    return {
        "wall_ns": wall_ns,
        "per_thread_ns": per_thread_ns,
        "voluntary_csw": cs1[0] - cs0[0],
        "involuntary_csw": cs1[1] - cs0[1],
        "final": [int(buf[s]) for s in slots],
        "byte_offsets": [s * 8 for s in slots],
    }


def line_map(stride: int, n_threads: int, line: int) -> list:
    return [(t * stride * 8) // line for t in range(n_threads)]


def main_configs() -> list:
    configs = [(f"stride_{s}", worker_strided_accumulate, s) for s in REQUIRED_STRIDES + EXTRA_STRIDES]
    return configs + [("private", worker_private_accumulate, 8),
                      ("plain_nonatomic_stride_1", worker_plain_accumulate, 1)]


def lcg_configs() -> list:
    return [(f"lcg_stride_{s}", worker_strided_lcg, s) for s in REQUIRED_STRIDES] + \
           [("lcg_private", private_lcg, 8)]


def store_configs() -> list:
    return [(f"store_stride_{s}", worker_strided_store, s) for s in REQUIRED_STRIDES + EXTRA_STRIDES]


def run_stride_benchmark(ctx: dict, iters: int, reps: int, configs=None,
                         n_threads: int = NUM_THREADS) -> dict:
    configs = configs or main_configs()
    seeds = thread_seeds(ctx["instance_seed"], n_threads)
    runs = {name: [] for name, _, _ in configs}
    meta = {name: stride for name, _, stride in configs}
    for rep in range(reps):                         # interleaved so drift hits all configs
        for name, kernel, stride in configs:
            r = run_threads(kernel, stride, seeds, iters, n_threads)
            runs[name].append(r)
            print(f"  rep {rep + 1}/{reps} T={n_threads} {name:26s} wall {r['wall_ns'] / 1e6:9.1f} ms  "
                  f"per-thread ms {[round(x / 1e6, 1) for x in r['per_thread_ns']]}  "
                  f"csw v/iv {r['voluntary_csw']}/{r['involuntary_csw']}", flush=True)
    return {name: summarize(meta[name], rs, iters, ctx, n_threads) for name, rs in runs.items()}


def summarize(stride: int, rs: list, iters: int, ctx: dict, n_threads: int) -> dict:
    walls = sorted(r["wall_ns"] for r in rs)
    median = walls[len(walls) // 2]
    return {
        "threads": n_threads,
        "stride_elems": stride,
        "spacing_bytes": stride * 8,
        "lines_64B": line_map(stride, n_threads, 64),
        "lines_native": line_map(stride, n_threads, ctx["cache_line_size_bytes"]),
        "wall_ns_runs": [r["wall_ns"] for r in rs],
        "wall_ns_median": median,
        "wall_ns_min": walls[0],
        "ns_per_update_per_thread": median / iters,
        "ns_per_update_aggregate": median / (iters * n_threads),
        "per_thread_ns_runs": [r["per_thread_ns"] for r in rs],
        "voluntary_csw_runs": [r["voluntary_csw"] for r in rs],
        "involuntary_csw_runs": [r["involuntary_csw"] for r in rs],
        "final_values": rs[-1]["final"],
        "finals_identical_across_reps": all(r["final"] == rs[0]["final"] for r in rs),
    }


def single_thread_reference(ctx: dict, iters: int, reps: int) -> dict:
    """1 thread, stride 1, same kernels: the no-contention cost per update."""
    seeds = thread_seeds(ctx["instance_seed"], 1)
    out = {}
    for name, kernel in (("strided_1thread", worker_strided_accumulate),
                         ("private_1thread", worker_private_accumulate)):
        r = median_run(kernel, 1, seeds, iters, 1, reps)
        out[name] = {"wall_ns": r["wall_ns"], "wall_ns_runs": r["wall_ns_runs"],
                     "ns_per_update": r["wall_ns"] / iters}
        print(f"  {name:26s} median wall {r['wall_ns'] / 1e6:9.1f} ms  {r['wall_ns'] / iters:.3f} ns/update  "
              f"runs ms {[round(x / 1e6, 1) for x in r['wall_ns_runs']]}")
    return out


def median_run(kernel, stride: int, seeds: list, iters: int, n: int, reps: int) -> dict:
    runs = sorted((run_threads(kernel, stride, seeds, iters, n) for _ in range(reps)),
                  key=lambda r: r["wall_ns"])
    med = dict(runs[len(runs) // 2])
    med["wall_ns_runs"] = [r["wall_ns"] for r in runs]
    return med


def gil_experiment(ctx: dict, numba_iters: int, py_iters: int, reps: int) -> dict:
    """Extra (Q5): stride 1 with 1, 2, 4 threads; median of `reps` runs each."""
    out = {"numba_with_gil": {}, "numba_nogil": {}, "python_bytecode": {}}
    for n in (1, 2, 4):
        seeds = thread_seeds(ctx["instance_seed"], n)
        for key, kernel, iters in (("numba_with_gil", worker_strided_accumulate_gil, numba_iters),
                                   ("numba_nogil", worker_strided_accumulate, numba_iters),
                                   ("python_bytecode", python_bytecode_accumulate, py_iters)):
            r = median_run(kernel, 1, seeds, iters, n, reps)
            out[key][n] = {"iters_per_thread": iters, "wall_ns": r["wall_ns"], "wall_ns_runs": r["wall_ns_runs"],
                           "ns_per_update_aggregate": r["wall_ns"] / (iters * n),
                           "per_thread_ns": r["per_thread_ns"],
                           "voluntary_csw": r["voluntary_csw"], "involuntary_csw": r["involuntary_csw"],
                           "final": r["final"]}
            print(f"  {key:16s} T={n}  iters/thread {iters:>10,d}  median wall {r['wall_ns'] / 1e6:8.1f} ms  "
                  f"{r['wall_ns'] / (iters * n):8.3f} ns/update (aggregate)  "
                  f"per-thread ms {[round(x / 1e6, 1) for x in r['per_thread_ns']]}  "
                  f"csw v/iv {r['voluntary_csw']}/{r['involuntary_csw']}", flush=True)
    return out


# ==============================================================================
# TASK 5: Exact State Parity & LLVM IR Verification
# ==============================================================================
def expected_finals(ctx: dict, iters: int, kernel=None, n_threads: int = NUM_THREADS) -> list:
    """Reference counters computed single-threaded, one thread seed at a time."""
    kernel = kernel or worker_private_accumulate
    vals = []
    for seed in thread_seeds(ctx["instance_seed"], n_threads):
        b = np.zeros(1, dtype=np.uint64)
        kernel(b, 0, iters, np.uint64(seed))
        vals.append(int(b[0]))
    return vals


def python_reference(seed: int, iters: int) -> int:
    x = 0
    for i in range(iters):
        x = (x + (seed ^ i)) & MASK64
    return x


def core_function_ir(kernel) -> str:
    """The optimized LLVM IR of the kernel itself (not the CPython/cfunc wrappers)."""
    module = next(iter(kernel.inspect_llvm().values()))
    for chunk in module.split("\ndefine ")[1:]:
        header = chunk.split("\n", 1)[0]
        if kernel.__name__ in header and "cpython" not in header and "cfunc" not in header:
            return "define " + chunk.split("\n}\n", 1)[0] + "\n}"
    raise RuntimeError(f"core function for {kernel.__name__} not found in IR")


def _blocks(fn_ir: str) -> dict:
    blocks, name = {}, "entry"
    for line in fn_ir.splitlines()[1:]:
        m = re.match(r'^([A-Za-z0-9_."\-]+):', line)
        if m:
            name = m.group(1).strip('"')
            blocks[name] = []
        else:
            blocks.setdefault(name, []).append(line)
    return blocks


def _in_cycle(blocks: dict, start: str) -> bool:
    succ = {b: re.findall(r'label %"?([A-Za-z0-9_.\-]+)"?', "\n".join(ls)) for b, ls in blocks.items()}
    seen, stack = set(), list(succ.get(start, []))
    while stack:
        b = stack.pop()
        if b == start:
            return True
        if b not in seen:
            seen.add(b)
            stack.extend(succ.get(b, []))
    return False


def ir_store_report(kernel) -> dict:
    fn = core_function_ir(kernel)
    blocks = _blocks(fn)
    stores = []
    for b, lines in blocks.items():
        for ln in lines:
            if re.search(r"\bstore\b.*\bi64\b", ln):
                stores.append({"block": b, "in_loop": _in_cycle(blocks, b), "ir": ln.strip()})
    return {
        "function_header": fn.split("\n", 1)[0][:160],
        "i64_stores": stores,
        "i64_stores_inside_loop": sum(s["in_loop"] for s in stores),
        "atomic_monotonic_stores": len(re.findall(r"store atomic i64 .* monotonic", fn)),
        "atomic_monotonic_loads": len(re.findall(r"load atomic i64, .* monotonic", fn)),
        "python_api_calls": re.findall(r"@(Py\w+|NRT_\w+|numba_\w+)", fn),
        "ir_lines": len(fn.splitlines()),
        "ir_text": fn,
    }


def asm_loop_excerpt(kernel) -> str:
    asm = next(iter(kernel.inspect_asm().values()))
    lines = [ln for ln in asm.splitlines() if re.search(r"\b(ldr|str|madd|mul|eor|add|subs|b\.\w+|cbnz)\b", ln)]
    return "\n".join(lines[:40])


def audit_llvm_and_parity(ctx: dict, bench: dict, iters: int, lcg: dict, tsweep: dict,
                          store: dict) -> dict:
    expected = expected_finals(ctx, iters)
    expected_lcg = expected_finals(ctx, iters, private_lcg)
    py_check_iters = 200_000
    seed0 = thread_seeds(ctx["instance_seed"], 1)[0]
    b = np.zeros(1, dtype=np.uint64)
    worker_strided_accumulate(b, 0, py_check_iters, np.uint64(seed0))
    py_ref_ok = int(b[0]) == python_reference(seed0, py_check_iters)

    parity = {name: (res["final_values"] == expected and res["finals_identical_across_reps"])
              for name, res in bench.items()}
    parity.update({name: (res["final_values"] == expected_lcg and res["finals_identical_across_reps"])
                   for name, res in lcg.items()})
    expected_store = [s ^ (iters - 1) for s in thread_seeds(ctx["instance_seed"], NUM_THREADS)]
    parity.update({name: (res["final_values"] == expected_store and res["finals_identical_across_reps"])
                   for name, res in store.items()})
    for n, group in tsweep.items():
        exp_n = expected_finals(ctx, iters, n_threads=n)
        parity.update({f"T{n}_{name}": (res["final_values"] == exp_n and res["finals_identical_across_reps"])
                       for name, res in group.items()})
    ir = {k.__name__: ir_store_report(k) for k in
          (worker_strided_accumulate, worker_private_accumulate, worker_plain_accumulate,
           worker_strided_lcg, worker_strided_store)}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for name, rep in ir.items():
        with open(os.path.join(RESULTS_DIR, f"llvm_{name}.ll"), "w", encoding="utf-8") as f:
            f.write(rep["ir_text"] + "\n")
    with open(os.path.join(RESULTS_DIR, "asm_worker_strided_accumulate.txt"), "w", encoding="utf-8") as f:
        f.write(asm_loop_excerpt(worker_strided_accumulate) + "\n")

    s = ir["worker_strided_accumulate"]
    llvm_ok = (s["atomic_monotonic_stores"] >= 1 and s["atomic_monotonic_loads"] >= 1
               and s["i64_stores_inside_loop"] >= 1 and not s["python_api_calls"])
    return {
        "expected_final_values": expected,
        "expected_final_values_lcg": expected_lcg,
        "python_reference_check": {"iters": py_check_iters, "match": py_ref_ok},
        "parity_per_layout": parity,
        "parity_all": all(parity.values()),
        "llvm": {k: {kk: vv for kk, vv in v.items() if kk != "ir_text"} for k, v in ir.items()},
        "llvm_ok": llvm_ok,
        "verification_passed": bool(all(parity.values()) and llvm_ok and py_ref_ok),
    }


def warm_up(ctx: dict) -> None:
    b = aligned_buffer(BUFFER_ELEMS, BUFFER_ALIGN)
    s = np.uint64(ctx["instance_seed"])
    for k in (worker_strided_accumulate, worker_plain_accumulate, worker_private_accumulate,
              worker_strided_accumulate_gil, worker_strided_lcg, private_lcg, worker_strided_store):
        k(b, 0, 10, s)


def print_table(bench: dict, ref8: str, ref16: str) -> None:
    print(f"\n  {'layout':26s} {'B apart':>7s} {'64B lines':>16s} {'128B lines':>16s} {'median ms':>10s} "
          f"{'min ms':>8s} {'ns/upd/thr':>10s} {'vs ' + ref8:>16s} {'vs ' + ref16:>17s}")
    for name, r in bench.items():
        print(f"  {name:26s} {r['spacing_bytes']:7d} {str(r['lines_64B']):>16s} {str(r['lines_native']):>16s} "
              f"{r['wall_ns_median'] / 1e6:10.1f} {r['wall_ns_min'] / 1e6:8.1f} {r['ns_per_update_per_thread']:10.3f} "
              f"{r['wall_ns_median'] / bench[ref8]['wall_ns_median']:16.2f} "
              f"{r['wall_ns_median'] / bench[ref16]['wall_ns_median']:17.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--iters", type=int, default=ITERATIONS_PER_THREAD)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--extra-reps", type=int, default=7)
    ap.add_argument("--gil-numba-iters", type=int, default=30_000_000)
    ap.add_argument("--gil-python-iters", type=int, default=300_000)
    args = ap.parse_args()

    t_start = time.perf_counter()
    ctx = init_system_context(GIT_USERNAME)
    print("[Task 1] system context")
    for k in ("git_username", "cpu_brand", "cpu_signature", "logical_cores", "perf_cores", "eff_cores",
              "cache_line_size_bytes", "instance_seed_hex", "affinity_note"):
        print(f"  {k:22s} {ctx[k]}")
    warm_up(ctx)

    print(f"\n[Task 3/4] T={NUM_THREADS} threads, {args.iters:,d} updates per thread, {args.reps} reps")
    bench = run_stride_benchmark(ctx, args.iters, args.reps)
    print_table(bench, "stride_8", "stride_16")

    print(f"\n[Extra] latency-bound LCG update, T={NUM_THREADS}")
    lcg = run_stride_benchmark(ctx, args.iters, args.extra_reps, lcg_configs())
    print_table(lcg, "lcg_stride_8", "lcg_stride_16")

    print(f"\n[Extra] store-throughput-bound kernel (no load), T={NUM_THREADS}")
    store = run_stride_benchmark(ctx, args.iters, args.extra_reps, store_configs())
    print_table(store, "store_stride_8", "store_stride_16")

    print("\n[Extra] thread-count sweep, stride 1 vs stride 16 (Task 2 kernel)")
    tsweep = {}
    for n in (2, 8):
        tsweep[n] = run_stride_benchmark(ctx, args.iters, args.extra_reps,
                                         [("stride_1", worker_strided_accumulate, 1),
                                          ("stride_8", worker_strided_accumulate, 8),
                                          ("stride_16", worker_strided_accumulate, 16)], n)
        print_table(tsweep[n], "stride_8", "stride_16")

    print("\n[Reference] single thread, no contention")
    single = single_thread_reference(ctx, args.iters, args.extra_reps)

    print("\n[Q5 extra] GIL experiment, stride 1")
    gil = gil_experiment(ctx, args.gil_numba_iters, args.gil_python_iters, 3)

    print("\n[Task 5] parity + LLVM IR audit")
    audit = audit_llvm_and_parity(ctx, bench, args.iters, lcg, tsweep, store)
    print(f"  expected finals: {[hex(v) for v in audit['expected_final_values']]}")
    print(f"  parity per layout: {audit['parity_per_layout']}")
    print(f"  python reference check ({audit['python_reference_check']['iters']:,d} iters): "
          f"{audit['python_reference_check']['match']}")
    for name, rep in audit["llvm"].items():
        print(f"  {name}: i64 stores {len(rep['i64_stores'])}, inside loop {rep['i64_stores_inside_loop']}, "
              f"atomic monotonic load/store {rep['atomic_monotonic_loads']}/{rep['atomic_monotonic_stores']}, "
              f"python API calls {rep['python_api_calls'] or 'none'}")
        for st in rep["i64_stores"]:
            print(f"      [{st['block']}{' LOOP' if st['in_loop'] else ''}] {st['ir']}")
    print(f"  verification_passed: {audit['verification_passed']}")

    manifest = {
        "lab": "Cache Line & False Sharing Mechanics (Week 12)",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "system_context": ctx,
        "config": {"num_threads": NUM_THREADS, "iterations_per_thread": args.iters, "reps": args.reps,
                   "buffer_align_bytes": BUFFER_ALIGN, "update": "x = x * 6364136223846793005 + (seed ^ i) mod 2^64",
                   "timer": "time.perf_counter_ns", "context_switches": "resource.getrusage(RUSAGE_SELF) delta"},
        "stride_benchmark": bench,
        "lcg_latency_bound_benchmark": lcg,
        "store_bound_benchmark": store,
        "thread_count_sweep": {str(n): g for n, g in tsweep.items()},
        "single_thread_reference": single,
        "gil_experiment": gil,
        "slowdown_stride1_vs_stride8": bench["stride_1"]["wall_ns_median"] / bench["stride_8"]["wall_ns_median"],
        "slowdown_stride1_vs_stride16": bench["stride_1"]["wall_ns_median"] / bench["stride_16"]["wall_ns_median"],
        "audit": audit,
        "verification_passed": audit["verification_passed"],
        "total_runtime_s": time.perf_counter() - t_start,
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote {os.path.relpath(MANIFEST_PATH, HERE)}; total runtime {manifest['total_runtime_s']:.1f} s")
    return 0 if audit["verification_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
