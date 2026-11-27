# OpenMP Multithreading & Performance Engineering in Python
# Instructor - Sufyan bin Uzayr
# Student: Akbota Kutushova (230103377) - run locally on Apple M4 (4P + 6E cores)

# SECTION 0: LOCAL OPENMP ENVIRONMENT & DETERMINISTIC SEED HARNESS
import os
os.environ["NUMBA_THREADING_LAYER"] = "omp"  # Force native OpenMP runtime (libgomp / iomp)

import time, hashlib, json, math, platform
import numpy as np
import numba as nb
from numba import njit, prange, set_num_threads, get_num_threads, get_thread_id
from numba.extending import intrinsic
from numba.core import types, cgutils

STUDENT_NAME = "Akbota Kutushova"
STUDENT_ID = 230103377

assert isinstance(STUDENT_ID, int) and STUDENT_ID > 0, "Set a valid numeric STUDENT_ID!"
assert (os.cpu_count() or 1) >= 4, "Local multi-core CPU (>=4 logical cores) required; no Colab!"


# Trigger OpenMP layer initialization and verify backend
@njit(parallel=True)
def _omp_probe():
    acc = 0
    for i in prange(64):
        acc += get_thread_id() + 1
    return acc


_ = _omp_probe()
assert nb.threading_layer() == "omp", f"Expected 'omp' layer, got {nb.threading_layer()}"

rng = np.random.default_rng(STUDENT_ID)

# Student-specific OpenMP parameters
N_T1 = 4_000_000 + (STUDENT_ID % 991) * 1024
BINS_T1 = 48 + (STUDENT_ID % 17)
ITERS_T2 = 35_000_000 + (STUDENT_ID % 503) * 10_000
PAD_STRIDE = 8 + (STUDENT_ID % 5) * 8  # >= 8 int64 slots (>= 64 bytes cache-line isolation)
ROWS_T3 = 12_000 + (STUDENT_ID % 257) * 16
SKEW_MOD = 37 + (STUDENT_ID % 19)
GRID_H = 384 + (STUDENT_ID % 4) * 32
GRID_W = 512 + (STUDENT_ID % 5) * 32
OMEGA_SOR = np.float64(1.72 + (STUDENT_ID % 11) * 0.015)

data_t1 = rng.integers(0, BINS_T1, size=N_T1, dtype=np.int32)
weights_t1 = rng.uniform(0.25, 2.50, size=N_T1).astype(np.float64)

print(f"Backend: {nb.threading_layer()} | Logical Cores: {os.cpu_count()} | Student: {STUDENT_ID}")

# ---------------------------------------------------------------------------
# Shared benchmark helpers
# ---------------------------------------------------------------------------
THREAD_COUNTS = (1, 2, 4, 8)
TIMING_REPEATS = 3  # median of 3 timed runs after a warm-up call
RESULTS = {"params": {
    "N_T1": N_T1, "BINS_T1": BINS_T1, "ITERS_T2": ITERS_T2, "PAD_STRIDE": PAD_STRIDE,
    "ROWS_T3": ROWS_T3, "SKEW_MOD": SKEW_MOD, "GRID_H": GRID_H, "GRID_W": GRID_W,
    "OMEGA_SOR": float(OMEGA_SOR)}}

# Apple M4 uses 128-byte cache lines (sysctl hw.cachelinesize), so the padding
# must also cover 128 B, not only the 64 B mentioned in the handout.
CACHE_LINE_BYTES = 128 if platform.machine() == "arm64" else 64
assert PAD_STRIDE * 8 >= CACHE_LINE_BYTES, "PAD_STRIDE does not isolate a full cache line"


def time_call(fn, *args):
    """Run fn once as warm-up, then return (median seconds, last result)."""
    result = fn(*args)
    samples = []
    for _ in range(TIMING_REPEATS):
        t0 = time.perf_counter()
        result = fn(*args)
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples)), result


def print_scaling(title, times):
    base = times[1]
    print(f"  {title}")
    for t in THREAD_COUNTS:
        print(f"    threads={t}  time={times[t]*1e3:9.2f} ms  speedup={base / times[t]:5.2f}x")


# ===========================================================================
# TASK 1: Race-condition forensics & thread-private histogram reduction
# ===========================================================================
@njit(parallel=True)
def buggy_hist_omp(indices, weights, num_bins):
    hist = np.zeros(num_bins, dtype=np.int64)
    w_sums = np.zeros(num_bins, dtype=np.float64)
    for i in prange(indices.shape[0]):
        b = indices[i]
        hist[b] += 1  # Data race across OpenMP threads
        w_sums[b] += weights[i]  # Non-associative floating-point race
    return hist, w_sums


@njit
def serial_hist(indices, weights, num_bins):
    hist = np.zeros(num_bins, dtype=np.int64)
    w_sums = np.zeros(num_bins, dtype=np.float64)
    for i in range(indices.shape[0]):
        hist[indices[i]] += 1
        w_sums[indices[i]] += weights[i]
    return hist, w_sums


# Weights live in [0.25, 2.5): every such float64 is an exact multiple of 2^-54,
# so w * 2^54 is an exact integer. Splitting it into a high and low 27-bit limb
# lets each thread accumulate exact int64 sums that cannot overflow and whose
# result does not depend on the order of addition.
FIX_SHIFT = 54
LIMB_BITS = 27
LIMB_MASK = (1 << LIMB_BITS) - 1


@njit(parallel=True)
def safe_hist_omp(indices, weights, num_bins, pad):
    max_threads = get_num_threads()
    priv_hist = np.zeros((max_threads, num_bins), dtype=np.int64)
    priv_hi = np.zeros((max_threads, num_bins), dtype=np.int64)
    priv_lo = np.zeros((max_threads, num_bins), dtype=np.int64)
    worker_hits = np.zeros((max_threads, pad), dtype=np.int64)  # padded counters
    scale = 2.0 ** FIX_SHIFT
    for i in prange(indices.shape[0]):
        tid = get_thread_id()
        b = indices[i]
        m = np.int64(weights[i] * scale)  # exact (power-of-two scaling)
        priv_hist[tid, b] += 1
        priv_hi[tid, b] += m >> LIMB_BITS
        priv_lo[tid, b] += m & LIMB_MASK
        worker_hits[tid, 0] += 1
    # Deterministic row-wise reduction over tid = 0 .. max_threads-1
    hist = np.zeros(num_bins, dtype=np.int64)
    hi = np.zeros(num_bins, dtype=np.int64)
    lo = np.zeros(num_bins, dtype=np.int64)
    for t in range(max_threads):
        for b in range(num_bins):
            hist[b] += priv_hist[t, b]
            hi[b] += priv_hi[t, b]
            lo[b] += priv_lo[t, b]
    w_sums = np.empty(num_bins, dtype=np.float64)
    for b in range(num_bins):
        w_sums[b] = np.float64(hi[b]) * 2.0 ** (LIMB_BITS - FIX_SHIFT) + np.float64(lo[b]) * 2.0 ** (-FIX_SHIFT)
    return hist, w_sums, worker_hits[:, 0].copy()


def run_task1():
    print("\n=== TASK 1: race forensics & thread-private histogram ===")
    assert weights_t1.min() >= 0.25 and weights_t1.max() < 4.0, "fixed-point limb layout assumes w in [0.25, 4)"
    set_num_threads(1)
    ref_hist, ref_w = serial_hist(data_t1, weights_t1, BINS_T1)
    assert int(ref_hist.sum()) == N_T1

    set_num_threads(8)
    buggy_hist_omp(data_t1, weights_t1, BINS_T1)  # compile
    runs = []
    for run in range(10):
        h, w = buggy_hist_omp(data_t1, weights_t1, BINS_T1)
        lost = int(N_T1 - h.sum())
        max_count_diff = int(np.max(np.abs(h - ref_hist)))
        max_w_diff = float(np.max(np.abs(w - ref_w)))
        runs.append({"run": run, "lost": lost, "max_count_diff": max_count_diff, "max_wsum_diff": max_w_diff})
        print(f"  buggy run {run:2d} @8T: lost={lost:8d}  max|hist-ref|={max_count_diff:7d}  max|wsum-ref|={max_w_diff:.4e}")
    RESULTS["t1_buggy_runs"] = runs

    safe_out, safe_times, hits = {}, {}, {}
    for t in THREAD_COUNTS:
        set_num_threads(t)
        safe_times[t], (h, w, wh) = time_call(safe_hist_omp, data_t1, weights_t1, BINS_T1, PAD_STRIDE)
        safe_out[t] = (h, w)
        hits[t] = wh.tolist()
        assert int(h.sum()) == N_T1 and np.array_equal(h, ref_hist), "safe histogram counts wrong"
        assert np.count_nonzero(wh) == t, f"expected {t} participating workers, got {wh}"
        print(f"  safe @{t}T worker_hits={wh.tolist()}")
    for t in THREAD_COUNTS:
        assert np.array_equal(safe_out[t][0], safe_out[1][0]), "counts differ across thread counts"
        assert np.array_equal(safe_out[t][1].view(np.int64), safe_out[1][1].view(np.int64)), \
            f"w_sums not bit-identical at {t} threads"
    max_vs_serial = float(np.max(np.abs(safe_out[8][1] - ref_w)))
    print(f"  safe w_sums bit-identical across 1/2/4/8 threads: True | max|safe-serial float|={max_vs_serial:.3e}")
    print_scaling("safe_hist_omp timings", safe_times)
    RESULTS["t1_safe"] = {"times_s": {str(t): safe_times[t] for t in THREAD_COUNTS},
                          "worker_hits": {str(t): hits[t] for t in THREAD_COUNTS},
                          "bit_identical": True, "max_abs_vs_serial_float": max_vs_serial}
    return safe_out[8]


# ===========================================================================
# TASK 2: Eliminating cache-line false sharing
# ===========================================================================
@intrinsic
def xor_store_relaxed(typingctx, arr, idx, val):
    """arr[idx] ^= val with a relaxed (monotonic) atomic load + store.

    On arm64 this is a plain ldr/str, exactly like the C code, but LLVM is not
    allowed to keep the element in a register for the whole loop. Without it,
    LLVM promotes state[tid] to a register and the store happens only once
    (measured below as the 'as-given' kernel), which hides false sharing.
    """
    if not (isinstance(arr, types.Array) and arr.dtype == types.int64 and isinstance(idx, types.BaseTuple)):
        return None
    sig = types.void(arr, types.UniTuple(types.intp, len(idx)), types.int64)

    def codegen(context, builder, signature, args):
        arr_ty, idx_ty, _ = signature.args
        ary = context.make_array(arr_ty)(context, builder, args[0])
        indices = cgutils.unpack_tuple(builder, args[1], count=len(idx_ty))
        ptr = cgutils.get_item_pointer(context, builder, arr_ty, ary, indices)
        old = builder.load_atomic(ptr, "monotonic", 8)
        builder.store_atomic(builder.xor(old, args[2]), ptr, "monotonic", 8)
        return context.get_dummy_value()

    return sig, codegen


@njit(parallel=True)
def unpadded_state_as_given(iters, seed_val):
    """Starter code unchanged: LLVM hoists the store out of the inner loop."""
    n_threads = get_num_threads()
    state = np.zeros(n_threads, dtype=np.int64)
    for t in prange(n_threads):
        tid = get_thread_id()
        x = np.int64(seed_val + tid * 1315423911)
        for step in range(iters):
            x = (x * np.int64(6364136223846793005) + np.int64(1442695040888963407)) & np.int64(0x7FFFFFFFFFFFFFFF)
            state[tid] ^= (x >> 17)
    return state


@njit(parallel=True)
def unpadded_state_omp(iters, seed_val):
    n_threads = get_num_threads()
    state = np.zeros(n_threads, dtype=np.int64)
    for t in prange(n_threads):
        tid = get_thread_id()
        x = np.int64(seed_val + tid * 1315423911)
        for step in range(iters):
            x = (x * np.int64(6364136223846793005) + np.int64(1442695040888963407)) & np.int64(0x7FFFFFFFFFFFFFFF)
            xor_store_relaxed(state, (tid,), x >> 17)  # real store to the shared cache line every iteration
    return state


@njit(parallel=True)
def padded_state_omp(iters, seed_val, pad_stride):
    n_threads = get_num_threads()
    padded_state = np.zeros((n_threads, pad_stride), dtype=np.int64)
    for t in prange(n_threads):
        tid = get_thread_id()
        x = np.int64(seed_val + tid * 1315423911)
        for step in range(iters):
            x = (x * np.int64(6364136223846793005) + np.int64(1442695040888963407)) & np.int64(0x7FFFFFFFFFFFFFFF)
            xor_store_relaxed(padded_state, (tid, 0), x >> 17)  # same store, but on its own cache line
    return padded_state


def run_task2():
    print("\n=== TASK 2: false sharing vs padded state ===")
    t_unpad, t_pad, t_given = {}, {}, {}
    state_8 = None
    for t in THREAD_COUNTS:
        set_num_threads(t)
        t_given[t], s_given = time_call(unpadded_state_as_given, ITERS_T2, STUDENT_ID)
        t_unpad[t], s_un = time_call(unpadded_state_omp, ITERS_T2, STUDENT_ID)
        t_pad[t], s_pad = time_call(padded_state_omp, ITERS_T2, STUDENT_ID, PAD_STRIDE)
        assert np.array_equal(s_un, s_pad[:, 0]), f"padded != unpadded at {t} threads"
        assert np.array_equal(s_un, s_given), f"as-given kernel differs at {t} threads"
        if t == 8:
            state_8 = s_un
        print(f"  threads={t}  unpadded={t_unpad[t]*1e3:8.2f} ms  padded={t_pad[t]*1e3:8.2f} ms  "
              f"ratio={t_unpad[t] / t_pad[t]:5.2f}x  | as-given (store hoisted by LLVM)={t_given[t]*1e3:8.2f} ms  (exact equality OK)")
    speedup_8 = t_unpad[8] / t_pad[8]
    RESULTS["t2"] = {"unpadded_s": {str(t): t_unpad[t] for t in THREAD_COUNTS},
                     "padded_s": {str(t): t_pad[t] for t in THREAD_COUNTS},
                     "as_given_s": {str(t): t_given[t] for t in THREAD_COUNTS},
                     "padding_speedup_8T": speedup_8, "cache_line_bytes": CACHE_LINE_BYTES}
    return state_8, t_unpad, t_pad


# ===========================================================================
# TASK 3: Irregular workload skew & dynamic chunk-stealing scheduler
# ===========================================================================
CHUNK_T3 = 16

# Per-thread finish-time telemetry: mach_absolute_time() is a monotonic tick
# counter in libSystem; Numba can call ctypes functions from nopython code.
import ctypes
assert platform.system() == "Darwin", "finish-time telemetry uses mach_absolute_time (macOS)"
_mach_absolute_time = ctypes.CDLL("/usr/lib/libSystem.B.dylib").mach_absolute_time
_mach_absolute_time.restype = ctypes.c_uint64
_mach_absolute_time.argtypes = ()


@intrinsic
def atomic_fetch_add(typingctx, arr, idx, val):
    """Lock-free fetch-and-add on an int64 array element (LLVM atomicrmw, seq_cst)."""
    if not (isinstance(arr, types.Array) and arr.dtype == types.int64 and arr.ndim == 1):
        return None
    sig = types.int64(arr, types.intp, types.int64)

    def codegen(context, builder, signature, args):
        arr_ty = signature.args[0]
        ary = context.make_array(arr_ty)(context, builder, args[0])
        ptr = cgutils.get_item_pointer(context, builder, arr_ty, ary, [args[1]])
        return builder.atomic_rmw("add", ptr, args[2], "seq_cst")

    return sig, codegen


def make_work_units():
    r = np.arange(ROWS_T3, dtype=np.int64)
    return 40 + ((r * 7919 + STUDENT_ID) % SKEW_MOD) ** 2


@njit
def row_score(r, units):
    acc = 0.0
    for s in range(units):
        x = r * 1e-3 + s * 1e-4
        acc += math.sin(x) * math.cos(x) + math.sqrt(abs(x) + 1.0)
    return acc


@njit(parallel=True)
def static_skew_omp(work_units, pad):
    n_threads = get_num_threads()
    out = np.zeros(work_units.shape[0], dtype=np.float64)
    ops = np.zeros((n_threads, pad), dtype=np.int64)
    finish = np.zeros((n_threads, pad), dtype=np.int64)
    t0 = np.int64(_mach_absolute_time())
    for r in prange(work_units.shape[0]):
        tid = get_thread_id()
        out[r] = row_score(r, work_units[r])
        ops[tid, 0] += work_units[r]
        finish[tid, 0] = np.int64(_mach_absolute_time()) - t0  # last row done by this thread
    return out, ops[:, 0].copy(), finish[:, 0].copy()


@njit(parallel=True)
def balanced_skew_omp(work_units, order, chunk, pad):
    n = work_units.shape[0]
    n_threads = get_num_threads()
    out = np.zeros(n, dtype=np.float64)
    ops = np.zeros((n_threads, pad), dtype=np.int64)
    cursor = np.zeros(pad, dtype=np.int64)  # shared cursor alone on its cache line
    finish = np.zeros((n_threads, pad), dtype=np.int64)
    t0 = np.int64(_mach_absolute_time())
    for t in prange(n_threads):  # one persistent worker per OpenMP thread
        tid = get_thread_id()
        while True:
            start = atomic_fetch_add(cursor, 0, chunk)  # claim the next batch
            if start >= n:
                break
            end = min(start + chunk, n)
            for k in range(start, end):
                r = order[k]
                out[r] = row_score(r, work_units[r])
                ops[tid, 0] += work_units[r]
        finish[tid, 0] = np.int64(_mach_absolute_time()) - t0
    return out, ops[:, 0].copy(), finish[:, 0].copy()


def imbalance(values):
    """max / min over active threads (ops counts or finish times)."""
    active = values[values > 0]
    return float(active.max() / active.min())


def run_task3():
    print("\n=== TASK 3: workload skew, static vs dynamic chunk scheduler ===")
    work_units = make_work_units()
    # Largest-first order (stable): heavy rows are claimed early, so the last
    # chunks are the cheapest ones and the tail at the barrier is short.
    order = np.argsort(-work_units, kind="stable").astype(np.int64)
    t_st, t_bal, tel = {}, {}, {}
    out_static = out_bal = None
    for t in THREAD_COUNTS:
        set_num_threads(t)
        t_st[t], (out_static, o_s, f_s) = time_call(static_skew_omp, work_units, PAD_STRIDE)
        t_bal[t], (out_bal, o_b, f_b) = time_call(balanced_skew_omp, work_units, order, CHUNK_T3, PAD_STRIDE)
        tel[t] = {"ops_static": o_s, "ops_bal": o_b, "fin_static": f_s, "fin_bal": f_b}
        assert o_s.sum() == work_units.sum() and o_b.sum() == work_units.sum(), "ops not conserved"
        diff = float(np.max(np.abs(out_static - out_bal)))
        assert diff < 1e-11, f"static vs balanced mismatch {diff}"
        print(f"  threads={t}  static={t_st[t]*1e3:7.2f} ms (time-LIR {imbalance(f_s):.3f}, ops-LIR {imbalance(o_s):.3f})"
              f"  balanced={t_bal[t]*1e3:7.2f} ms (time-LIR {imbalance(f_b):.3f}, ops-LIR {imbalance(o_b):.3f})"
              f"  max|diff|={diff:.1e}")
    e = tel[8]
    lir_static, lir_bal = imbalance(e["fin_static"]), imbalance(e["fin_bal"])
    ops_lir_static, ops_lir_bal = imbalance(e["ops_static"]), imbalance(e["ops_bal"])
    print(f"  8T ops per thread static  : {e['ops_static'].tolist()}")
    print(f"  8T ops per thread balanced: {e['ops_bal'].tolist()}")
    print(f"  8T finish ticks static    : {e['fin_static'].tolist()}")
    print(f"  8T finish ticks balanced  : {e['fin_bal'].tolist()}")
    # Gate on time imbalance (barrier wait). On the heterogeneous M4 an equal
    # number of operations per thread is NOT balance: P-cores finish earlier.
    assert lir_bal < 1.15, f"balanced time-LIR {lir_bal:.3f} not below 1.15"
    RESULTS["t3"] = {"static_s": {str(t): t_st[t] for t in THREAD_COUNTS},
                     "balanced_s": {str(t): t_bal[t] for t in THREAD_COUNTS},
                     "time_lir_static_8T": lir_static, "time_lir_balanced_8T": lir_bal,
                     "ops_lir_static_8T": ops_lir_static, "ops_lir_balanced_8T": ops_lir_bal,
                     "ops_static_8T": e["ops_static"].tolist(), "ops_balanced_8T": e["ops_bal"].tolist(),
                     "finish_ticks_static_8T": e["fin_static"].tolist(),
                     "finish_ticks_balanced_8T": e["fin_bal"].tolist(),
                     "total_ops": int(work_units.sum()), "chunk": CHUNK_T3}
    return out_bal, t_st, t_bal, lir_static, lir_bal


# ===========================================================================
# TASK 4: Parallel Red-Black SOR solver
# ===========================================================================
SWEEPS_T4 = 600


def init_grid():
    U = np.zeros((GRID_H, GRID_W), dtype=np.float64)
    U[0, :] = 100.0
    U[-1, :] = -50.0
    U[:, 0] = 25.0 * np.sin(np.linspace(0, np.pi, GRID_H))
    U[:, -1] = 75.0 * np.cos(np.linspace(0, np.pi / 2, GRID_H))
    return U


@njit(parallel=True)
def sor_red_black_step(U, omega, color, thread_max):
    """Update every interior cell with (r + c) % 2 == color; neighbours are the other colour."""
    H, W = U.shape
    for r in prange(1, H - 1):
        tid = get_thread_id()
        c0 = 1 + ((r + 1 + color) % 2)  # first interior column of this colour
        local_max = thread_max[tid, 0]
        for c in range(c0, W - 1, 2):
            old = U[r, c]
            new = (1.0 - omega) * old + 0.25 * omega * (U[r - 1, c] + U[r + 1, c] + U[r, c - 1] + U[r, c + 1])
            U[r, c] = new
            d = abs(new - old)
            if d > local_max:
                local_max = d
        thread_max[tid, 0] = local_max  # padded slot, written once per row


@njit
def solve_pde_omp(U, omega, sweeps, pad):
    thread_max = np.zeros((get_num_threads(), pad), dtype=np.float64)
    history = np.zeros(sweeps, dtype=np.float64)
    for it in range(sweeps):
        thread_max[:, :] = 0.0
        sor_red_black_step(U, omega, 0, thread_max)  # red sweep (implicit barrier at end)
        sor_red_black_step(U, omega, 1, thread_max)  # black sweep reads fresh red values
        history[it] = thread_max[:, 0].max()
    return history


def solve_fresh(sweeps):
    U = init_grid()
    hist = solve_pde_omp(U, OMEGA_SOR, sweeps, PAD_STRIDE)
    return U, hist


def run_task4():
    print("\n=== TASK 4: Red-Black SOR solver ===")
    times, grids, hists = {}, {}, {}
    for t in THREAD_COUNTS:
        set_num_threads(t)
        times[t], (grids[t], hists[t]) = time_call(solve_fresh, SWEEPS_T4)
    for t in THREAD_COUNTS:
        diff = float(np.max(np.abs(grids[t] - grids[1])))
        assert diff < 1e-9, f"grid at {t} threads differs from 1 thread by {diff}"
        print(f"  threads={t}  time={times[t]*1e3:8.2f} ms  speedup={times[1]/times[t]:5.2f}x  "
              f"max|U-U1|={diff:.1e}  final max_delta={hists[t][-1]:.3e}")
    RESULTS["t4"] = {"times_s": {str(t): times[t] for t in THREAD_COUNTS},
                     "max_delta_first": float(hists[8][0]), "max_delta_last": float(hists[8][-1]),
                     "max_delta_at": {str(k): float(hists[8][k - 1]) for k in (1, 10, 100, 300, 600)},
                     "grid_sum": float(grids[8].sum())}
    return grids[8], times


# ===========================================================================
# RUN ALL TASKS + DIAGNOSIS STRINGS (numbers are inserted from this run)
# ===========================================================================
(hist_t1, wsums_t1) = run_task1()
state_t2, t2_unpad, t2_pad = run_task2()
out_t3, t3_static, t3_bal, lir_static, lir_bal = run_task3()
grid_t4, t4_times = run_task4()

lost_counts = [r["lost"] for r in RESULTS["t1_buggy_runs"]]
DIAGNOSIS_T1 = (
    f"In 10 runs of buggy_hist_omp at 8 threads between {min(lost_counts)} and {max(lost_counts)} of "
    f"{N_T1} increments were lost, because hist[b] += 1 is a load, add, store sequence and two threads can "
    "load the same old value before either stores. Even if the integer increment were made atomic, "
    "floating-point addition is not associative: (a + b) + c is rounded after each step, so it can differ "
    "from a + (b + c) in the last bits. With threads the order in which weights reach w_sums[b] depends on "
    "the OpenMP schedule and on timing, so the rounded total changes between runs and between thread counts. "
    "safe_hist_omp removes the race with per-thread rows indexed by get_thread_id(), and it removes the "
    "order dependence by accumulating each weight as an exact fixed-point integer (w * 2^54 split into two "
    "27-bit limbs); integer sums are associative, so the merged result is bit-identical at 1, 2, 4 and 8 threads."
)
DIAGNOSIS_T2 = (
    f"Unpadded, 8 threads took {t2_unpad[8]*1e3:.1f} ms versus {t2_unpad[1]*1e3:.1f} ms for 1 thread doing the "
    f"same per-thread work; the padded version took {t2_pad[8]*1e3:.1f} ms at 8 threads, a "
    f"{t2_unpad[8]/t2_pad[8]:.2f}x speedup. In the unpadded array all eight int64 slots sit in one cache line "
    f"(this M4 uses {CACHE_LINE_BYTES}-byte lines), so under MESI every store needs the line in the Modified "
    "state on the writing core; that invalidates the copies in the other cores, which must then fetch the "
    "line again before their own store. The line keeps moving between cores even though no value is shared. "
    f"With PAD_STRIDE={PAD_STRIDE} int64 ({PAD_STRIDE*8} bytes) each thread's slot is on its own line, which stays "
    "Modified in one core's L1, so the stores hit the cache and the threads run independently."
)
DIAGNOSIS_T3 = (
    f"Static prange gives each thread one contiguous block of rows. Because the cost pattern repeats every "
    f"{SKEW_MOD} rows, the operation counts per block are almost equal (LIR {lir_static:.3f}), but the M4 has 4 "
    "performance and 6 efficiency cores, so the threads on slower cores finish their blocks later and the "
    f"rest wait at the barrier ({t3_static[8]*1e3:.1f} ms at 8 threads). balanced_skew_omp starts one worker per "
    f"thread and each worker claims {CHUNK_T3} rows at a time with an atomic fetch-and-add on a shared cursor "
    "(LLVM atomicrmw via a Numba intrinsic, no locks). Rows are visited largest-first, so the final claims are "
    "small and the tail is short. Faster cores simply claim more chunks; "
    f"8-thread time was {t3_bal[8]*1e3:.1f} ms with LIR {lir_bal:.3f}, and every row score matches static exactly."
)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results_{STUDENT_ID}.json"), "w") as fh:
    json.dump({**RESULTS, "diagnosis": {"T1": DIAGNOSIS_T1, "T2": DIAGNOSIS_T2, "T3": DIAGNOSIS_T3}}, fh, indent=2)

# FINAL SECTION: DETERMINISTIC VERIFICATION & TELEMETRY BUNDLE
set_num_threads(8 if (os.cpu_count() or 1) >= 8 else os.cpu_count())

for idx, diag in enumerate([DIAGNOSIS_T1, DIAGNOSIS_T2, DIAGNOSIS_T3], start=1):
    assert isinstance(diag, str) and len(diag.strip()) >= 80, f"Complete DIAGNOSIS_T{idx}!"

checksum_blob = (
    f"{STUDENT_ID}|"
    f"{int(np.sum(hist_t1))}|{float(np.sum(wsums_t1)):.4f}|"
    f"{int(np.bitwise_xor.reduce(state_t2))}|"
    f"{float(np.sum(out_t3)):.6f}|"
    f"{float(np.sum(grid_t4)):.6f}"
)
lab_digest = hashlib.sha256(checksum_blob.encode()).hexdigest()[:20].upper()

bundle = {
    "instructor": "Instructor - Sufyan bin Uzayr",
    "student_name": STUDENT_NAME,
    "student_id": STUDENT_ID,
    "omp_layer": nb.threading_layer(),
    "cpu_cores": os.cpu_count(),
    "verification_digest": lab_digest,
}
print("=== OPENMP PYTHON LAB VERIFICATION BUNDLE ===")
print(json.dumps(bundle, indent=2))
