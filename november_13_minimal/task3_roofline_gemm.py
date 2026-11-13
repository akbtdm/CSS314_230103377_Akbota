"""Phase 3: empirical Roofline ceilings (3A: parallel memcpy bandwidth on 2 x 64 MB shared buffers,
16-way unrolled scalar FMA ceiling) and progressive cache-blocked GEMM with a tile sweep
(3B: naive i-j-k, reordered i-k-j, 6-loop blocked, N = 256) (task3_roofline_gemm.py)."""
import multiprocessing as mp
import random
import time
from array import array
from multiprocessing import shared_memory

from harness import run_benchmark, emit_verify_tag
from labutil import (build_parser, resolve_student_id, results_dir, save_json,
                     hardware_info, partition, median)

BUF_BYTES = 64_000_000
FLOPS_PER_WORKER = 60_000_000
UNROLL = 16
COPY_REPS = 9
ROUNDS = 3
N = 256
TILE_SIZES = [8, 16, 32, 64, 96, 128]
WARMUP, TIMED = 2, 7
SEED = 20261113
DOUBLE = 8

_A = None   # local array('d') copy of A in each pool worker
_B = None   # local array('d') copy of B in each pool worker
_C = None   # SharedMemory handle for C in each pool worker


def attach(name: str) -> shared_memory.SharedMemory:
    try:
        return shared_memory.SharedMemory(name=name, track=False)
    except TypeError:  # Python < 3.13 has no track= argument
        return shared_memory.SharedMemory(name=name)


# ----------------------------------------------------------------------------- Part 3A
def bw_worker(name_a, name_b, start, end, reps, barrier, out_med_ms, out_t0, out_t1, idx) -> None:
    shm_a, shm_b = attach(name_a), attach(name_b)
    buf_a, buf_b = shm_a.buf, shm_b.buf
    buf_a[start:end] = buf_b[start:end]  # warm-up: fault pages in before timing
    barrier.wait()
    samples = []
    t_first = time.perf_counter_ns()
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        buf_a[start:end] = buf_b[start:end]  # native memcpy of this worker's slice
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1e6)
    out_med_ms[idx] = median(samples)
    out_t0[idx], out_t1[idx] = t_first, t1
    del buf_a, buf_b
    shm_a.close()
    shm_b.close()


def fma_worker(iters, barrier, out_t0, out_t1, out_sum, idx) -> None:
    a, b = 0.9999999, 1e-7
    x0, x1, x2, x3 = 1.0, 1.1, 1.2, 1.3
    x4, x5, x6, x7 = 1.4, 1.5, 1.6, 1.7
    x8, x9, x10, x11 = 1.8, 1.9, 2.0, 2.1
    x12, x13, x14, x15 = 2.2, 2.3, 2.4, 2.5
    barrier.wait()
    t0 = time.perf_counter_ns()
    for _ in range(iters):
        x0 = x0 * a + b
        x1 = x1 * a + b
        x2 = x2 * a + b
        x3 = x3 * a + b
        x4 = x4 * a + b
        x5 = x5 * a + b
        x6 = x6 * a + b
        x7 = x7 * a + b
        x8 = x8 * a + b
        x9 = x9 * a + b
        x10 = x10 * a + b
        x11 = x11 * a + b
        x12 = x12 * a + b
        x13 = x13 * a + b
        x14 = x14 * a + b
        x15 = x15 * a + b
    t1 = time.perf_counter_ns()
    out_t0[idx], out_t1[idx] = t0, t1
    out_sum[idx] = (x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7
                    + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15)


def measure_bandwidth(p: int, buf_bytes: int, reps: int) -> dict:
    shm_a = shared_memory.SharedMemory(create=True, size=buf_bytes)
    shm_b = shared_memory.SharedMemory(create=True, size=buf_bytes)
    try:
        pattern = bytes(range(256)) * (buf_bytes // 256) + bytes(range(buf_bytes % 256))
        # macOS rounds the mapping up to a 16 KB page, so slice to the requested size
        shm_b.buf[:buf_bytes] = pattern
        shm_a.buf[:buf_bytes] = b"\x00" * buf_bytes
        del pattern
        barrier = mp.Barrier(p)
        out = mp.Array("d", p)
        t0s, t1s = mp.Array("q", p), mp.Array("q", p)
        procs = [mp.Process(target=bw_worker,
                            args=(shm_a.name, shm_b.name, s, e, reps, barrier, out, t0s, t1s, idx))
                 for idx, (s, e) in enumerate(partition(buf_bytes, p))]
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join()
        copy_ok = (bytes(shm_a.buf[:4096]) == bytes(shm_b.buf[:4096])
                   and bytes(shm_a.buf[buf_bytes - 4096:buf_bytes])
                   == bytes(shm_b.buf[buf_bytes - 4096:buf_bytes]))
        per_worker = list(out)
        t_med_ms = median(per_worker)
        bw_gbs = 2 * buf_bytes / (t_med_ms * 1e-3) / 1e9  # lab formula: assumes full overlap
        wall_ns = max(t1s) - min(t0s)                    # all reps of all workers, first start -> last end
        bw_wall_gbs = 2 * buf_bytes * reps / wall_ns     # bytes per ns == GB/s
        return {"t_median_worker_ms": t_med_ms, "per_worker_ms": per_worker,
                "bw_gbs": bw_gbs, "wall_ms": wall_ns / 1e6, "bw_wallclock_gbs": bw_wall_gbs,
                "copy_verified": copy_ok}
    finally:
        shm_a.close()
        shm_a.unlink()
        shm_b.close()
        shm_b.unlink()


def measure_compute(p: int, flops_per_worker: int) -> dict:
    iters = flops_per_worker // (2 * UNROLL)
    flops_each = iters * 2 * UNROLL
    barrier = mp.Barrier(p)
    t0s, t1s = mp.Array("q", p), mp.Array("q", p)
    sums = mp.Array("d", p)
    procs = [mp.Process(target=fma_worker, args=(iters, barrier, t0s, t1s, sums, idx))
             for idx in range(p)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
    wall_ns = max(t1s) - min(t0s)
    return {"iters_per_worker": iters, "flops_per_worker": flops_each,
            "total_flops": p * flops_each, "wall_ms": wall_ns / 1e6,
            "gflops_aggregate": p * flops_each / wall_ns,
            "per_worker_gflops": [flops_each / (t1 - t0) for t0, t1 in zip(t0s, t1s)],
            "per_worker_ms": [(t1 - t0) / 1e6 for t0, t1 in zip(t0s, t1s)],
            "checksum": sum(sums)}


def roofline_ceilings(p_phys: int, buf_bytes: int, flops: int, rounds: int) -> dict:
    print(f"--- Part 3A: ceilings  P_phys={p_phys}  buffers=2x{buf_bytes / 1e6:.0f} MB  "
          f"FLOPs/worker={flops:.1e}  rounds={rounds}")
    bw_rounds = []
    for r in range(rounds):
        res = measure_bandwidth(p_phys, buf_bytes, COPY_REPS)
        bw_rounds.append(res)
        print(f"[BW round {r}] median worker copy = {res['t_median_worker_ms']:.3f} ms  "
              f"-> BW(lab formula) = {res['bw_gbs']:.2f} GB/s;  all {COPY_REPS} reps x {p_phys} workers "
              f"wall = {res['wall_ms']:.3f} ms -> BW(wall-clock aggregate) = {res['bw_wallclock_gbs']:.2f} GB/s"
              f"  verified={res['copy_verified']}")
        print("    per-worker ms: " + " ".join(f"{t:.3f}" for t in res["per_worker_ms"]))
    bw_best = sorted(bw_rounds, key=lambda r: r["bw_gbs"])[len(bw_rounds) // 2]

    cp_rounds = []
    for r in range(rounds):
        res = measure_compute(p_phys, flops)
        cp_rounds.append(res)
        print(f"[FMA round {r}] wall = {res['wall_ms']:.1f} ms  -> P = {res['gflops_aggregate']:.4f} "
              f"GFLOP/s  checksum={res['checksum']:.6f}")
        print("    per-worker GFLOP/s: " + " ".join(f"{g:.4f}" for g in res["per_worker_gflops"]))
    cp_best = sorted(cp_rounds, key=lambda r: r["gflops_aggregate"])[len(cp_rounds) // 2]

    bw, perf = bw_best["bw_gbs"], cp_best["gflops_aggregate"]
    print(f"BW_sustained = {bw:.2f} GB/s (lab formula; wall-clock aggregate of the same round = "
          f"{bw_best['bw_wallclock_gbs']:.2f} GB/s)   P_sustained = {perf:.4f} GFLOP/s   "
          f"ridge point I* = {perf / bw:.6f} FLOP/Byte")
    return {"buf_bytes": buf_bytes, "copy_reps": COPY_REPS,
            "bandwidth_rounds": bw_rounds, "compute_rounds": cp_rounds,
            "bw_sustained_gbs": bw, "bw_wallclock_gbs": bw_best["bw_wallclock_gbs"],
            "p_sustained_gflops": perf, "ridge_point": perf / bw}


# ----------------------------------------------------------------------------- Part 3B
def init_worker(name_a: str, name_b: str, name_c: str, nbytes: int) -> None:
    global _A, _B, _C
    shm_a, shm_b = attach(name_a), attach(name_b)
    _A = array("d")
    _A.frombytes(shm_a.buf[:nbytes])  # mapping may be page-rounded on macOS
    _B = array("d")
    _B.frombytes(shm_b.buf[:nbytes])
    shm_a.close()
    shm_b.close()
    _C = attach(name_c)


def store_rows(row_start: int, row_end: int, n: int, values) -> None:
    data = values if isinstance(values, array) else array("d", values)
    _C.buf[row_start * n * DOUBLE:row_end * n * DOUBLE] = data.tobytes()


def gemm_naive(task) -> None:
    """Variant 1: i-j-k, B traversed with stride N, inline index arithmetic."""
    row_start, row_end, n = task
    A, B = _A, _B
    out = []
    for i in range(row_start, row_end):
        for j in range(n):
            dot = 0.0
            for k in range(n):
                dot += A[i * n + k] * B[k * n + j]
            out.append(dot)
    store_rows(row_start, row_end, n, out)


def gemm_ikj(task) -> None:
    """Variant 2: i-k-j, a_ik hoisted, unit-stride row slices of B and C."""
    row_start, row_end, n = task
    A, B = _A, _B
    b_rows = [B[k * n:(k + 1) * n] for k in range(n)]
    out = array("d")
    for i in range(row_start, row_end):
        c_row = [0.0] * n
        base = i * n
        for k in range(n):
            a_ik = A[base + k]
            b_row = b_rows[k]
            for j in range(n):
                c_row[j] += a_ik * b_row[j]
        out.extend(c_row)
    store_rows(row_start, row_end, n, out)


def gemm_blocked(task) -> None:
    """Variant 3: 6-loop BSxBS blocking, micro-kernel order (i,k,j) with a_ik hoisted."""
    row_start, row_end, n, bs = task
    A, B = _A, _B
    c_local = [0.0] * ((row_end - row_start) * n)
    for ii in range(row_start, row_end, bs):
        i_end = min(ii + bs, row_end)
        for jj in range(0, n, bs):
            j_end = min(jj + bs, n)
            for kk in range(0, n, bs):
                k_end = min(kk + bs, n)
                for i in range(ii, i_end):
                    c_off = (i - row_start) * n
                    a_off = i * n
                    for k in range(kk, k_end):
                        a_ik = A[a_off + k]
                        b_off = k * n
                        for j in range(jj, j_end):
                            c_local[c_off + j] += a_ik * B[b_off + j]
    store_rows(row_start, row_end, n, c_local)


def make_matrix(n: int, rng: random.Random) -> array:
    return array("d", (rng.random() for _ in range(n * n)))


def read_c(shm_c, nbytes: int) -> array:
    data = array("d")
    data.frombytes(shm_c.buf[:nbytes])
    return data


def max_abs_diff(x: array, y: array) -> float:
    return max(abs(a - b) for a, b in zip(x, y))


def gflops(n: int, median_ms: float) -> float:
    return 2.0 * n ** 3 / (median_ms * 1e6)


def bench_variant(pool, fn, tasks, n, warmup, timed, label) -> dict:
    stats = run_benchmark(lambda: pool.map(fn, tasks, chunksize=1), warmup=warmup, runs=timed)
    row = {"variant": label, "min_ms": stats.min_ms, "median_ms": stats.median_ms,
           "iqr_ms": stats.iqr_ms, "gflops": gflops(n, stats.median_ms)}
    print(f"[{label:>10}] med={stats.median_ms:9.2f} ms iqr={stats.iqr_ms:7.2f} "
          f"-> {row['gflops']:.4f} GFLOP/s")
    return row


def gemm_sweep(p_phys: int, n: int, tiles, warmup: int, timed: int) -> dict:
    rng = random.Random(SEED)
    a_mat, b_mat = make_matrix(n, rng), make_matrix(n, rng)
    nbytes = n * n * DOUBLE
    shm_a = shared_memory.SharedMemory(create=True, size=nbytes)
    shm_b = shared_memory.SharedMemory(create=True, size=nbytes)
    shm_c = shared_memory.SharedMemory(create=True, size=nbytes)
    print(f"--- Part 3B: GEMM N={n}  footprint={3 * nbytes / 1e6:.2f} MB  P_phys={p_phys} "
          f"warmup={warmup} timed={timed}")
    row_ranges = partition(n, p_phys)
    tasks_plain = [(s, e, n) for s, e in row_ranges]
    rows, checks = [], {}
    try:
        shm_a.buf[:nbytes] = a_mat.tobytes()
        shm_b.buf[:nbytes] = b_mat.tobytes()
        with mp.Pool(processes=p_phys, initializer=init_worker,
                     initargs=(shm_a.name, shm_b.name, shm_c.name, nbytes)) as pool:
            pool.map(gemm_ikj, tasks_plain, chunksize=1)
            c_ref = read_c(shm_c, nbytes)
            pool.map(gemm_naive, tasks_plain, chunksize=1)
            checks["max_abs_diff_v1_vs_v2"] = max_abs_diff(read_c(shm_c, nbytes), c_ref)
            pool.map(gemm_blocked, [(s, e, n, 16) for s, e in row_ranges], chunksize=1)
            checks["max_abs_diff_v3_bs16_vs_v2"] = max_abs_diff(read_c(shm_c, nbytes), c_ref)
            checks["c_checksum"] = sum(c_ref)
            print(f"numerical check: max|C1-C2|={checks['max_abs_diff_v1_vs_v2']:.3e}  "
                  f"max|C3(BS=16)-C2|={checks['max_abs_diff_v3_bs16_vs_v2']:.3e}  "
                  f"sum(C)={checks['c_checksum']:.6f}")
            rows.append(bench_variant(pool, gemm_naive, tasks_plain, n, warmup, timed, "V1 ijk"))
            rows.append(bench_variant(pool, gemm_ikj, tasks_plain, n, warmup, timed, "V2 ikj"))
            for bs in tiles:
                tasks = [(s, e, n, bs) for s, e in row_ranges]
                row = bench_variant(pool, gemm_blocked, tasks, n, warmup, timed, f"V3 BS={bs}")
                row["bs"] = bs
                row["tile_footprint_bytes"] = 3 * bs * bs * DOUBLE
                row["intensity_flop_per_byte"] = bs / 12.0
                rows.append(row)
    finally:
        for shm in (shm_a, shm_b, shm_c):
            shm.close()
            shm.unlink()
    best = max((r for r in rows if "bs" in r), key=lambda r: r["gflops"])
    print(f"best tile BS*={best['bs']} -> {best['gflops']:.4f} GFLOP/s "
          f"(footprint {best['tile_footprint_bytes']} B);  V1={rows[0]['gflops']:.4f}  "
          f"V2={rows[1]['gflops']:.4f} GFLOP/s")
    return {"n": n, "footprint_bytes": 3 * nbytes, "warmup_runs": warmup, "timed_runs": timed,
            "row_ranges": row_ranges, "checks": checks, "rows": rows, "best_bs": best["bs"]}


def main() -> None:
    args = build_parser(__doc__).parse_args()
    sid = resolve_student_id(args.id)
    hw = hardware_info()
    p_phys = hw["physical_cores"]
    buf_bytes = BUF_BYTES // 10 if args.quick else BUF_BYTES
    flops = FLOPS_PER_WORKER // 100 if args.quick else FLOPS_PER_WORKER
    rounds = 1 if args.quick else ROUNDS
    n = 64 if args.quick else N
    tiles = [8, 32] if args.quick else TILE_SIZES
    warmup, timed = (1, 3) if args.quick else (WARMUP, TIMED)

    print(f"=== PHASE 3: Roofline + GEMM  P_phys={p_phys} ===")
    result = {"student_id": sid, "p_phys": p_phys}
    result.update(roofline_ceilings(p_phys, buf_bytes, flops, rounds))
    result.update(gemm_sweep(p_phys, n, tiles, warmup, timed))
    v1 = result["rows"][0]["gflops"]
    v3 = max(r["gflops"] for r in result["rows"] if "bs" in r)
    tag = emit_verify_tag("PHASE3", sid, v1, v3)
    print("\n" + tag)
    result["verify"] = tag
    result["hash_inputs"] = {"primary": v1, "secondary": v3}
    save_json(results_dir(args.quick) / "phase3.json", result)


if __name__ == "__main__":
    main()
