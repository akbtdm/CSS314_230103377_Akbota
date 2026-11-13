"""Phase 3B: progressive parallel GEMM (naive i-j-k, reordered i-k-j, 6-loop blocked)
over flat array('d') buffers in SharedMemory, with a 10-point tile sweep (task3b_gemm.py)."""
import multiprocessing as mp
import random
from array import array
from multiprocessing import shared_memory

from harness import run_benchmark, emit_verify_tag
from labutil import (build_parser, resolve_student_id, results_dir, save_json,
                     hardware_info, partition)

N = 384
TILE_SIZES = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
WARMUP, TIMED = 2, 7
SEED = 20261113
DOUBLE = 8

_A = None   # local array('d') copy of A in each pool worker
_B = None   # local array('d') copy of B in each pool worker
_C = None   # SharedMemory handle for C in each pool worker


def attach(name: str) -> shared_memory.SharedMemory:
    try:
        return shared_memory.SharedMemory(name=name, track=False)
    except TypeError:
        return shared_memory.SharedMemory(name=name)


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


def bench_variant(pool, fn, tasks, n, warmup, timed, label):
    stats = run_benchmark(lambda: pool.map(fn, tasks, chunksize=1),
                          warmup_runs=warmup, timed_runs=timed)
    row = {"variant": label, "min_ms": stats.min_ms, "median_ms": stats.median_ms,
           "iqr_ms": stats.iqr_ms, "max_ms": stats.max_ms, "gflops": gflops(n, stats.median_ms)}
    print(f"[{label:>10}] med={stats.median_ms:9.2f} ms iqr={stats.iqr_ms:7.2f} "
          f"-> {row['gflops']:.4f} GFLOP/s")
    return row


def main() -> None:
    args = build_parser(__doc__).parse_args()
    sid = resolve_student_id(args.id)
    hw = hardware_info()
    p_phys = hw["physical_cores"]
    n = 96 if args.quick else N
    tiles = [4, 16, 32] if args.quick else TILE_SIZES
    warmup, timed = (1, 3) if args.quick else (WARMUP, TIMED)

    rng = random.Random(SEED)
    a_mat, b_mat = make_matrix(n, rng), make_matrix(n, rng)
    nbytes = n * n * DOUBLE
    shm_a = shared_memory.SharedMemory(create=True, size=nbytes)
    shm_b = shared_memory.SharedMemory(create=True, size=nbytes)
    shm_c = shared_memory.SharedMemory(create=True, size=nbytes)
    print(f"=== PHASE 3B: GEMM N={n}  footprint={3 * nbytes / 1e6:.2f} MB  P_phys={p_phys} "
          f"warmup={warmup} timed={timed} ===")

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

    v3_rows = [r for r in rows if "bs" in r]
    best = max(v3_rows, key=lambda r: r["gflops"])
    v1 = rows[0]
    print(f"\nbest tile BS*={best['bs']} -> {best['gflops']:.4f} GFLOP/s "
          f"(footprint {best['tile_footprint_bytes']} B);  V1={v1['gflops']:.4f}  "
          f"V2={rows[1]['gflops']:.4f} GFLOP/s")
    tag = emit_verify_tag("PHASE3", sid, v1["gflops"], best["gflops"])
    print("\n" + tag)
    save_json(results_dir(args.quick) / "phase3b.json", {
        "student_id": sid, "p_phys": p_phys, "n": n, "footprint_bytes": 3 * nbytes,
        "warmup_runs": warmup, "timed_runs": timed, "row_ranges": row_ranges,
        "checks": checks, "rows": rows, "best_bs": best["bs"],
        "verify": tag, "hash_inputs": {"primary": v1["gflops"], "secondary": best["gflops"]},
    })


if __name__ == "__main__":
    main()
