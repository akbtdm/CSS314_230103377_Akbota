"""Phase 3A: empirical Roofline ceilings — sustained DRAM bandwidth (parallel memoryview
memcpy on 2 x 96 MB shared buffers) and sustained pure-Python FMA throughput
(task3a_roofline.py)."""
import multiprocessing as mp
import time
from multiprocessing import shared_memory

from harness import emit_verify_tag  # noqa: F401  (Phase 3 hash is emitted by task3b_gemm.py)
from labutil import (build_parser, resolve_student_id, results_dir, save_json,
                     hardware_info, partition, median)

BUF_BYTES = 96_000_000
FLOPS_PER_WORKER = 80_000_000
UNROLL = 16
COPY_REPS = 9
ROUNDS = 3


def attach(name: str) -> shared_memory.SharedMemory:
    try:
        return shared_memory.SharedMemory(name=name, track=False)
    except TypeError:
        return shared_memory.SharedMemory(name=name)


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
    per_worker_gflops = [flops_each / (t1 - t0) for t0, t1 in zip(t0s, t1s)]
    return {"iters_per_worker": iters, "flops_per_worker": flops_each,
            "total_flops": p * flops_each, "wall_ms": wall_ns / 1e6,
            "gflops_aggregate": p * flops_each / wall_ns,
            "per_worker_gflops": per_worker_gflops,
            "per_worker_ms": [(t1 - t0) / 1e6 for t0, t1 in zip(t0s, t1s)],
            "checksum": sum(sums)}


def main() -> None:
    args = build_parser(__doc__).parse_args()
    sid = resolve_student_id(args.id)
    hw = hardware_info()
    p_phys = hw["physical_cores"]
    buf_bytes = BUF_BYTES // 10 if args.quick else BUF_BYTES
    flops = FLOPS_PER_WORKER // 100 if args.quick else FLOPS_PER_WORKER
    rounds = 1 if args.quick else ROUNDS

    print(f"=== PHASE 3A: Roofline ceilings  P_phys={p_phys}  buffers=2x{buf_bytes/1e6:.0f} MB  "
          f"FLOPs/worker={flops:.1e}  rounds={rounds} ===")
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
    ridge = perf / bw
    print(f"\nBW_sustained = {bw:.2f} GB/s (lab formula; wall-clock aggregate of the same round = "
          f"{bw_best['bw_wallclock_gbs']:.2f} GB/s)   P_sustained = {perf:.4f} GFLOP/s   "
          f"ridge point I* = {ridge:.6f} FLOP/Byte")
    save_json(results_dir(args.quick) / "phase3a.json", {
        "student_id": sid, "p_phys": p_phys, "buf_bytes": buf_bytes, "copy_reps": COPY_REPS,
        "bandwidth_rounds": bw_rounds, "compute_rounds": cp_rounds,
        "bw_sustained_gbs": bw, "bw_wallclock_gbs": bw_best["bw_wallclock_gbs"],
        "p_sustained_gflops": perf, "ridge_point": ridge,
    })


if __name__ == "__main__":
    main()
