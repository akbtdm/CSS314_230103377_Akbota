"""Phase 1: false sharing / cache-line contention sweep with Karp-Flatt diagnostics
(task1_falsesharing.py)."""
import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory

from harness import run_benchmark, emit_verify_tag
from labutil import (build_parser, resolve_student_id, results_dir, save_json,
                     hardware_info, fmt_stats, doubling_up_to)

M_TOTAL = 24_000_000          # total read-modify-write steps per trial
SHM_SIZE = 4096
SLOTS = SHM_SIZE // 8          # 512 x 64-bit words
STRIDES = [1, 2, 4, 8, 16, 32]
WARMUP, TIMED = 2, 7
WORD_BYTES = 8


def attach(name: str) -> shared_memory.SharedMemory:
    try:
        return shared_memory.SharedMemory(name=name, track=False)
    except TypeError:  # Python < 3.13 has no track= argument
        return shared_memory.SharedMemory(name=name)


def fs_worker(shm_name: str, slot_idx: int, iters: int) -> None:
    shm = attach(shm_name)
    arr = (ctypes.c_uint64 * SLOTS).from_buffer(shm.buf)
    acc = arr[slot_idx]
    mask = 0xFFFFFFFFFFFFFFFF
    mult = 6364136223846793005
    for i in range(iters):
        acc = (acc * mult + (i | 1)) & mask
        arr[slot_idx] = acc  # direct hardware store to the shared L1 cache line
    del arr  # release buffer export before closing SharedMemory
    shm.close()


def run_trial(shm_name: str, p: int, stride: int, m_total: int) -> None:
    iters = m_total // p
    procs = [mp.Process(target=fs_worker, args=(shm_name, t * stride, iters))
             for t in range(p)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()


def slot_values(shm, p: int, stride: int):
    arr = (ctypes.c_uint64 * SLOTS).from_buffer(shm.buf)
    values = [arr[t * stride] for t in range(p)]
    del arr
    return values


def bench(shm, p: int, stride: int, m_total: int, warmup: int, timed: int) -> dict:
    stats = run_benchmark(lambda: run_trial(shm.name, p, stride, m_total),
                          warmup_runs=warmup, timed_runs=timed)
    mops = m_total / (stats.median_ms * 1e3)
    return {"p": p, "stride": stride, "sep_bytes": stride * WORD_BYTES,
            "min_ms": stats.min_ms, "median_ms": stats.median_ms,
            "iqr_ms": stats.iqr_ms, "max_ms": stats.max_ms, "mops": mops,
            "slot0_final": f"{slot_values(shm, p, stride)[0]:016X}"}


def karp_flatt(speedup: float, p: int):
    if p < 2:
        return None
    return (1.0 / speedup - 1.0 / p) / (1.0 - 1.0 / p)


def add_scaling_metrics(rows):
    t1 = rows[0]["median_ms"]
    for row in rows:
        row["speedup"] = t1 / row["median_ms"]
        row["efficiency"] = row["speedup"] / row["p"]
        row["karp_flatt"] = karp_flatt(row["speedup"], row["p"])
    return rows


def print_table_1a(rows):
    print("\nTable 1A — stride sensitivity at p = P_phys")
    print(f"{'stride':>6} {'bytes':>6} {'median_ms':>11} {'iqr_ms':>9} {'Mops/s':>9}")
    for r in rows:
        print(f"{r['stride']:>6} {r['sep_bytes']:>6} {r['median_ms']:>11.2f} "
              f"{r['iqr_ms']:>9.2f} {r['mops']:>9.3f}")


def print_table_1b(stride, rows):
    print(f"\nTable 1B — process scaling, stride = {stride} ({stride * WORD_BYTES} B)")
    print(f"{'p':>3} {'median_ms':>11} {'iqr_ms':>9} {'S_p':>7} {'E_p':>7} {'e(p)':>8}")
    for r in rows:
        kf = "-" if r["karp_flatt"] is None else f"{r['karp_flatt']:.4f}"
        print(f"{r['p']:>3} {r['median_ms']:>11.2f} {r['iqr_ms']:>9.2f} "
              f"{r['speedup']:>7.3f} {r['efficiency']:>7.3f} {kf:>8}")


def main() -> None:
    args = build_parser(__doc__).parse_args()
    sid = resolve_student_id(args.id)
    hw = hardware_info()
    p_phys, p_log = hw["physical_cores"], hw["logical_threads"]

    m_total = M_TOTAL // 100 if args.quick else M_TOTAL
    warmup, timed = (1, 3) if args.quick else (WARMUP, TIMED)
    strides = [1, 16, 32] if args.quick else STRIDES
    p_values = [1, 2, p_phys] if args.quick else doubling_up_to(p_log)

    print(f"=== PHASE 1: false sharing  M={m_total}  P_phys={p_phys}  P_log={p_log} "
          f"warmup={warmup} timed={timed} ===")
    shm = shared_memory.SharedMemory(create=True, size=SHM_SIZE)
    try:
        stride_rows = []
        for stride in strides:
            row = bench(shm, p_phys, stride, m_total, warmup, timed)
            stride_rows.append(row)
            print(f"[1A] p={p_phys:2d} stride={stride:2d} ({row['sep_bytes']:3d} B): "
                  f"med={row['median_ms']:.2f} ms iqr={row['iqr_ms']:.2f} "
                  f"{row['mops']:.3f} Mops/s slot0={row['slot0_final']}")
        print_table_1a(stride_rows)

        scaling = {}
        for stride in (1, 16):
            rows = []
            for p in p_values:
                row = bench(shm, p, stride, m_total, warmup, timed)
                rows.append(row)
                print(f"[1B] stride={stride:2d} p={p:2d}: med={row['median_ms']:.2f} ms "
                      f"iqr={row['iqr_ms']:.2f}")
            scaling[str(stride)] = add_scaling_metrics(rows)
            print_table_1b(stride, rows)
    finally:
        shm.close()
        shm.unlink()

    by_stride = {r["stride"]: r for r in stride_rows}
    med_s1 = by_stride[1]["median_ms"]
    med_s16 = by_stride[16]["median_ms"]
    tag = emit_verify_tag("PHASE1", sid, med_s1, med_s16)
    print("\n" + tag)
    save_json(results_dir(args.quick) / "phase1.json", {
        "student_id": sid, "p_phys": p_phys, "p_log": p_log, "m_total": m_total,
        "warmup_runs": warmup, "timed_runs": timed,
        "stride_sweep": stride_rows, "scaling": scaling,
        "verify": tag, "hash_inputs": {"primary": med_s1, "secondary": med_s16},
    })


if __name__ == "__main__":
    main()
