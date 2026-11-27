"""Milestone 2: lockless SPSC ring buffer inside one POSIX shared memory segment.

Layout of segment f"shm_ring_{UID}" (C = RING_CAPACITY = 16):
  0x00 uint32 head_index     consumer read cursor   (written only by the consumer)
  0x04 uint32 tail_index     producer write cursor  (written only by the producer)
  0x08 uint32 dropped_count  producer full-buffer stalls (written only by the producer)
  0x0C C slots x 32 B        [u64 sequence_id | u64 timestamp_ns | 16 B raw data]

The three cursors are accessed through memoryview.cast('I'): each index read/write
is one aligned native 32-bit load/store (single-copy atomic on arm64/x86-64), and
each cursor has exactly one writer, so no lock or mutex is needed. The producer
fills the slot first and only then stores tail_index (publish order).

Usage: python3 ring_buffer.py [--trials 3] [--records 100000] [--inject-gap]
"""
import argparse
import os
import struct
import sys
import time
from multiprocessing import Process, shared_memory

from lab_constants import RING_CAPACITY, RING_NAME, TARGET_SEED, describe

HEADER_BYTES = 12
SLOT_BYTES = 32
SLOT_FMT = "<QQQQ"  # seq, timestamp_ns, data0, data1 (data0/data1 = the 16 raw bytes)
RING_BYTES = HEADER_BYTES + RING_CAPACITY * SLOT_BYTES
HEAD, TAIL, DROPPED = 0, 1, 2
FULL_BACKOFF_S = 0.0001
GOLDEN = 0x9E3779B97F4A7C15
MASK64 = (1 << 64) - 1
assert struct.calcsize(SLOT_FMT) == SLOT_BYTES


def payload_for(seq: int):
    return seq ^ TARGET_SEED, (seq * GOLDEN) & MASK64


def producer(n_records: int, inject_gap: bool) -> None:
    shm = shared_memory.SharedMemory(name=RING_NAME, track=False)
    buf = shm.buf
    idx = buf[:HEADER_BYTES].cast("I")
    pack_into, now, cap = struct.pack_into, time.monotonic_ns, RING_CAPACITY
    tail = idx[TAIL]
    stalled_records = 0
    t0 = now()
    for seq in range(n_records):
        if inject_gap and seq == n_records // 2:
            seq += 1  # deliberately corrupt the stream to prove the consumer's assertion fires
        nxt = (tail + 1) % cap
        if nxt == idx[HEAD]:  # full: (tail + 1) mod C == head
            stalled_records += 1
            while nxt == idx[HEAD]:
                idx[DROPPED] += 1
                time.sleep(FULL_BACKOFF_S)
        d0, d1 = payload_for(seq)
        pack_into(SLOT_FMT, buf, HEADER_BYTES + tail * SLOT_BYTES, seq, now(), d0, d1)  # 1) payload
        idx[TAIL] = nxt                                                                    # 2) publish
        tail = nxt
    t1 = now()
    print(f"  [producer pid={os.getpid()}] sent {n_records} records in {(t1 - t0) / 1e6:.1f} ms; "
          f"records that hit a full ring: {stalled_records}; backoff sleeps: {idx[DROPPED]}", flush=True)
    del idx, buf
    shm.close()


def consumer(n_records: int) -> None:
    shm = shared_memory.SharedMemory(name=RING_NAME, track=False)
    buf = shm.buf
    idx = buf[:HEADER_BYTES].cast("I")
    unpack_from, now, cap, yield_ = struct.unpack_from, time.monotonic_ns, RING_CAPACITY, os.sched_yield
    head = idx[HEAD]
    lat = [0] * n_records
    spins = 0
    t_first = None
    for expected in range(n_records):
        while head == idx[TAIL]:  # empty: head == tail
            spins += 1
            yield_()
        seq, ts, d0, d1 = unpack_from(SLOT_FMT, buf, HEADER_BYTES + head * SLOT_BYTES)
        if seq != expected:
            del idx, buf
            shm.close()
        assert seq == expected, (f"MEMORY CORRUPTION: expected sequence_id {expected}, read {seq} "
                                 f"from slot {head}")
        assert (d0, d1) == payload_for(seq), f"MEMORY CORRUPTION: payload mismatch at seq {seq}"
        t = now()
        if t_first is None:
            t_first = ts  # producer's timestamp of record 0 = start of the transfer
        lat[expected] = t - ts
        head = (head + 1) % cap
        idx[HEAD] = head  # release the slot
    t_end = now()
    wall_s = (t_end - t_first) / 1e9
    lat.sort()
    print(f"  [consumer pid={os.getpid()}] validated {n_records} strictly sequential records", flush=True)
    print(f"RESULT wall_ms={wall_s * 1e3:.2f} throughput_rps={n_records / wall_s:.0f} "
          f"lat_p50_us={lat[n_records // 2] / 1e3:.2f} lat_p99_us={lat[int(n_records * 0.99)] / 1e3:.2f} "
          f"empty_spins={spins}", flush=True)
    del idx, buf
    shm.close()


def run_trial(n_records: int, inject_gap: bool):
    shm = shared_memory.SharedMemory(name=RING_NAME, create=True, size=RING_BYTES)
    try:
        struct.pack_into(f"{RING_BYTES}x", shm.buf, 0)
        cons = Process(target=consumer, args=(n_records,))
        prod = Process(target=producer, args=(n_records, inject_gap))
        t0 = time.perf_counter()
        cons.start()
        prod.start()
        while prod.is_alive():  # if the consumer aborts, the ring stays full: stop the producer
            prod.join(timeout=0.05)
            if cons.exitcode not in (None, 0):
                prod.terminate()
                prod.join()
                print("  [master] consumer aborted -> producer terminated")
        cons.join()
        total_ms = (time.perf_counter() - t0) * 1e3
        idx = shm.buf[:HEADER_BYTES].cast("I")
        head, tail, dropped = idx[HEAD], idx[TAIL], idx[DROPPED]
        del idx
        print(f"  [master] end state head={head} tail={tail} dropped_count={dropped}; "
              f"consumer exit={cons.exitcode} producer exit={prod.exitcode}; "
              f"total incl. process spawn {total_ms:.0f} ms")
        return cons.exitcode, dropped
    finally:
        shm.close()
        shm.unlink()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--records", type=int, default=100_000)
    ap.add_argument("--inject-gap", action="store_true", help="negative test: skip one sequence id")
    a = ap.parse_args()
    print(describe())
    print(f"segment '{RING_NAME}': {RING_BYTES} B logical = 12 B header + {RING_CAPACITY} x {SLOT_BYTES} B "
          f"slots (usable capacity {RING_CAPACITY - 1}); full-buffer backoff time.sleep({FULL_BACKOFF_S})")
    codes = []
    for t in range(1, a.trials + 1):
        print(f"--- trial {t}/{a.trials}: {a.records} records{' (gap injected)' if a.inject_gap else ''} ---",
              flush=True)
        code, _ = run_trial(a.records, a.inject_gap)
        codes.append(code)
    if a.inject_gap:
        ok = all(c != 0 for c in codes)
        print("RESULT negative test:", "PASS (consumer aborted on out-of-order id)" if ok else "FAIL")
    else:
        ok = all(c == 0 for c in codes)
        print("RESULT all trials:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
