"""Milestone 1: raw 512-byte shared-memory slab with struct serialization.

The master maps a POSIX shared memory slab, writes one telemetry record in place
through a zero-copy memoryview with struct.pack_into, and spawns a worker that
attaches by name, validates the magic byte and the 16-bit XOR checksum, and
prints the decoded record. No Queue/Pipe/Manager/pickle is used for data: the
worker receives nothing but the (deterministic) segment name it derives itself.

Usage: python3 slab_allocator.py
"""
import os
import struct
import sys
import time
from multiprocessing import Process, shared_memory

from lab_constants import MAGIC_BYTE, OFFSET_PAD, SLAB_NAME, describe

SLAB_SIZE = 512
# Big-endian, no padding: magic(1) pid(4) ts(8) 4 x double(32) -> 45 bytes, then crc(2)
HEADER_FMT = ">BIQ4d"
CRC_FMT = ">H"
HEADER_LEN = struct.calcsize(HEADER_FMT)            # 45
RECORD_LEN = HEADER_LEN + struct.calcsize(CRC_FMT)  # 47
FIELD_OFFSETS = {"magic_header": 0, "writer_pid": 1, "timestamp_ns": 5,
                 "payload_floats": 13, "checksum_crc": 45}
PAYLOAD = (3.141592653589793, 2.718281828459045, 1.4142135623730951, 230103377 / 1e6)

EXIT_OK, EXIT_BAD_MAGIC, EXIT_BAD_CRC = 0, 2, 3


def xor16(mv: memoryview) -> int:
    """XOR of big-endian 16-bit words over mv (odd trailing byte padded with 0x00).

    The handout asks for a uint16 'XOR sum of all preceding bytes'; a plain byte XOR
    can only produce 8 bits, so the bytes are folded as 16-bit words instead.
    Iterates the memoryview directly, so no bytes copy is made.
    """
    acc = 0
    n = len(mv)
    for i in range(0, n - 1, 2):
        acc ^= (mv[i] << 8) | mv[i + 1]
    if n % 2:
        acc ^= mv[n - 1] << 8
    return acc


def write_record(buf: memoryview, base: int, payload=PAYLOAD) -> int:
    """Pack one record in place at `base`; returns the checksum written."""
    struct.pack_into(HEADER_FMT, buf, base, MAGIC_BYTE[0], os.getpid(), time.time_ns(), *payload)
    crc = xor16(buf[base:base + HEADER_LEN])  # memoryview slice = view, not a copy
    struct.pack_into(CRC_FMT, buf, base + HEADER_LEN, crc)
    return crc


def validate_record(buf: memoryview, base: int):
    """Return (status, decoded_dict). status is EXIT_OK / EXIT_BAD_MAGIC / EXIT_BAD_CRC."""
    magic, pid, ts, *floats = struct.unpack_from(HEADER_FMT, buf, base)
    (stored_crc,) = struct.unpack_from(CRC_FMT, buf, base + HEADER_LEN)
    calc_crc = xor16(buf[base:base + HEADER_LEN])
    rec = {"magic": magic, "pid": pid, "ts": ts, "floats": floats,
           "stored_crc": stored_crc, "calc_crc": calc_crc}
    if magic != MAGIC_BYTE[0]:
        return EXIT_BAD_MAGIC, rec
    if stored_crc != calc_crc:
        return EXIT_BAD_CRC, rec
    return EXIT_OK, rec


def worker() -> None:
    shm = shared_memory.SharedMemory(name=SLAB_NAME, track=False)
    try:
        buf = shm.buf
        status, rec = validate_record(buf, OFFSET_PAD)
        age_us = (time.time_ns() - rec["ts"]) / 1e3
        print(f"[worker pid={os.getpid()}] attached '{SLAB_NAME}' mapped={shm.size} B")
        print(f"[worker] magic_header = 0x{rec['magic']:02X} (expected 0x{MAGIC_BYTE.hex().upper()}) "
              f"-> {'OK' if rec['magic'] == MAGIC_BYTE[0] else 'MISMATCH'}")
        print(f"[worker] checksum_crc = 0x{rec['stored_crc']:04X}, recomputed 0x{rec['calc_crc']:04X} "
              f"-> {'OK' if rec['stored_crc'] == rec['calc_crc'] else 'MISMATCH'}")
        if status == EXIT_OK:
            print(f"[worker] writer_pid   = {rec['pid']}")
            print(f"[worker] timestamp_ns = {rec['ts']} (record age when read: {age_us:.1f} us)")
            print(f"[worker] payload      = {', '.join(f'{x:.12g}' for x in rec['floats'])}")
        else:
            print(f"[worker] REJECTED record (status {status})")
        del buf
    finally:
        shm.close()
    sys.exit(status)


def run_worker() -> int:
    p = Process(target=worker)
    p.start()
    p.join()
    return p.exitcode


def main() -> int:
    print(describe())
    print(f"record layout (big-endian, base = OFFSET_PAD = {OFFSET_PAD}): "
          + ", ".join(f"{k}@{v + OFFSET_PAD}" for k, v in FIELD_OFFSETS.items())
          + f"; record length {RECORD_LEN} B; struct header calcsize {HEADER_LEN} B")
    shm = shared_memory.SharedMemory(name=SLAB_NAME, create=True, size=SLAB_SIZE)
    failures = 0
    try:
        buf = shm.buf  # zero-copy memoryview over the mmap
        print(f"[master pid={os.getpid()}] created '{SLAB_NAME}': requested {SLAB_SIZE} B, "
              f"kernel mapping {shm.size} B (macOS rounds shm to the 16 KiB page)")
        struct.pack_into(f"{SLAB_SIZE}x", buf, 0)  # zero the logical slab in place
        crc = write_record(buf, OFFSET_PAD)
        print(f"[master] wrote record in place, checksum 0x{crc:04X}; raw bytes "
              f"{buf[OFFSET_PAD:OFFSET_PAD + RECORD_LEN].hex()}")

        print("\n--- run 1: valid record ---")
        code = run_worker()
        print(f"[master] worker exit code {code} (expected {EXIT_OK})")
        failures += code != EXIT_OK

        print("\n--- run 2: negative test, one payload byte flipped in place ---")
        buf[OFFSET_PAD + 20] ^= 0xFF
        code = run_worker()
        print(f"[master] worker exit code {code} (expected {EXIT_BAD_CRC})")
        failures += code != EXIT_BAD_CRC
        buf[OFFSET_PAD + 20] ^= 0xFF

        print("\n--- run 3: negative test, wrong magic byte ---")
        buf[OFFSET_PAD] = 0x00
        code = run_worker()
        print(f"[master] worker exit code {code} (expected {EXIT_BAD_MAGIC})")
        failures += code != EXIT_BAD_MAGIC
        del buf
    finally:
        shm.close()
        shm.unlink()
        print(f"\n[master] closed and unlinked '{SLAB_NAME}'")
    print("RESULT:", "PASS" if failures == 0 else f"FAIL ({failures} checks)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
