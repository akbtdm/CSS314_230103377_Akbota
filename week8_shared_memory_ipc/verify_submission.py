"""Section 6: automated self-verification & checksum harness (RECONSTRUCTED).

The handout's harness listing is cut off after its first line
("import os, sys, struct, hashlib from multiprocessing import shared_memory
def run_self_audit(") in both the PDF and its text extraction, so the original body
is not available. This file implements run_self_audit() with the same imports and the
checks the handout names (structural alignment, checksum compliance, personal
signature). The AUTH token it prints is therefore produced by THIS reconstruction,
not by the instructor's (unknown) algorithm.

Token = "AUTH-" + first 12 hex digits (upper case) of
        sha256("UID:OFFSET_PAD:RING_CAPACITY:MAGIC_HEX:TARGET_SEED|" + sha256 of each deliverable .py)

Usage: python3 verify_submission.py
"""
import hashlib
import os
import struct
import sys
from multiprocessing import shared_memory

import lab_constants as K
import ring_buffer
import slab_allocator

HERE = os.path.dirname(os.path.abspath(__file__))
DELIVERABLES = ["slab_allocator.py", "ring_buffer.py", "race_inducer.py",
                "cache_collision.py", "matrix_sifter.py"]
PSHMNAMLEN = 31  # macOS limit incl. the leading '/'


def check(results, name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")


def run_self_audit() -> str:
    r = []
    uid = K.UID
    check(r, "OFFSET_PAD", K.OFFSET_PAD == (uid % 7) * 8, f"{K.OFFSET_PAD}")
    check(r, "RING_CAPACITY", K.RING_CAPACITY == 8 + (uid % 5) * 4, f"{K.RING_CAPACITY}")
    check(r, "MAGIC_BYTE", K.MAGIC_BYTE == bytes([0xA0 + uid % 16]), f"0x{K.MAGIC_BYTE.hex()}")
    check(r, "TARGET_SEED", K.TARGET_SEED == uid ^ 0x5A5A5A5A, f"{K.TARGET_SEED}")

    # Structural alignment of the Milestone 1 record (offsets from the handout table)
    fmt = slab_allocator.HEADER_FMT
    offs = [struct.calcsize(fmt[:i]) for i in (1, 2, 3, 4)]  # '>' , '>B', '>BI', '>BIQ'
    check(r, "slab field offsets 0/1/5/13/45", offs[1:] == [1, 5, 13] and slab_allocator.HEADER_LEN == 45
          and slab_allocator.RECORD_LEN == 47, f"computed {offs[1:]} + crc@{slab_allocator.HEADER_LEN}")
    check(r, "slab fits in 512 B", K.OFFSET_PAD + slab_allocator.RECORD_LEN <= slab_allocator.SLAB_SIZE)

    # Checksum compliance on a real shared memory slab (created, verified, unlinked)
    name = f"shm_audit_{uid}"
    shm = shared_memory.SharedMemory(name=name, create=True, size=slab_allocator.SLAB_SIZE)
    try:
        buf = shm.buf
        crc = slab_allocator.write_record(buf, K.OFFSET_PAD)
        status, _ = slab_allocator.validate_record(buf, K.OFFSET_PAD)
        check(r, "XOR-16 checksum round trip", status == slab_allocator.EXIT_OK, f"0x{crc:04X}")
        buf[K.OFFSET_PAD + 30] ^= 0x01
        status, _ = slab_allocator.validate_record(buf, K.OFFSET_PAD)
        check(r, "single-bit corruption detected", status == slab_allocator.EXIT_BAD_CRC)
        del buf
    finally:
        shm.close()
        shm.unlink()

    # Ring layout
    check(r, "ring size = 12 + 32*C", ring_buffer.RING_BYTES == 12 + 32 * K.RING_CAPACITY,
          f"{ring_buffer.RING_BYTES} B")
    check(r, "ring slot = 32 B", struct.calcsize(ring_buffer.SLOT_FMT) == 32)
    check(r, "ring name", K.RING_NAME == f"shm_ring_{uid}", K.RING_NAME)

    # Segment names fit macOS PSHMNAMLEN, and no segment was leaked by earlier runs
    for seg in (K.SLAB_NAME, K.RING_NAME, K.SIFT_NAME, K.SIFT_RESULT_NAME, name):
        check(r, f"name length '{seg}'", len(seg) + 1 <= PSHMNAMLEN, f"{len(seg) + 1} chars")
        try:
            leaked = shared_memory.SharedMemory(name=seg, track=False)
            leaked.close()
            check(r, f"no leaked segment '{seg}'", False, "segment still exists")
        except FileNotFoundError:
            check(r, f"no leaked segment '{seg}'", True)

    # Forbidden high-level IPC must not appear in the deliverables
    banned = ("multiprocessing.Queue", "Queue(", "Pipe(", "Manager(", "import pickle", "pickle.")
    digest = hashlib.sha256(f"{uid}:{K.OFFSET_PAD}:{K.RING_CAPACITY}:{K.MAGIC_BYTE.hex()}:{K.TARGET_SEED}|"
                            .encode())
    for f in DELIVERABLES:
        path = os.path.join(HERE, f)
        exists = os.path.isfile(path)
        check(r, f"deliverable {f} present", exists)
        if not exists:
            continue
        src = open(path, "rb").read()
        text = src.decode()
        hits = [b for b in banned if b in text]
        check(r, f"{f} uses no Queue/Pipe/Manager/pickle", not hits, ", ".join(hits))
        digest.update(hashlib.sha256(src).digest())

    token = "AUTH-" + digest.hexdigest()[:12].upper()
    print(f"\n{sum(r)}/{len(r)} checks passed")
    print("NOTE: harness reconstructed (handout listing truncated); token is not the instructor's algorithm.")
    print(f"VERIFICATION TOKEN: {token}")
    if not all(r):
        sys.exit(1)
    return token


if __name__ == "__main__":
    run_self_audit()
