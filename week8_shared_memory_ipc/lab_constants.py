"""Per-student runtime constants (handout section 1), derived from the Student ID.

Student: Akbota Kutushova, UID 230103377.
"""

UID = 230103377

OFFSET_PAD = (UID % 7) * 8                # 230103377 % 7 = 0  -> 0 bytes
RING_CAPACITY = 8 + (UID % 5) * 4         # 230103377 % 5 = 2  -> 16 slots
MAGIC_BYTE = bytes([0xA0 + (UID % 16)])   # 230103377 % 16 = 1 -> b'\xa1'
TARGET_SEED = UID ^ 0x5A5A5A5A            # -> 1475166987 (0x57ED430B)

# macOS limits POSIX shm names to 31 chars (PSHMNAMLEN) including the leading '/'.
SLAB_NAME = f"shm_slab_{UID}"
RING_NAME = f"shm_ring_{UID}"
SIFT_NAME = f"shm_sift_{UID}"
SIFT_RESULT_NAME = f"shm_sres_{UID}"


def describe() -> str:
    return (f"UID={UID} OFFSET_PAD={OFFSET_PAD} RING_CAPACITY={RING_CAPACITY} "
            f"MAGIC_BYTE=0x{MAGIC_BYTE.hex()} TARGET_SEED={TARGET_SEED} (0x{TARGET_SEED:08X})")
