"""Shared helpers for the lab tasks: student ID, hardware probe, partitioning, JSON output.
Standard library only."""
import argparse
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent
ID_FILE = LAB_DIR / "student_id.txt"


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--id", type=int, default=None,
                        help="student ID used as student_seed (overrides student_id.txt)")
    parser.add_argument("--quick", action="store_true",
                        help="tiny smoke-test configuration, results go to results_quick/")
    return parser


def resolve_student_id(cli_value):
    if cli_value is not None:
        return cli_value
    if ID_FILE.exists():
        text = ID_FILE.read_text().strip()
        if text.isdigit():
            return int(text)
        sys.exit(f"student_id.txt must contain digits only, got {text!r}")
    sys.exit("Student ID missing: pass --id <digits> or create student_id.txt")


def results_dir(quick: bool) -> Path:
    path = LAB_DIR / ("results_quick" if quick else "results")
    path.mkdir(exist_ok=True)
    return path


def save_json(path, data) -> None:
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)
    print(f"[saved] {path}")


def _sysctl(key: str):
    try:
        proc = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    value = proc.stdout.strip()
    return value or None


def _sysctl_int(key: str):
    value = _sysctl(key)
    return int(value) if value and value.isdigit() else None


def hardware_info() -> dict:
    info = {
        "cpu_count_logical": mp.cpu_count(),
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()} {platform.machine()}",
    }
    if platform.system() == "Darwin":
        info.update({
            "cpu_model": _sysctl("machdep.cpu.brand_string"),
            "physical_cores": _sysctl_int("hw.physicalcpu") or mp.cpu_count(),
            "logical_threads": _sysctl_int("hw.logicalcpu") or mp.cpu_count(),
            "cache_line_bytes": _sysctl_int("hw.cachelinesize"),
            "l1d_bytes": _sysctl_int("hw.l1dcachesize"),
            "l2_bytes": _sysctl_int("hw.l2cachesize"),
            "l3_bytes": _sysctl_int("hw.l3cachesize"),
            "perf_cores": _sysctl_int("hw.perflevel0.physicalcpu"),
            "eff_cores": _sysctl_int("hw.perflevel1.physicalcpu"),
            "mem_bytes": _sysctl_int("hw.memsize"),
            "p_core_l1d_bytes": _sysctl_int("hw.perflevel0.l1dcachesize"),
            "p_core_l2_bytes": _sysctl_int("hw.perflevel0.l2cachesize"),
            "e_core_l1d_bytes": _sysctl_int("hw.perflevel1.l1dcachesize"),
            "e_core_l2_bytes": _sysctl_int("hw.perflevel1.l2cachesize"),
        })
    else:
        logical = os.cpu_count() or 1
        info.update({"cpu_model": platform.processor(), "physical_cores": logical,
                     "logical_threads": logical})
    return info


def partition(n: int, parts: int):
    """Split range(n) into `parts` contiguous (start, end) ranges, sizes differing by at most 1."""
    base, extra = divmod(n, parts)
    bounds = []
    start = 0
    for t in range(parts):
        end = start + base + (1 if t < extra else 0)
        bounds.append((start, end))
        start = end
    return bounds


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    return ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])


def fmt_stats(stats) -> str:
    return f"min={stats.min_ms:9.3f}  med={stats.median_ms:9.3f}  iqr={stats.iqr_ms:8.3f} ms"


def doubling_up_to(limit: int):
    """[1, 2, 4, ..., limit] — powers of two below limit, then limit itself."""
    values = []
    p = 1
    while p < limit:
        values.append(p)
        p *= 2
    values.append(limit)
    return values
