"""Milestone 0: host topology profiling and deterministic student-ID salting.

Usage:
    python3 src/profile_env.py --student-id 230103377

Writes:
    results/host_profile.json   - CPU model, cores, caches, OS and Python details
    results/salted_params.json  - problem dimensions derived from the student ID
"""
import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

from salt import get_student_params

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def sysctl(key: str):
    """Return the value of a macOS sysctl key as a string, or None if unavailable."""
    try:
        proc = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None
    value = proc.stdout.strip()
    if proc.returncode != 0 or not value:
        return None
    return value


def sysctl_int(key: str):
    value = sysctl(key)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def read_proc_cpuinfo():
    """Linux fallback: parse /proc/cpuinfo for model name and cache size."""
    info = {}
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if ":" not in line:
                    continue
                key, _, value = line.partition(":")
                info.setdefault(key.strip(), value.strip())
    except OSError:
        pass
    return info


def collect_cpu():
    system = platform.system()
    cpu = {
        "model": None,
        "microarchitecture": None,
        "isa": platform.machine(),
        "base_frequency_hz": None,
        "physical_cores": os.cpu_count(),
        "logical_threads": os.cpu_count(),
        "smt_enabled": None,
        "performance_cores": None,
        "efficiency_cores": None,
    }
    if system == "Darwin":
        brand = sysctl("machdep.cpu.brand_string")
        cpu["model"] = brand
        if brand and "Apple M" in brand:
            cpu["microarchitecture"] = f"Apple Silicon ({brand}), heterogeneous P/E core clusters"
        cpu["base_frequency_hz"] = sysctl_int("hw.cpufrequency")
        cpu["physical_cores"] = sysctl_int("hw.physicalcpu")
        cpu["logical_threads"] = sysctl_int("hw.logicalcpu")
        cpu["performance_cores"] = sysctl_int("hw.perflevel0.physicalcpu")
        cpu["efficiency_cores"] = sysctl_int("hw.perflevel1.physicalcpu")
    else:
        info = read_proc_cpuinfo()
        cpu["model"] = info.get("model name") or platform.processor() or None
        try:
            import psutil

            cpu["physical_cores"] = psutil.cpu_count(logical=False)
            cpu["logical_threads"] = psutil.cpu_count(logical=True)
            freq = psutil.cpu_freq()
            if freq and freq.current:
                cpu["base_frequency_hz"] = int(freq.current * 1_000_000)
        except ImportError:
            pass
    if cpu["physical_cores"] and cpu["logical_threads"]:
        cpu["smt_enabled"] = cpu["logical_threads"] > cpu["physical_cores"]
    return cpu


def collect_caches():
    caches = {
        "cache_line_bytes": sysctl_int("hw.cachelinesize"),
        "l1d_bytes": sysctl_int("hw.l1dcachesize"),
        "l1i_bytes": sysctl_int("hw.l1icachesize"),
        "l2_bytes": sysctl_int("hw.l2cachesize"),
        "l3_bytes": sysctl_int("hw.l3cachesize"),
        "performance_cluster": {
            "l1d_bytes": sysctl_int("hw.perflevel0.l1dcachesize"),
            "l2_bytes": sysctl_int("hw.perflevel0.l2cachesize"),
            "cores_sharing_l2": sysctl_int("hw.perflevel0.cpusperl2"),
        },
        "efficiency_cluster": {
            "l1d_bytes": sysctl_int("hw.perflevel1.l1dcachesize"),
            "l2_bytes": sysctl_int("hw.perflevel1.l2cachesize"),
            "cores_sharing_l2": sysctl_int("hw.perflevel1.cpusperl2"),
        },
    }
    if caches["cache_line_bytes"] is None:
        caches["cache_line_bytes"] = 64
    return caches


def collect_software():
    try:
        import numpy

        numpy_version = numpy.__version__
    except ImportError:
        numpy_version = None
    return {
        "os_system": platform.system(),
        "os_kernel_release": platform.release(),
        "os_kernel_version": platform.version(),
        "macos_version": platform.mac_ver()[0] or None,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_compiler": platform.python_compiler(),
        "python_executable": sys.executable,
        "numpy_version": numpy_version,
    }


def build_notes(cpu, caches):
    notes = []
    if cpu["base_frequency_hz"] is None:
        notes.append("Nominal CPU frequency is not exposed by sysctl on Apple Silicon.")
    if caches["l3_bytes"] is None:
        notes.append("No L3 reported: Apple Silicon uses a system-level cache (SLC) shared with the GPU instead of a classic L3.")
    if caches["cache_line_bytes"] == 128:
        notes.append("Cache line is 128 bytes on this CPU, so false-sharing padding must be 16 x int64, not 8 x int64.")
    if cpu["performance_cores"] and cpu["efficiency_cores"]:
        notes.append("Heterogeneous cores: scaling beyond the performance-core count will schedule work onto slower efficiency cores.")
    return notes


def main():
    parser = argparse.ArgumentParser(description="Profile host hardware and derive salted lab parameters.")
    parser.add_argument("--student-id", required=True, help="Student ID used to salt the problem dimensions")
    args = parser.parse_args()

    cpu = collect_cpu()
    caches = collect_caches()
    profile = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "cpu": cpu,
        "caches": caches,
        "memory_bytes": sysctl_int("hw.memsize"),
        "software": collect_software(),
        "notes": build_notes(cpu, caches),
    }
    params = get_student_params(args.student_id)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    profile_path = os.path.join(RESULTS_DIR, "host_profile.json")
    params_path = os.path.join(RESULTS_DIR, "salted_params.json")
    with open(profile_path, "w", encoding="utf-8") as fh:
        json.dump(profile, fh, indent=2)
    with open(params_path, "w", encoding="utf-8") as fh:
        json.dump(params, fh, indent=2)

    print(f"CPU: {cpu['model']}  physical={cpu['physical_cores']} logical={cpu['logical_threads']} "
          f"(P={cpu['performance_cores']}, E={cpu['efficiency_cores']}, SMT={cpu['smt_enabled']})")
    print(f"Cache line: {caches['cache_line_bytes']} B  L1d: {caches['l1d_bytes']}  L2: {caches['l2_bytes']}  L3: {caches['l3_bytes']}")
    print(f"Salted params: N={params['N']} seed={params['seed']} buffer_elements={params['buffer_elements']}")
    print(f"Wrote {profile_path}")
    print(f"Wrote {params_path}")


if __name__ == "__main__":
    main()
