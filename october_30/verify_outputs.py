#!/usr/bin/env python3
"""Milestone 4: automated verification of every practicum artefact.

Checks, in order:
  M0  results/host_profile.json and results/salted_params.json exist and the salted
      parameters re-derive exactly from the student ID.
  M1  The i-k-j matrix product is regenerated from the salted seed and its salted
      SHA-256 digest (values rounded to 6 dp) must equal the recorded digest; it must
      also match NumPy's BLAS result within a relative 1e-12, and the stride
      benchmark must show the cache-line cliff.
  M2  Every false-sharing run must have terminated with exactly P * K increments.
  M3  The parallel stencil grids for every P must carry the same digest as the P = 1
      reference, and the Roofline summary must be present.

On success a salted SHA-256 receipt over all result files is written to
results/verification_receipt.sig.

Usage:
    python3 verify_outputs.py
"""
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

import numpy as np  # noqa: E402

from salt import get_student_params  # noqa: E402

REL_TOLERANCE = 1e-12
RECEIPT_NAME = "verification_receipt.sig"
RESULT_FILES = (
    "host_profile.json", "salted_params.json", "stride_results.tsv", "loop_interchange.json",
    "false_sharing_karp_flatt.tsv", "scaling_metrics.tsv", "roofline_summary.json",
)


class VerificationFailure(Exception):
    pass


def result_path(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        raise VerificationFailure(f"missing results/{name}")
    return path


def load_json(name):
    with open(result_path(name), encoding="utf-8") as fh:
        return json.load(fh)


def read_tsv(name):
    with open(result_path(name), encoding="utf-8") as fh:
        lines = [line for line in fh if not line.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t"))


# ----------------------------------------------------------------------------- M0
def check_milestone_0():
    profile = load_json("host_profile.json")
    params = load_json("salted_params.json")
    expected = get_student_params(str(params["student_id"]))
    if params != expected:
        raise VerificationFailure(f"salted parameters {params} do not match derivation {expected}")
    cpu, caches = profile.get("cpu", {}), profile.get("caches", {})
    for key, value in (("cpu.model", cpu.get("model")), ("cpu.physical_cores", cpu.get("physical_cores")),
                       ("caches.cache_line_bytes", caches.get("cache_line_bytes")),
                       ("software", profile.get("software"))):
        if not value:
            raise VerificationFailure(f"host_profile.json is missing {key}")
    return (f"Host topology ({cpu['model']}, {cpu['physical_cores']} cores, {caches['cache_line_bytes']} B line) "
            f"and salted parameters (N={params['N']}, seed={params['seed']}) validated."), params


# ----------------------------------------------------------------------------- M1
def check_milestone_1(params):
    from dense_kernel import matmul_ikj, run_kernel, salted_digest

    record = load_json("loop_interchange.json")
    primary = record["blocks"][0]
    checks = primary["correctness"]
    if not checks["within_tolerance"] or not checks["naive_vs_interchanged_bitwise_equal"]:
        raise VerificationFailure("recorded loop-interchange correctness flags are not all true")

    N, M, seed, student_id = int(params["N"]), int(primary["M"]), int(params["seed"]), str(params["student_id"])
    rng = np.random.default_rng(seed)
    A = np.ascontiguousarray(rng.random((N, N))[:M, :M])
    B = np.ascontiguousarray(rng.random((N, N))[:M, :M])
    C = run_kernel(matmul_ikj, A, B)
    digest = salted_digest(student_id, C)
    if digest != primary["salted_sha256_rounded_6dp"]:
        raise VerificationFailure(f"regenerated digest {digest[:12]}... != recorded {primary['salted_sha256_rounded_6dp'][:12]}...")
    reference = A @ B
    rel = float(np.max(np.abs(C - reference) / np.maximum(np.abs(reference), 1e-300)))
    if rel > REL_TOLERANCE:
        raise VerificationFailure(f"interchanged kernel deviates from BLAS by {rel:.2e} > {REL_TOLERANCE}")

    rows = read_tsv("stride_results.tsv")
    if len(rows) < 8:
        raise VerificationFailure("stride_results.tsv has fewer than 8 stride rows")
    ns = {int(r["stride"]): float(r["ns_per_element"]) for r in rows}
    cliff = ns[max(ns)] / ns[1]
    if cliff < 2.0:
        raise VerificationFailure(f"no cache-line cliff: stride-{max(ns)} latency only {cliff:.2f}x stride-1")
    return (f"Matrix numerical digest matches reference: {digest[:8]}... (M={M}, rel. error {rel:.1e}, "
            f"speedup {record['speedup']:.2f}x); stride cliff {cliff:.1f}x.")


# ----------------------------------------------------------------------------- M2
def check_milestone_2():
    rows = read_tsv("false_sharing_karp_flatt.tsv")
    if not rows:
        raise VerificationFailure("false_sharing_karp_flatt.tsv is empty")
    for r in rows:
        total, expected = int(r["total_count"]), int(r["expected_count"])
        if r["count_ok"] != "True" or total != expected or expected != int(r["P"]) * int(r["K"]):
            raise VerificationFailure(f"{r['backend']}/{r['variant']} P={r['P']}: counted {total}, expected {expected}")
    largest = max(rows, key=lambda r: int(r["expected_count"]))
    backend = "numba" if any(r["backend"] == "numba" for r in rows) else rows[0]["backend"]
    by_key = {(r["variant"], int(r["P"])): float(r["ns_per_increment"]) for r in rows if r["backend"] == backend}
    slow_unpadded = by_key[("unpadded", 4)] / by_key[("unpadded", 1)]
    slow_padded = by_key[("padded_line", 4)] / by_key[("padded_line", 1)]
    warning = None
    if slow_unpadded < 1.3:
        warning = f"unpadded slowdown at P=4 is only {slow_unpadded:.2f}x; contention collapse is weak"
    return (f"Reduction sum exact match: {largest['total_count']} == {largest['expected_count']} "
            f"(all {len(rows)} runs exact); unpadded {slow_unpadded:.2f}x slower at P=4 vs padded {slow_padded:.2f}x."), warning


# ----------------------------------------------------------------------------- M3
def check_milestone_3():
    rows = read_tsv("scaling_metrics.tsv")
    strong = [r for r in rows if r["mode"] == "strong"]
    weak = [r for r in rows if r["mode"] == "weak"]
    if len(strong) < 2 or len(weak) < 2:
        raise VerificationFailure("scaling_metrics.tsv lacks strong or weak scaling rows")
    checksums = {r["checksum"] for r in strong}
    if len(checksums) != 1 or any(r["digest_matches_P1"] != "True" for r in strong):
        raise VerificationFailure(f"strong-scaling checksums differ across P: {sorted(checksums)}")
    roof = load_json("roofline_summary.json")
    if not roof.get("all_strong_digests_match"):
        raise VerificationFailure("roofline_summary.json reports mismatching stencil digests")
    regime = roof.get("roofline_position", {}).get("regime")
    if regime not in ("memory-bandwidth-bound", "compute-bound"):
        raise VerificationFailure("roofline_summary.json has no regime classification")
    s_fit = roof["strong_scaling"]["amdahl_serial_fraction_fit"]
    best = max(float(r["speedup_S_P"]) for r in strong)
    return (f"Stencil numerical tolerance verified: checksum {checksums.pop()} identical for P in "
            f"{[int(r['P']) for r in strong]}; best speedup {best:.2f}x, Amdahl s={s_fit:.3f}, {regime}.")


# ----------------------------------------------------------------------------- receipt
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def emit_receipt(student_id, messages):
    files = {name: sha256_file(result_path(name)) for name in RESULT_FILES}
    combined = hashlib.sha256()
    combined.update(student_id.encode("utf-8"))
    for name in RESULT_FILES:
        combined.update(f"{name}:{files[name]}\n".encode("utf-8"))
    receipt = {
        "student_id": student_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "verifier": "verify_outputs.py",
        "status": "PASS",
        "checks": messages,
        "file_sha256": files,
        "receipt_sha256": combined.hexdigest(),
    }
    out = os.path.join(RESULTS_DIR, RECEIPT_NAME)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(receipt, fh, indent=2)
    return out, receipt["receipt_sha256"]


def main():
    messages, warnings = {}, []
    try:
        text, params = check_milestone_0()
        print(f"[PASS] Milestone 0: {text}")
        messages["M0"] = text

        text = check_milestone_1(params)
        print(f"[PASS] Milestone 1: {text}")
        messages["M1"] = text

        text, warning = check_milestone_2()
        print(f"[PASS] Milestone 2: {text}")
        messages["M2"] = text
        if warning:
            warnings.append(warning)

        text = check_milestone_3()
        print(f"[PASS] Milestone 3: {text}")
        messages["M3"] = text
    except VerificationFailure as exc:
        print(f"[FAIL] {exc}")
        return 1
    except Exception as exc:  # unexpected error: report and fail
        print(f"[FAIL] unexpected error during verification: {type(exc).__name__}: {exc}")
        return 1

    for warning in warnings:
        print(f"[WARN] {warning}")
    out, digest = emit_receipt(str(params["student_id"]), messages)
    print(f"All empirical datasets verified successfully. Emitting {RECEIPT_NAME} ({digest[:16]}...).")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
