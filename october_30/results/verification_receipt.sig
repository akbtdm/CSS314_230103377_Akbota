{
  "student_id": "230103377",
  "generated_at_utc": "2026-10-30T06:05:36.470692+00:00",
  "verifier": "verify_outputs.py",
  "status": "PASS",
  "checks": {
    "M0": "Host topology (Apple M4, 10 cores, 128 B line) and salted parameters (N=1280, seed=30385) validated.",
    "M1": "Matrix numerical digest matches reference: cbbce5e6... (M=640, rel. error 3.9e-15, speedup 4.84x); stride cliff 22.8x.",
    "M2": "Reduction sum exact match: 40000000 == 40000000 (all 32 runs exact); unpadded 2.00x slower at P=4 vs padded 1.09x.",
    "M3": "Stencil numerical tolerance verified: checksum 819441.775585 identical for P in [1, 2, 4, 8, 10]; best speedup 2.08x, Amdahl s=0.432, memory-bandwidth-bound."
  },
  "file_sha256": {
    "host_profile.json": "13b937a86b525c9213a919d135e3a5cdf60f641f402f3f03c6c0487c5fcc1869",
    "salted_params.json": "6dde643890a9eaaa60e4917b016786bc61e1569c9f4fc9facbd3cb7f08d7f499",
    "stride_results.tsv": "3f86731f8319bc62008f90bd5565f7d970b8f90c5c379b6b18fdc3fd785bedd6",
    "loop_interchange.json": "a0af03c2d15dd8bbdb7462a3936647ad1eabbf135eb7db203803036dd91a68b0",
    "false_sharing_karp_flatt.tsv": "ca6b5b5ee161c02b4ca5a8b8cbd8f0f47b2d8bdcdb6f4a3a3531a8d13f11d951",
    "scaling_metrics.tsv": "000c3bca4eb05e0b4a25177e1b855ae17dbb8e407baad7927e8243dd14179b1c",
    "roofline_summary.json": "0531409f301dd9dd7da8530f31db7054a7cf8864b749a4fe7e77776b4d748695"
  },
  "receipt_sha256": "3ac6d298c4203da684cd28ecccaefd3f346ae954f280973502d9250bd437e614"
}