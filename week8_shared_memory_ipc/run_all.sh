#!/usr/bin/env bash
# Runs every milestone once, sequentially, tee-ing console output into logs/,
# then builds results/report.txt and the submission ZIP shm_lab_230103377.zip.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
mkdir -p logs results
run() { local log=$1; shift; echo "### $(date '+%F %T') \$ $PY $*" | tee "logs/$log"; "$PY" "$@" 2>&1 | tee -a "logs/$log"; }

{
  echo "date: $(date)"; echo "cpu: $(sysctl -n machdep.cpu.brand_string)"
  echo "cores: $(sysctl -n hw.physicalcpu) physical ($(sysctl -n hw.perflevel0.physicalcpu) P + $(sysctl -n hw.perflevel1.physicalcpu) E), logical $(sysctl -n hw.logicalcpu)"
  echo "hw.cachelinesize: $(sysctl -n hw.cachelinesize) B; L1d P/E: $(sysctl -n hw.perflevel0.l1dcachesize)/$(sysctl -n hw.perflevel1.l1dcachesize) B; L2 P/E: $(sysctl -n hw.perflevel0.l2cachesize)/$(sysctl -n hw.perflevel1.l2cachesize) B"
  echo "memory: $(sysctl -n hw.memsize) B; pagesize: $(sysctl -n hw.pagesize) B"
  echo "os: $(sw_vers -productName) $(sw_vers -productVersion) ($(uname -r))"
  echo "python: $($PY -c 'import sys,numpy,numba;print(sys.version.split()[0],"numpy",numpy.__version__,"numba",numba.__version__)')"
} | tee logs/hw_info.log

run m1_slab_allocator.log slab_allocator.py
run m2_ring_buffer.log ring_buffer.py --trials 3
run m2_ring_buffer_negative.log ring_buffer.py --trials 1 --inject-gap || true
run m3_race_inducer.log race_inducer.py --workers 1 2 4 8 16 --reps 3 --lock-reps 1
run m3_cache_collision.log cache_collision.py --reps 3
run m4_matrix_sifter.log matrix_sifter.py --compare-copy
run verify_submission.log verify_submission.py

echo "leftover python multiprocessing processes: $(pgrep -f 'multiprocessing' | wc -l | tr -d ' ')" | tee logs/cleanup_check.log

{
  echo "Shared Memory & Concurrent IPC lab -- report.txt"
  echo "Student: Akbota Kutushova, UID 230103377"
  echo
  cat logs/hw_info.log
  for f in m1_slab_allocator m2_ring_buffer m2_ring_buffer_negative m3_race_inducer m3_cache_collision m4_matrix_sifter verify_submission cleanup_check; do
    echo; echo "================ $f ================"; cat "logs/$f.log"
  done
  echo; grep -h "VERIFICATION TOKEN" logs/verify_submission.log
} > results/report.txt

rm -f results/shm_lab_230103377.zip
zip -j -q results/shm_lab_230103377.zip slab_allocator.py ring_buffer.py race_inducer.py cache_collision.py \
  matrix_sifter.py verify_submission.py lab_constants.py results/report.txt
unzip -l results/shm_lab_230103377.zip
