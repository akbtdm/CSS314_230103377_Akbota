#!/bin/sh
# Task 1: host silicon audit (macOS). Output is tee'd into logs/task1_hw_audit.txt
cd "$(dirname "$0")"
{
  echo "=== sw_vers / uname ==="; sw_vers; uname -a
  echo; echo "=== sysctl -a | grep machdep.cpu ==="; sysctl -a | grep machdep.cpu
  echo; echo "=== core topology and caches (hw.*) ==="
  sysctl hw.physicalcpu hw.logicalcpu hw.ncpu hw.cachelinesize hw.memsize hw.nperflevels
  sysctl -a | grep -E '^hw\.perflevel[01]\.'
  echo; echo "=== clock frequency keys (Apple Silicon does not expose them) ==="
  sysctl hw.cpufrequency hw.cpufrequency_max 2>&1
  echo; echo "=== SIMD / vector extensions ==="
  sysctl -a | grep -E '^hw\.optional\.(neon|AdvSIMD|arm\.FEAT_(SME|SME2|SVE|BF16|FP16|I8MM|DotProd))'
  echo; echo "=== empirical clock probe (dependent add chain) ==="
  clang -O2 -o clock_probe clock_probe.c && ./clock_probe performance && ./clock_probe efficiency
} 2>&1 | tee logs/task1_hw_audit.txt
