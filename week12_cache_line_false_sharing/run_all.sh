#!/usr/bin/env bash
# Runs the whole lab once (benchmarks run one at a time inside one process),
# tee-ing console output into logs/. Produces cache_lab_manifest.json and results/.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
mkdir -p logs results
{
  echo "date: $(date)"; echo "cpu: $(sysctl -n machdep.cpu.brand_string)"
  echo "cores: $(sysctl -n hw.physicalcpu) physical ($(sysctl -n hw.perflevel0.physicalcpu) P + $(sysctl -n hw.perflevel1.physicalcpu) E), logical $(sysctl -n hw.logicalcpu)"
  echo "hw.cachelinesize: $(sysctl -n hw.cachelinesize) B; L1d P/E: $(sysctl -n hw.perflevel0.l1dcachesize)/$(sysctl -n hw.perflevel1.l1dcachesize) B; L2 P/E: $(sysctl -n hw.perflevel0.l2cachesize)/$(sysctl -n hw.perflevel1.l2cachesize) B"
  echo "memory: $(sysctl -n hw.memsize) B; pagesize: $(sysctl -n hw.pagesize) B"
  echo "os: $(sw_vers -productName) $(sw_vers -productVersion) ($(uname -r))"
  echo "python: $($PY -c 'import sys,numpy,numba;print(sys.version.split()[0],"numpy",numpy.__version__,"numba",numba.__version__)')"
  echo "load before run: $(uptime | sed 's/.*load averages*: //')"
} | tee logs/hw_info.log
echo "### $(date '+%F %T') \$ $PY false_sharing_lab.py $*" | tee logs/false_sharing_lab.log
"$PY" false_sharing_lab.py "$@" 2>&1 | tee -a logs/false_sharing_lab.log
