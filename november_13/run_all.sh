#!/bin/zsh
# Runs the whole lab sequentially (never in parallel: the phases must not disturb each other).
# Usage: ./run_all.sh            full configuration, results/ and logs/
#        ./run_all.sh --quick    smoke test, results_quick/
set -u
cd "$(dirname "$0")"
mkdir -p logs results
{
  echo "# macOS hardware audit (lab section 1.1 commands)  $(date)"
  echo '$ sysctl -n machdep.cpu.brand_string'
  sysctl -n machdep.cpu.brand_string
  echo '$ sysctl hw.physicalcpu hw.logicalcpu hw.cachelinesize hw.l1dcachesize hw.l2cachesize hw.l3cachesize'
  sysctl hw.physicalcpu hw.logicalcpu hw.cachelinesize hw.l1dcachesize hw.l2cachesize hw.l3cachesize 2>&1
  echo '$ sysctl hw.perflevel0.physicalcpu hw.perflevel0.name hw.perflevel1.physicalcpu hw.perflevel1.name hw.memsize'
  sysctl hw.perflevel0.physicalcpu hw.perflevel0.name hw.perflevel1.physicalcpu hw.perflevel1.name hw.memsize 2>&1
  echo '$ sysctl hw.perflevel0.l1dcachesize hw.perflevel0.l2cachesize hw.perflevel1.l1dcachesize hw.perflevel1.l2cachesize'
  sysctl hw.perflevel0.l1dcachesize hw.perflevel0.l2cachesize hw.perflevel1.l1dcachesize hw.perflevel1.l2cachesize 2>&1
  echo '$ python3 --version && sw_vers'
  python3 --version; sw_vers
} > logs/hw_audit.log 2>&1
for t in task0_cal task1_falsesharing task2_scaling task3a_roofline task3b_gemm task4_scheduling; do
  echo "### $t  start $(date +%T)"
  python3 "$t.py" "$@" 2>&1 | tee "logs/$t.log"
  echo "### $t  end   $(date +%T)"
done
echo "### ALL DONE $(date +%T)"
