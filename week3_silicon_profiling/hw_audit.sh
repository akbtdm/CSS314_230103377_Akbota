#!/bin/sh
# Hardware topology queries from the worksheet appendix (macOS rows).
echo "== date =="; date
echo "== sysctl -n machdep.cpu.brand_string =="; sysctl -n machdep.cpu.brand_string
echo "== core topology =="
sysctl hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu hw.physicalcpu hw.logicalcpu
echo "== caches =="
sysctl hw.l1dcachesize hw.l2cachesize hw.cachelinesize \
  hw.perflevel0.l1dcachesize hw.perflevel0.l2cachesize \
  hw.perflevel1.l1dcachesize hw.perflevel1.l2cachesize
echo "l3 keys found: $(sysctl -a 2>/dev/null | grep -c l3cache)"
echo "== memory =="
sysctl hw.memsize
system_profiler SPMemoryDataType
echo "== machine =="
system_profiler SPHardwareDataType | grep -E "Model Name|Model Identifier|Chip|Total Number of Cores|Memory"
echo "== OS / Python =="
sw_vers; uname -sr; python3 --version
python3 -c "import multiprocessing as mp, numpy, sys; print('mp start method:', mp.get_start_method()); print('numpy', numpy.__version__); print('GIL enabled:', getattr(sys, '_is_gil_enabled', lambda: True)())"
