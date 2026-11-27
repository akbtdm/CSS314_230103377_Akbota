#!/bin/sh
# Report Section I: hardware and software of the machine that ran the labs.
echo "date: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "--- CPU / cores / caches (sysctl) ---"
sysctl machdep.cpu.brand_string hw.model hw.physicalcpu hw.logicalcpu hw.ncpu \
       hw.nperflevels hw.perflevel0.name hw.perflevel0.physicalcpu hw.perflevel0.logicalcpu \
       hw.perflevel0.l1dcachesize hw.perflevel0.l1icachesize hw.perflevel0.l2cachesize \
       hw.perflevel0.cpusperl2 \
       hw.perflevel1.name hw.perflevel1.physicalcpu hw.perflevel1.logicalcpu \
       hw.perflevel1.l1dcachesize hw.perflevel1.l1icachesize hw.perflevel1.l2cachesize \
       hw.perflevel1.cpusperl2 \
       hw.cachelinesize hw.l1dcachesize hw.l2cachesize hw.memsize 2>/dev/null
L3=$(sysctl -n hw.l3cachesize 2>/dev/null)
echo "hw.l3cachesize: ${L3:-not reported by sysctl (no L3 on this chip)}"
echo "--- OS ---"
sw_vers
uname -a
echo "--- runtimes ---"
python3 - <<'PY'
import os, sys, platform, multiprocessing as mp
os.environ.setdefault("NUMBA_THREADING_LAYER", "omp")
os.environ.setdefault("KMP_WARNINGS", "0")
import numpy, numba, psutil, matplotlib
from numba import njit, prange
@njit(parallel=True)
def f(n):
    s = 0
    for i in prange(n):
        s += i
    return s
f(100)
print(f"python {sys.version.split()[0]} ({sys.executable})")
print(f"numpy {numpy.__version__} | numba {numba.__version__} | psutil {psutil.__version__} | matplotlib {matplotlib.__version__}")
print(f"numba threading layer in use: {numba.threading_layer()} | NUMBA_NUM_THREADS default: {numba.config.NUMBA_NUM_THREADS}")
print(f"GIL enabled: {getattr(sys, '_is_gil_enabled', lambda: True)()} | multiprocessing start method: {mp.get_start_method()}")
print(f"os.cpu_count(): {os.cpu_count()} | platform: {platform.platform()}")
PY
