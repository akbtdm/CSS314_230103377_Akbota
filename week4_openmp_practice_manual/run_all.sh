#!/bin/sh
# Runs every lab one at a time and tees the console output into logs/.
# Usage: ./run_all.sh            (about 4-6 minutes on the Apple M4)
set -e
cd "$(dirname "$0")"
export NUMBA_THREADING_LAYER=omp KMP_WARNINGS=0 PYTHONUNBUFFERED=1
mkdir -p logs results

./hw_info.sh | tee logs/hw_info.txt

# Lab 1, Task 1.1: starter (ThreadPoolExecutor) and Numba OpenMP team, 10 runs each
: > logs/lab1_task1_1_nondeterminism.txt
: > logs/lab1_task1_1_numba_team.txt
for i in 1 2 3 4 5 6 7 8 9 10; do
  echo "##### run $i: starter_fork_join.py" | tee -a logs/lab1_task1_1_nondeterminism.txt
  python3 lab1/starter_fork_join.py | tee -a logs/lab1_task1_1_nondeterminism.txt
done
for i in 1 2 3 4 5 6 7 8 9 10; do
  printf "run %2d: " "$i" | tee -a logs/lab1_task1_1_numba_team.txt
  python3 lab1/lab1_fork_join.py team 8 | tee -a logs/lab1_task1_1_numba_team.txt
done
NUMBA_NUM_THREADS=64 python3 lab1/lab1_fork_join.py sweep    | tee logs/lab1_task1_2_sweep.txt
NUMBA_NUM_THREADS=10 python3 lab1/lab1_fork_join.py saturate | tee logs/lab1_task1_3_saturation.txt

# Lab 2
NUMBA_NUM_THREADS=16 python3 lab2/lab2_pi.py all | tee logs/lab2_pi.txt

# Lab 3
NUMBA_NUM_THREADS=16 python3 lab3/lab3_mandelbrot.py | tee logs/lab3_mandelbrot.txt

# Lab 4 (perf stat is Linux-only; not available on macOS)
NUMBA_NUM_THREADS=16 python3 lab4/lab4_false_sharing.py | tee logs/lab4_false_sharing.txt

# Lab 5
python3 lab5/starter_process_pool.py | tee logs/lab5_starter_process_pool.txt
python3 lab5/lab5_merge_sort.py sweep    | tee logs/lab5_cutoff_sweep.txt
python3 lab5/lab5_merge_sort.py workspan | tee logs/lab5_workspan.txt

# Lab 6 (bonus)
python3 lab6/lab6_pipeline.py | tee logs/lab6_pipeline.txt

# Figures from the CSVs
python3 make_plots.py | tee logs/make_plots.txt
