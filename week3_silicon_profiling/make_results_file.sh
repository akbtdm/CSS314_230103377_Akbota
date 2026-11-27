#!/bin/sh
# Concatenate the raw console logs (saved with tee) into the handout's
# required logfile results_<student_id>.txt. No numbers are edited.
cd "$(dirname "$0")"
OUT=results_230103377.txt
{
  echo "Akbota Kutushova | 230103377 | Silicon Profiling Lab | raw terminal logs"
  for f in hw_audit task1_amdahl task1_pool_overhead task2_falsesharing task2_python_sizes task2_falsesharing_c \
           task3_sync task3_unsafe_inc_bytecode task3_race_c task3_race_c_asm task3_lockless \
           task4_roofline task4_roofline_repeats task4_bandwidth_first_attempt_50M task4_bandwidth \
           bonus_gil_scaling; do
    echo; echo "################ logs/$f.txt ################"; cat "logs/$f.txt"
  done
} > "$OUT"
echo "wrote $OUT ($(wc -l < "$OUT") lines)"
