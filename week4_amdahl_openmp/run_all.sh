#!/usr/bin/env bash
# Runs every benchmark one at a time; console output is tee'd into logs/.
set -euo pipefail
cd "$(dirname "$0")"
CSV=results.csv
echo "experiment,variant,threads,run1_cold_s,run2_s,run3_s,avg_run2_run3_s,max_steps,checksum,hits_gt100,thread_finish_spread_s" > "$CSV"
date '+%Y-%m-%d %H:%M:%S %Z' | tee logs/run_timestamp.txt
./collatz_seq "$CSV"                        | tee logs/phase2_seq.txt
./collatz scale "$CSV" 1 2 4 6 8 10 12 16        | tee logs/phase3_scaling.txt
./collatz false "$CSV" 10                   | tee logs/phase4a_false_sharing.txt
./collatz sched "$CSV" 10                   | tee logs/phase4b_scheduling.txt
date '+%Y-%m-%d %H:%M:%S %Z' | tee -a logs/run_timestamp.txt
