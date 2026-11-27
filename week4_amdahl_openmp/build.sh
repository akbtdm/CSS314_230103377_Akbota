#!/usr/bin/env bash
# Build both programs with -O2 and OpenMP (Apple clang + Homebrew libomp).
set -euo pipefail
cd "$(dirname "$0")"
OMP_PREFIX="${OMP_PREFIX:-/opt/homebrew/opt/libomp}"
FLAGS=(-O2 -Wall -Wextra -Xpreprocessor -fopenmp -I"$OMP_PREFIX/include" -L"$OMP_PREFIX/lib" -lomp -Wl,-rpath,"$OMP_PREFIX/lib")
clang --version | head -1
for prog in collatz_seq collatz; do
  echo "\$ clang ${FLAGS[*]} $prog.c -o $prog"
  clang "${FLAGS[@]}" "$prog.c" -o "$prog"
done
echo "build OK (no warnings printed above means zero warnings)"
