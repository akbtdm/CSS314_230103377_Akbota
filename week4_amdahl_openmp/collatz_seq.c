/*
 * collatz_seq.c - Phase 2 sequential baseline (Amdahl Reality Gap worksheet)
 * Student: Akbota Kutushova, ID 230103377
 * N = 10,000,000 + 3377 * 1,000 = 13,377,000
 *
 * Build (macOS, Apple clang + Homebrew libomp, `brew install libomp`, used only for omp_get_wtime):
 *   clang -O2 -Wall -Wextra -Xpreprocessor -fopenmp \
 *         -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp \
 *         -Wl,-rpath,/opt/homebrew/opt/libomp/lib collatz_seq.c -o collatz_seq
 * Worksheet equivalent on Linux: gcc -O2 -fopenmp collatz_seq.c -o collatz_seq
 * Run: ./collatz_seq [results.csv]
 */
#include <inttypes.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>

#define STUDENT_ID_LAST4 3377ULL
#define N_WORK (10000000ULL + STUDENT_ID_LAST4 * 1000ULL)
#define MOD 1000000007ULL
#define RUNS 3

static inline uint32_t collatz_steps(uint64_t n) {
    uint32_t steps = 0;
    while (n > 1) {
        if ((n & 1) == 0) n >>= 1;
        else n = 3 * n + 1;
        steps++;
    }
    return steps;
}

int main(int argc, char **argv) {
    const char *csv_path = argc > 1 ? argv[1] : NULL;
    double t[RUNS];
    uint32_t max_steps = 0;
    uint64_t max_arg = 0, checksum = 0;

    for (int r = 0; r < RUNS; r++) {
        uint32_t mx = 0;
        uint64_t arg = 0, sum = 0;
        double t0 = omp_get_wtime();
        for (uint64_t i = 1; i <= N_WORK; i++) {
            uint32_t s = collatz_steps(i);
            sum += s;
            if (s > mx) { mx = s; arg = i; }
        }
        t[r] = omp_get_wtime() - t0;
        max_steps = mx; max_arg = arg; checksum = sum % MOD;
        printf("seq run %d: %.6f s  max_steps=%u (n=%" PRIu64 ")  checksum=%" PRIu64 "\n",
               r + 1, t[r], mx, arg, checksum);
    }
    double t_seq = (t[1] + t[2]) / 2.0;
    printf("N=%" PRIu64 "  T_seq=(run2+run3)/2=%.6f s  max_steps=%u at n=%" PRIu64
           "  checksum(sum mod 1e9+7)=%" PRIu64 "\n",
           (uint64_t)N_WORK, t_seq, max_steps, max_arg, checksum);

    if (csv_path) {
        FILE *f = fopen(csv_path, "a");
        if (!f) { perror("fopen"); return 1; }
        fprintf(f, "seq,sequential,1,%.6f,%.6f,%.6f,%.6f,%u,%" PRIu64 ",,\n",
                t[0], t[1], t[2], t_seq, max_steps, checksum);
        fclose(f);
    }
    return 0;
}
