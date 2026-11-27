/*
 * collatz.c - Phases 3 and 4 of the "Amdahl Reality Gap" worksheet (OpenMP)
 * Student: Akbota Kutushova, ID 230103377
 * N = 10,000,000 + 3377 * 1,000 = 13,377,000
 *
 * Build (macOS, Apple clang has no bundled OpenMP runtime; Homebrew libomp, `brew install libomp`):
 *   clang -O2 -Wall -Wextra -Xpreprocessor -fopenmp \
 *         -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp \
 *         -Wl,-rpath,/opt/homebrew/opt/libomp/lib collatz.c -o collatz
 * Worksheet equivalent on Linux: gcc -O2 -fopenmp collatz.c -o collatz
 *
 * Run:
 *   ./collatz scale  results.csv 1 2 4 6 8 10 12 16   # Table 1 (Phase 3)
 *   ./collatz false  results.csv 10              # Table 2 (Experiment A)
 *   ./collatz sched  results.csv 10              # Table 3 (Experiment B)
 *
 * Every configuration is run 3 times in the same process: run 1 is the cold
 * warm-up and is discarded, the reported time is (run 2 + run 3) / 2.
 * Timing uses omp_get_wtime(). Each thread also records when it finished its
 * share of the loop (nowait), so the spread max-min of those finish times
 * shows load imbalance (idle time at the final barrier). The spread is exact
 * for every variant except 'reduction', whose reduction clause may add a
 * barrier before the finish time is read.
 */
#include <inttypes.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STUDENT_ID_LAST4 3377ULL
#define N_WORK (10000000ULL + STUDENT_ID_LAST4 * 1000ULL)
#define MOD 1000000007ULL
#define RUNS 3
#define MAX_THREADS 64
#define HIT_THRESHOLD 100
#define LINE_M4 128 /* hw.cachelinesize on Apple M4 */

static inline uint32_t collatz_steps(uint64_t n) {
    uint32_t steps = 0;
    while (n > 1) {
        if ((n & 1) == 0) n >>= 1;
        else n = 3 * n + 1;
        steps++;
    }
    return steps;
}

typedef struct {
    double seconds;
    double spread;      /* max - min thread finish time inside the loop */
    uint32_t max_steps;
    uint64_t checksum;  /* sum of stopping times mod 1e9+7 */
    uint64_t hits;      /* numbers with > 100 steps (Experiment A only) */
} RunResult;

typedef enum { V_PLAIN, V_NAIVE, V_PAD64, V_PAD128, V_REDUCTION } Variant;

/* Experiment A counters: all live outside the parallel region. */
static int hit_count[MAX_THREADS];
struct Padded { int count; char pad[60]; };            /* worksheet struct, 64 B */
struct Padded128 { int count; char pad[LINE_M4 - sizeof(int)]; };
static struct Padded hit_pad64[MAX_THREADS] __attribute__((aligned(LINE_M4)));
static struct Padded128 hit_pad128[MAX_THREADS] __attribute__((aligned(LINE_M4)));

static double spread_of(const double *finish, int k) {
    double lo = finish[0], hi = finish[0];
    for (int t = 1; t < k; t++) {
        if (finish[t] < lo) lo = finish[t];
        if (finish[t] > hi) hi = finish[t];
    }
    return hi - lo;
}

/* One timed pass over [1, N]. Schedule comes from omp_set_schedule (runtime). */
static RunResult run_once(int k, Variant v) {
    RunResult r = {0};
    double finish[MAX_THREADS] = {0};
    uint64_t sum = 0, total_hits = 0;
    uint32_t mx = 0;

    memset(hit_count, 0, sizeof hit_count);
    memset(hit_pad64, 0, sizeof hit_pad64);
    memset(hit_pad128, 0, sizeof hit_pad128);

    double t0 = omp_get_wtime();
#pragma omp parallel num_threads(k)
    {
        /* Private accumulators, merged once per thread after the loop. This is
         * what reduction(+:sum) reduction(max:mx) does, written out by hand so
         * each thread's finish time is read BEFORE any reduction barrier
         * (libomp's tree reduction for larger teams contains a barrier). */
        int tid = omp_get_thread_num();
        uint64_t lsum = 0;
        uint32_t lmx = 0;
        if (v == V_REDUCTION) {
            /* Variant 2a: OpenMP reduction clause on the hit counter. */
#pragma omp for schedule(runtime) reduction(+:total_hits) nowait
            for (uint64_t i = 1; i <= N_WORK; i++) {
                uint32_t s = collatz_steps(i);
                lsum += s;
                if (s > lmx) lmx = s;
                if (s > HIT_THRESHOLD) total_hits++;
            }
        } else {
#pragma omp for schedule(runtime) nowait
            for (uint64_t i = 1; i <= N_WORK; i++) {
                uint32_t s = collatz_steps(i);
                lsum += s;
                if (s > lmx) lmx = s;
                if (s > HIT_THRESHOLD) {
                    switch (v) {
                    case V_NAIVE: hit_count[omp_get_thread_num()]++; break;        /* Variant 1 */
                    case V_PAD64: hit_pad64[omp_get_thread_num()].count++; break;  /* Variant 2b */
                    case V_PAD128: hit_pad128[omp_get_thread_num()].count++; break;/* Variant 2c */
                    default: break;
                    }
                }
            }
        }
        finish[tid] = omp_get_wtime() - t0;
#pragma omp critical(merge)
        {
            sum += lsum;
            if (lmx > mx) mx = lmx;
        }
    }
    r.seconds = omp_get_wtime() - t0;
    r.spread = spread_of(finish, k);

    for (int t = 0; t < k; t++)
        total_hits += (uint64_t)hit_count[t] + (uint64_t)hit_pad64[t].count +
                      (uint64_t)hit_pad128[t].count;
    r.max_steps = mx;
    r.checksum = sum % MOD;
    r.hits = total_hits;
    return r;
}

static void bench(FILE *csv, const char *exp, const char *label, int k, Variant v) {
    RunResult res[RUNS];
    for (int i = 0; i < RUNS; i++) {
        res[i] = run_once(k, v);
        printf("%-6s %-22s k=%2d run %d: %.6f s  spread=%.6f s  max=%u  checksum=%" PRIu64
               "  hits=%" PRIu64 "\n", exp, label, k, i + 1, res[i].seconds, res[i].spread,
               res[i].max_steps, res[i].checksum, res[i].hits);
        fflush(stdout);
    }
    double avg = (res[1].seconds + res[2].seconds) / 2.0;
    double spread = (res[1].spread + res[2].spread) / 2.0;
    printf("%-6s %-22s k=%2d AVG(run2,run3)=%.6f s  avg_spread=%.6f s\n", exp, label, k, avg, spread);
    fprintf(csv, "%s,%s,%d,%.6f,%.6f,%.6f,%.6f,%u,%" PRIu64 ",%" PRIu64 ",%.6f\n", exp, label, k,
            res[0].seconds, res[1].seconds, res[2].seconds, avg, res[2].max_steps,
            res[2].checksum, res[2].hits, spread);
    fflush(csv);
}

static void do_scale(FILE *csv, int argc, char **argv) {
    omp_set_schedule(omp_sched_static, 0); /* default "#pragma omp parallel for" */
    for (int a = 3; a < argc; a++) {
        int k = atoi(argv[a]);
        if (k < 1 || k > MAX_THREADS) { fprintf(stderr, "bad k %s\n", argv[a]); continue; }
        bench(csv, "scale", "static_default", k, V_PLAIN);
    }
}

static void do_false(FILE *csv, int k) {
    omp_set_schedule(omp_sched_static, 0);
    bench(csv, "false", "naive_hits[tid]++", k, V_NAIVE);
    bench(csv, "false", "padded64_struct", k, V_PAD64);
    bench(csv, "false", "padded128_struct", k, V_PAD128);
    bench(csv, "false", "reduction", k, V_REDUCTION);
}

static void do_sched(FILE *csv, int k) {
    static const struct { omp_sched_t kind; int chunk; const char *label; } cfg[] = {
        {omp_sched_static, 0, "static"},
        {omp_sched_static, 1000, "static_1000"},
        {omp_sched_dynamic, 100, "dynamic_100"},
        {omp_sched_dynamic, 10000, "dynamic_10000"},
        {omp_sched_guided, 0, "guided"},
        {omp_sched_dynamic, 1, "dynamic_1"},
        {omp_sched_dynamic, 10, "dynamic_10"},
        {omp_sched_dynamic, 1000, "dynamic_1000"},
        {omp_sched_dynamic, 100000, "dynamic_100000"},
        {omp_sched_dynamic, 1000000, "dynamic_1000000"},
    };
    for (size_t c = 0; c < sizeof cfg / sizeof cfg[0]; c++) {
        omp_set_schedule(cfg[c].kind, cfg[c].chunk);
        bench(csv, "sched", cfg[c].label, k, V_PLAIN);
    }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s scale|false|sched results.csv k...\n", argv[0]);
        return 2;
    }
    FILE *csv = fopen(argv[2], "a");
    if (!csv) { perror("fopen"); return 1; }
    printf("N=%" PRIu64 "  omp_get_max_threads=%d  omp_get_num_procs=%d\n", (uint64_t)N_WORK,
           omp_get_max_threads(), omp_get_num_procs());

    int k = atoi(argv[3]);
    if (strcmp(argv[1], "scale") == 0) do_scale(csv, argc, argv);
    else if (strcmp(argv[1], "false") == 0) do_false(csv, k);
    else if (strcmp(argv[1], "sched") == 0) do_sched(csv, k);
    else { fprintf(stderr, "unknown mode %s\n", argv[1]); fclose(csv); return 2; }
    fclose(csv);
    return 0;
}
