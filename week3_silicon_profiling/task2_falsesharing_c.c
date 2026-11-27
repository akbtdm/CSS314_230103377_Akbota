/* Supplement to task2_falsesharing.py: the same experiment without the GIL.
 * 4 pthreads each increment their own 8-byte counter ITERATIONS times.
 * The counters sit STRIDE bytes apart in one 128-byte-aligned buffer:
 *   8   -> adjacent (all 4 counters in one cache line)
 *   64  -> 64-byte padding (enough on x86, but the M4 line is 128 bytes,
 *          so counters 0/1 and 2/3 still share a line)
 *   128 -> one counter per 128-byte M4 cache line
 * Build: clang -O2 -pthread -o task2_falsesharing_c task2_falsesharing_c.c */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define THREADS 4
#define ITERATIONS 100000000L
#define TRIALS 3

static char *buf;
static long stride;

static void *worker(void *arg) {
    long id = (long)arg;
    volatile long *ctr = (volatile long *)(buf + id * stride);
    for (long i = 0; i < ITERATIONS; i++) (*ctr)++;
    return NULL;
}

static double run(long s) {
    stride = s;
    for (int i = 0; i < THREADS * 128; i++) buf[i] = 0;
    pthread_t t[THREADS];
    struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    for (long i = 0; i < THREADS; i++) pthread_create(&t[i], NULL, worker, (void *)i);
    for (int i = 0; i < THREADS; i++) pthread_join(t[i], NULL);
    clock_gettime(CLOCK_MONOTONIC, &b);
    for (long i = 0; i < THREADS; i++)
        if (*(long *)(buf + i * s) != ITERATIONS) { fprintf(stderr, "bad count\n"); exit(1); }
    return (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9;
}

int main(void) {
    if (posix_memalign((void **)&buf, 128, THREADS * 128) != 0) return 1;
    long strides[] = {8, 64, 128};
    printf("C pthreads, %d threads x %ld increments, buffer aligned to 128 B\n", THREADS, ITERATIONS);
    for (int s = 0; s < 3; s++) {
        printf("stride %3ld B:", strides[s]);
        for (int k = 0; k < TRIALS; k++) printf("  %.4fs", run(strides[s]));
        printf("\n");
    }
    free(buf);
    return 0;
}
