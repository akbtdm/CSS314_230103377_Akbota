/* clock_probe.c - measure the effective core clock empirically.
 * A chain of dependent integer adds retires 1 add per cycle, so
 * adds / seconds ~= cycles / second. Run once with P-core QoS and once with
 * background QoS (macOS schedules background work on E-cores).
 * Build: clang -O2 -o clock_probe clock_probe.c */
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define ITERS 20000000ULL
#define ADDS_PER_ITER 100ULL
#define A10 "add %0, %0, #1\n\t" "add %0, %0, #1\n\t" "add %0, %0, #1\n\t" "add %0, %0, #1\n\t" "add %0, %0, #1\n\t" \
            "add %0, %0, #1\n\t" "add %0, %0, #1\n\t" "add %0, %0, #1\n\t" "add %0, %0, #1\n\t" "add %0, %0, #1\n\t"

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double probe(void) {
    unsigned long long x = 0;
    double t0 = now();
    for (unsigned long long i = 0; i < ITERS; i++) {
        __asm__ volatile(A10 A10 A10 A10 A10 A10 A10 A10 A10 A10 : "+r"(x));
    }
    double dt = now() - t0;
    if (x != ITERS * ADDS_PER_ITER) fprintf(stderr, "checksum mismatch\n");
    return (double)(ITERS * ADDS_PER_ITER) / dt / 1e9;
}

int main(int argc, char **argv) {
    int background = argc > 1 && strcmp(argv[1], "efficiency") == 0;
    pthread_set_qos_class_self_np(background ? QOS_CLASS_BACKGROUND : QOS_CLASS_USER_INTERACTIVE, 0);
    probe(); /* warm-up so the core ramps up its clock */
    for (int r = 1; r <= 3; r++)
        printf("%s-core QoS run %d: %.3f GHz (dependent adds per ns)\n",
               background ? "efficiency" : "performance", r, probe());
    return 0;
}
