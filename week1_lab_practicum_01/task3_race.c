/* task3_race.c - Task 3: unsynchronized shared counter vs mutex-protected counter.
 * 10 pthreads x 1,000,000 increments of one shared global int, 10 runs each.
 * `volatile` forces a real Load -> Add -> Store to memory on every iteration
 * (without it clang keeps the counter in a register and the race looks different).
 * Build: clang -O2 -pthread -o task3_race task3_race.c
 * Why C: CPython's GIL makes `counter += 1` effectively atomic (see task3_python_gil.py),
 * so a true hardware race needs real parallel threads touching shared memory. */
#include <pthread.h>
#include <stdio.h>
#include <time.h>

#define THREADS 10
#define ITERS 1000000
#define RUNS 10
#define EXPECTED (THREADS * ITERS)

static volatile int counter = 0;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

static void *unlocked_worker(void *arg) {
    (void)arg;
    for (int i = 0; i < ITERS; i++) counter++; /* ldr w8,[x]; add w8,w8,#1; str w8,[x] */
    return NULL;
}

static void *locked_worker(void *arg) {
    (void)arg;
    for (int i = 0; i < ITERS; i++) {
        pthread_mutex_lock(&lock);
        counter++;
        pthread_mutex_unlock(&lock);
    }
    return NULL;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static double run(void *(*fn)(void *), int *result) {
    pthread_t t[THREADS];
    counter = 0;
    double t0 = now_ms();
    for (int i = 0; i < THREADS; i++) pthread_create(&t[i], NULL, fn, NULL);
    for (int i = 0; i < THREADS; i++) pthread_join(t[i], NULL);
    *result = counter;
    return now_ms() - t0;
}

int main(void) {
    int value;
    double sum_unlocked = 0, sum_locked = 0;
    printf("UNLOCKED: %d threads x %d increments, expected %d\n", THREADS, ITERS, EXPECTED);
    for (int r = 1; r <= RUNS; r++) {
        double ms = run(unlocked_worker, &value);
        sum_unlocked += ms;
        printf("run #%2d  output=%9d  error=%9d  time=%8.2f ms\n", r, value, EXPECTED - value, ms);
    }
    printf("\nLOCKED (pthread_mutex): same work\n");
    for (int r = 1; r <= RUNS; r++) {
        double ms = run(locked_worker, &value);
        sum_locked += ms;
        printf("run #%2d  output=%9d  error=%9d  time=%8.2f ms\n", r, value, EXPECTED - value, ms);
    }
    printf("\nmean unlocked = %.2f ms, mean locked = %.2f ms, slowdown = %.1fx\n",
           sum_unlocked / RUNS, sum_locked / RUNS, sum_locked / sum_unlocked);
    return 0;
}
