/* Supplement to task3_sync.py: the UnsafeCounter without the GIL.
 * 4 pthreads each do 500,000 unprotected `counter++` (LOAD, ADD, STORE),
 * so the target is 2,000,000. `volatile` stops clang from folding the loop
 * into one add, so every increment is a real load/add/store.
 * Build: clang -O2 -pthread -o task3_race_c task3_race_c.c */
#include <pthread.h>
#include <stdio.h>

#define THREADS 4
#define OPS_PER_THREAD 500000L

static volatile long counter;

static void *work(void *arg) {
    (void)arg;
    for (long i = 0; i < OPS_PER_THREAD; i++) counter++;
    return NULL;
}

int main(void) {
    for (int run = 1; run <= 5; run++) {
        pthread_t t[THREADS];
        counter = 0;
        for (int i = 0; i < THREADS; i++) pthread_create(&t[i], NULL, work, NULL);
        for (int i = 0; i < THREADS; i++) pthread_join(t[i], NULL);
        long target = THREADS * OPS_PER_THREAD;
        printf("Run %d: Value = %ld / %ld | lost updates = %ld\n", run, counter, target, target - counter);
    }
    return 0;
}
