import java.util.SplittableRandom;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Monte Carlo pi: data race, synchronization trap, and OpenMP-style reduction.
 * Student: Akbota Kutushova (230103377). Run: javac PiLab.java && java PiLab
 */
public class PiLab {
    static final long SEED = 230103377L;
    static final long PART1_POINTS = 50_000_000L;
    static final long PART3_POINTS = 100_000_000L;
    static final int PART12_THREADS = 4;
    static final int[] PART3_THREADS = {1, 2, 4, 8, 16, 32};
    static final int REPEATS = 3;

    static long totalHits = 0;                       // Part 1: shared, unsynchronized
    static final AtomicLong atomicHits = new AtomicLong();
    static long syncHits = 0;
    static final Object LOCK = new Object();

    interface Body { void run(int tid, long points); }

    static long runThreads(int nThreads, long totalPoints, Body body) throws InterruptedException {
        Thread[] threads = new Thread[nThreads];
        long t0 = System.nanoTime();
        for (int t = 0; t < nThreads; t++) {
            final int tid = t;
            final long share = totalPoints / nThreads + (t < totalPoints % nThreads ? 1 : 0);
            threads[t] = new Thread(() -> body.run(tid, share));
            threads[t].start();
        }
        for (Thread th : threads) th.join();
        return System.nanoTime() - t0;
    }

    static boolean inside(SplittableRandom rng) {
        double x = rng.nextDouble(), y = rng.nextDouble();
        return x * x + y * y <= 1.0;
    }

    static long serialHits(long points) {
        SplittableRandom rng = new SplittableRandom(SEED);
        long hits = 0;
        for (long i = 0; i < points; i++) if (inside(rng)) hits++;
        return hits;
    }

    static void part1() throws InterruptedException {
        System.out.println("PART 1 - The Phantom Bug: 4 threads, totalHits++ on a shared static long, "
                + PART1_POINTS / 1_000_000 + "M points");
        for (int run = 1; run <= 5; run++) {
            totalHits = 0;
            long ns = runThreads(PART12_THREADS, PART1_POINTS, (tid, n) -> {
                SplittableRandom rng = new SplittableRandom(SEED + 1000L * tid);
                for (long i = 0; i < n; i++) if (inside(rng)) totalHits++;
            });
            System.out.printf("  run %d: pi = %.6f   hits = %,d of expected ~%,d   (%d ms)%n",
                    run, 4.0 * totalHits / PART1_POINTS, totalHits, (long) (PART1_POINTS * Math.PI / 4), ns / 1_000_000);
        }
    }

    static void part2() throws InterruptedException {
        System.out.println("\nPART 2 - The Synchronization Trap: " + PART1_POINTS / 1_000_000 + "M points");
        long t0 = System.nanoTime();
        long hits = serialHits(PART1_POINTS);
        long serialMs = (System.nanoTime() - t0) / 1_000_000;
        System.out.printf("  single-threaded plain loop : pi = %.6f   %6d ms%n", 4.0 * hits / PART1_POINTS, serialMs);

        atomicHits.set(0);
        long atomicMs = runThreads(PART12_THREADS, PART1_POINTS, (tid, n) -> {
            SplittableRandom rng = new SplittableRandom(SEED + 1000L * tid);
            for (long i = 0; i < n; i++) if (inside(rng)) atomicHits.incrementAndGet();
        }) / 1_000_000;
        System.out.printf("  4 threads, AtomicLong      : pi = %.6f   %6d ms  (%.2fx the single-thread time)%n",
                4.0 * atomicHits.get() / PART1_POINTS, atomicMs, (double) atomicMs / serialMs);

        syncHits = 0;
        long syncMs = runThreads(PART12_THREADS, PART1_POINTS, (tid, n) -> {
            SplittableRandom rng = new SplittableRandom(SEED + 1000L * tid);
            for (long i = 0; i < n; i++) if (inside(rng)) synchronized (LOCK) { syncHits++; }
        }) / 1_000_000;
        System.out.printf("  4 threads, synchronized    : pi = %.6f   %6d ms  (%.2fx the single-thread time)%n",
                4.0 * syncHits / PART1_POINTS, syncMs, (double) syncMs / serialMs);
    }

    static double reductionRunMs(int nThreads, long[] hitsOut) throws InterruptedException {
        long[] partial = new long[nThreads * 16];    // 16 longs = 128 B apart: one M4 cache line each
        long ns = runThreads(nThreads, PART3_POINTS, (tid, n) -> {
            SplittableRandom rng = new SplittableRandom(SEED + 1000L * tid);
            long local = 0;                          // private counter, like reduction(+:totalHits)
            for (long i = 0; i < n; i++) if (inside(rng)) local++;
            partial[tid * 16] = local;               // written once, at the end
        });
        long sum = 0;
        for (int t = 0; t < nThreads; t++) sum += partial[t * 16];
        hitsOut[0] = sum;
        return ns / 1e6;
    }

    static void part3() throws InterruptedException {
        System.out.println("\nPART 3 - OpenMP-style reduction: " + PART3_POINTS / 1_000_000
                + "M points, median of " + REPEATS + " runs after one warm-up");
        reductionRunMs(4, new long[1]);              // JIT warm-up
        double base = 0;
        System.out.println("  Threads (T) | Runtime (ms) | Speedup vs 1 thread | Efficiency | pi");
        System.out.println("  ------------+--------------+---------------------+------------+---------");
        for (int t : PART3_THREADS) {
            double[] samples = new double[REPEATS];
            long[] hits = new long[1];
            for (int r = 0; r < REPEATS; r++) samples[r] = reductionRunMs(t, hits);
            java.util.Arrays.sort(samples);
            double ms = samples[REPEATS / 2];
            if (t == 1) base = ms;
            double speedup = base / ms;
            System.out.printf("  %12s | %12.1f | %18.2fx | %9.1f%% | %.6f%n",
                    t == 1 ? "1 (baseline)" : Integer.toString(t), ms, speedup, 100 * speedup / t,
                    4.0 * hits[0] / PART3_POINTS);
        }
    }

    public static void main(String[] args) throws InterruptedException {
        java.util.Locale.setDefault(java.util.Locale.ROOT);
        System.out.printf("Java %s on %s %s, %d logical cores (Apple M4: 4 performance + 6 efficiency)%n%n",
                System.getProperty("java.version"), System.getProperty("os.name"),
                System.getProperty("os.arch"), Runtime.getRuntime().availableProcessors());
        part1();
        part2();
        part3();
    }
}
