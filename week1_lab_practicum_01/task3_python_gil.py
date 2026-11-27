"""Task 3 (Python cross-check): same experiment with CPython threads.

10 threads x 1,000,000 `counter += 1`, 10 runs unlocked, then 10 runs with
threading.Lock. With the GIL only one thread runs bytecode at a time and
CPython 3.13 only switches threads at loop back-edges / calls, never between
the LOAD and STORE of `counter += 1`, so no updates are lost here.
Usage: python3 task3_python_gil.py | tee logs/task3_python_gil.txt
"""
import sys
import threading
import time

THREADS = 10
ITERS = 1_000_000
RUNS = 10
EXPECTED = THREADS * ITERS
counter = 0
lock = threading.Lock()


def unlocked():
    global counter
    for _ in range(ITERS):
        counter += 1


def locked():
    global counter
    for _ in range(ITERS):
        with lock:
            counter += 1


def run(fn):
    global counter
    counter = 0
    threads = [threading.Thread(target=fn) for _ in range(THREADS)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return (time.perf_counter() - t0) * 1e3, counter


def main():
    gil = sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else True
    print(f"Python {sys.version.split()[0]}, GIL enabled: {gil}")
    for name, fn in (("UNLOCKED", unlocked), ("LOCKED (threading.Lock)", locked)):
        print(f"\n{name}: expected {EXPECTED}")
        times = []
        for r in range(1, RUNS + 1):
            ms, value = run(fn)
            times.append(ms)
            print(f"run #{r:2d}  output={value:9d}  error={EXPECTED - value:9d}  time={ms:8.1f} ms")
        print(f"mean time = {sum(times) / RUNS:.1f} ms")


if __name__ == "__main__":
    main()
