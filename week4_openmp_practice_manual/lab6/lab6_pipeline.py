"""Lab 6 (bonus): 3-stage pipeline with bounded blocking queues.

Stage 1 reader   : takes 512x512 float64 frames from an in-memory pool, stamps
                   the ingest time, puts them on q1 (bounded, maxsize QSIZE).
Stage 2 filter   : 5x5 Gaussian blur applied `passes` times (Numba nogil kernel,
                   so it runs outside the GIL); W worker threads share q1.
Stage 3 writer   : min / max / mean / checksum (Numba nogil), stores results.
Shutdown: the reader sends one sentinel per filter worker; the writer stops
after it has seen W sentinels (graceful, no daemon threads, nothing dropped).
A sampler thread records q1/q2 occupancy every 2 ms.
"""
import csv
import os
import queue
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from omp_numba import banner  # noqa: E402

import numpy as np  # noqa: E402
from numba import njit  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
SIDE = 512
PACKETS = 120
QSIZE = 8
SENTINEL = None
KERNEL = (np.outer([1, 4, 6, 4, 1], [1, 4, 6, 4, 1]) / 256.0).astype(np.float64)


@njit(nogil=True)
def blur(src, k, passes):
    h, w = src.shape
    a = src.copy()
    b = np.empty_like(a)
    for _ in range(passes):
        for y in range(h):
            for x in range(w):
                s = 0.0
                for dy in range(-2, 3):
                    yy = min(max(y + dy, 0), h - 1)
                    for dx in range(-2, 3):
                        xx = min(max(x + dx, 0), w - 1)
                        s += k[dy + 2, dx + 2] * a[yy, xx]
                b[y, x] = s
        a, b = b, a
    return a


@njit(nogil=True)
def stats(img):
    mn, mx, tot, chk = img[0, 0], img[0, 0], 0.0, 0.0
    h, w = img.shape
    for y in range(h):
        for x in range(w):
            v = img[y, x]
            mn = min(mn, v)
            mx = max(mx, v)
            tot += v
            chk += v * ((y * w + x) % 7 + 1)
    return mn, mx, tot / (h * w), chk


def run_pipeline(frames, passes, workers):
    q1, q2 = queue.Queue(maxsize=QSIZE), queue.Queue(maxsize=QSIZE)
    busy = {"reader": 0.0, "filter": [0.0] * workers, "writer": 0.0}
    latencies, results, occ = [], [], []
    stop_sampler = threading.Event()

    def reader():
        for i, f in enumerate(frames):
            t0 = time.perf_counter()
            pkt = (i, time.perf_counter(), f.copy())        # ingest from memory
            busy["reader"] += time.perf_counter() - t0
            q1.put(pkt)
        for _ in range(workers):
            q1.put(SENTINEL)

    def filt(wid):
        while True:
            pkt = q1.get()
            if pkt is SENTINEL:
                q2.put(SENTINEL)
                return
            i, t_in, img = pkt
            t0 = time.perf_counter()
            out = blur(img, KERNEL, passes)
            busy["filter"][wid] += time.perf_counter() - t0
            q2.put((i, t_in, out))

    def writer():
        seen = 0
        while seen < workers:
            pkt = q2.get()
            if pkt is SENTINEL:
                seen += 1
                continue
            i, t_in, img = pkt
            t0 = time.perf_counter()
            results.append((i,) + stats(img))
            busy["writer"] += time.perf_counter() - t0
            latencies.append(time.perf_counter() - t_in)

    def sampler():
        while not stop_sampler.is_set():
            occ.append((q1.qsize(), q2.qsize()))
            time.sleep(0.002)

    threads = ([threading.Thread(target=reader)]
               + [threading.Thread(target=filt, args=(w,)) for w in range(workers)]
               + [threading.Thread(target=writer)])
    smp = threading.Thread(target=sampler)
    smp.start()
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    stop_sampler.set()
    smp.join()
    assert len(results) == len(frames)
    return wall, busy, latencies, results, occ


def serial(frames, passes):
    t0 = time.perf_counter()
    res = [(i,) + stats(blur(f.copy(), KERNEL, passes)) for i, f in enumerate(frames)]
    return time.perf_counter() - t0, res


def main():
    banner(f"Lab 6 bonus: 3-stage pipeline, {PACKETS} packets of {SIDE}x{SIDE} float64, "
           f"queue maxsize {QSIZE}")
    rng = np.random.default_rng(230103377)
    pool = [rng.random((SIDE, SIDE)) for _ in range(16)]
    frames = [pool[i % len(pool)] for i in range(PACKETS)]
    blur(pool[0][:16, :16].copy(), KERNEL, 1)
    stats(pool[0])
    rows = []
    for passes in (1, 8, 32):
        t_ser, ref = serial(frames, passes)
        for workers in (1, 4):
            wall, busy, lat, res, occ = run_pipeline(frames, passes, workers)
            assert sorted(res) == sorted(ref), "pipeline result differs from serial"
            occ = np.array(occ)
            fil = sum(busy["filter"])
            util = dict(reader=busy["reader"] / wall,
                        filter=fil / (wall * workers), writer=busy["writer"] / wall)
            bott = max(util, key=util.get)
            row = dict(passes=passes, filter_workers=workers, packets=PACKETS,
                       serial_s=f"{t_ser:.4f}", pipeline_s=f"{wall:.4f}",
                       throughput_pkt_s=f"{PACKETS / wall:.1f}",
                       serial_throughput_pkt_s=f"{PACKETS / t_ser:.1f}",
                       latency_mean_ms=f"{statistics.mean(lat) * 1e3:.2f}",
                       latency_p95_ms=f"{np.percentile(lat, 95) * 1e3:.2f}",
                       q1_mean_occ=f"{occ[:, 0].mean():.2f}", q1_max=int(occ[:, 0].max()),
                       q2_mean_occ=f"{occ[:, 1].mean():.2f}", q2_max=int(occ[:, 1].max()),
                       util_reader=f"{util['reader']:.3f}",
                       util_filter_per_worker=f"{util['filter']:.3f}",
                       util_writer=f"{util['writer']:.3f}", bottleneck=bott)
            rows.append(row)
            print(f"passes={passes:<2} W={workers}: pipeline {wall:.3f} s "
                  f"({PACKETS / wall:6.1f} pkt/s) vs serial {t_ser:.3f} s "
                  f"({PACKETS / t_ser:6.1f} pkt/s) | latency mean {row['latency_mean_ms']} ms "
                  f"p95 {row['latency_p95_ms']} ms | q1 occ mean {row['q1_mean_occ']} "
                  f"max {row['q1_max']} | q2 occ mean {row['q2_mean_occ']} max {row['q2_max']}"
                  f" | busy: reader {util['reader']:.2f} filter {util['filter']:.2f} "
                  f"writer {util['writer']:.2f} -> bottleneck {bott}", flush=True)
    path = os.path.join(RESULTS, "lab6_pipeline.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}; every pipeline run matched the serial results (assert)")


if __name__ == "__main__":
    main()
