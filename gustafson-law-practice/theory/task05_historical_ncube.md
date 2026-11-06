# Task 05 — Deconstructing the "Amdahl's Pessimism" Fallacy: the nCUBE 10 Case

## 1. Historical setting

By the mid-1980s, Amdahl's Law was widely cited as proof that massive parallelism could not work: with even 1% serial code, 1024 processors could deliver at most about 91×. In 1985 Alan Karp offered a prize for a 200× speedup on a real application, and the IEEE Gordon Bell Prize was created for practical parallel speedup.

In 1988 **John Gustafson, Gary Montry and Robert Benner** (Sandia National Laboratories) reported results on an **nCUBE 10 hypercube with 1024 processors**. Each node had 512 KB of private memory, roughly 80 KFLOPS of performance, and direct links to 10 neighbours. The work was published in *SIAM J. Sci. Stat. Comput.* 9(4), 1988, with the short companion note "Reevaluating Amdahl's Law" in *CACM* 31(5), 1988. It won the first Gordon Bell Prize and the Karp challenge.

## 2. The three applications

| Application | Method | Fixed-size speedup (1024 PE) | Scaled speedup (1024 PE) |
|---|---|---|---|
| Beam stress analysis | Conjugate gradients | 502 | 1021 |
| Baffled surface-wave simulation | Explicit finite differences | 637 | 1020 |
| Unstable fluid flow | Flux-corrected transport | 519 | 1016 |

Two measurement lines were used:
* **Research line (fixed size):** a problem small enough to run on one node, spread over 1024 nodes. The serial component was $s \approx 0.0006$–$0.001$, giving 502–637×.
* **Application line (scaled size):** each node keeps the same subproblem size, so the global problem grows 1024×. The equivalent serial component dropped to about 3–10 parts per million, giving 1016–1021×.

**Evidence of constant time:** for the wave simulation, 4096 grid points on one node took 12,780 s, while 4096 points *per node* on 1024 nodes (about 4 million points) took 12,824 s, only 0.3% longer for 1024× more work. For fluid flow, a 32×32 grid on one node took 28,039 s versus 28,258 s for 32×32 per node on 1024 nodes.

## 3. Why Amdahl "forbade" this

Applying Amdahl to the serial fraction quoted in the CACM note ($s = 0.4\%$–$0.8\%$, measured on the parallel runs):
$$S_A(1024) = \frac{1}{0.004 + 0.996/1024} \approx 201, \qquad \frac{1}{0.008 + 0.992/1024} \approx 111$$
with an absolute ceiling of $1/s = 125$–$250$. The measured 1016–1021× seemed impossible.

## 4. How problem expansion dismantled the ceiling

1. **The serial fraction was measured in the wrong frame.** 0.4%–0.8% is the parallel-machine $\alpha$. Gustafson's formula with that frame gives
$$S_G = 1024 - 0.004\cdot 1023 \approx 1020, \qquad 1024 - 0.008 \cdot 1023 \approx 1016$$
which matches the measurements almost exactly.
2. **Serial work did not grow with the problem.** Host I/O, setup and global broadcasts stayed roughly constant in absolute time, while the parallel part grew 1024×. The paper reports that the parallel part scaled by factors of about 1023.997.
3. **Communication was local.** Each node exchanged only boundary data with hypercube neighbours. The surface-to-volume ratio per node was constant under scaling (see Task 17), so communication did not erode efficiency.
4. **Engineering reduced the residual serial part:** overlapping communication with computation, fewer message start-ups, balanced loads, and some redundant computation where it was cheaper than communication. The authors introduced *operation efficiency* to measure this trade-off.
5. **Consistency check on the fixed-size line:** Amdahl with the measured fixed-size $s = 0.0006$–$0.001$ predicts $1/(s + (1-s)/1024) \approx 634$–$506$, consistent with the measured 637 and 502. Amdahl was never wrong; it was answering a different question.

## 5. Lessons

* "Amdahl's pessimism" was a **misuse of a correct formula**: a fixed-size model applied to machines that are actually used to run bigger problems.
* The experiment also showed the cost of scaling: the 1024-node problem needs 1024× the memory (512 MB in total). Speedup is ultimately bounded by memory and communication, not by the serial fraction alone (Sun–Ni, Task 11).

## Sources
* J. L. Gustafson, G. R. Montry, R. E. Benner, "Development of Parallel Methods for a 1024-Processor Hypercube," *SIAM J. Sci. Stat. Comput.* 9(4):609–638, 1988. https://doi.org/10.1137/0909041
* J. L. Gustafson, "Reevaluating Amdahl's Law," *Communications of the ACM* 31(5):532–533, 1988. https://users.cs.utah.edu/~hari/teaching/paralg/Gustafson88.pdf
* J. Dongarra et al., Gordon Bell Prize 1987 summary (netlib). https://netlib.org/benchmark/bell1
