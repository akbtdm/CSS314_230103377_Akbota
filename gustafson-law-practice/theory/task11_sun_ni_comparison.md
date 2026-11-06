# Task 11 — Sun–Ni Memory-Bounded Speedup vs. Gustafson's Law

## 1. Three models, one family

Let $f$ be the serial fraction of the workload and let $G(P)$ be the factor by which the **parallel workload** grows when the machine grows from 1 to $P$ processors. Sun and Ni (1990, 1993) define memory-bounded speedup as

$$S_{SN}(P) = \frac{f + (1-f)\,G(P)}{f + \dfrac{(1-f)\,G(P)}{P}}$$

where $G(P)$ is determined by **how much work fits in $P$ times the memory**: if the algorithm needs memory $M(N)$ and work $W(N)$, then $G(P) = W(M^{-1}(P\cdot M_1))/W_1$.

| $G(P)$ | Resulting model | Constraint |
|---|---|---|
| $G = 1$ | $S = \dfrac{1}{f + (1-f)/P}$, **Amdahl** | fixed size |
| $G = P$ | $S = f + (1-f)P$, **Gustafson** | fixed time |
| $G(P)$ from memory | **Sun–Ni** | fixed memory per node |

**Proof of the Gustafson case:** with $G=P$ the denominator is $f + (1-f) = 1$, so $S = f + (1-f)P$. $\blacksquare$

## 2. Comparison of assumptions

| | Gustafson | Sun–Ni |
|---|---|---|
| What limits the problem | User's patience (time) | Memory capacity |
| Time on $P$ processors | Constant by definition | Allowed to grow |
| Work growth | Linear in $P$ | Whatever memory permits: $G(P)$ |
| Measures | Throughput at fixed time | Largest problem the machine can hold |

## 3. Worked algorithm classes

| Algorithm | $W(N)$ | $M(N)$ | $G(P)$ under memory bound | vs. $P$ |
|---|---|---|---|---|
| Stencil / FFT-like | $N^d$ | $N^d$ | $P$ | equal: Sun–Ni = Gustafson |
| Dense GEMM, LU | $N^3$ | $N^2$ | $P^{3/2}$ | $G > P$ |
| Direct N-body | $N^2$ | $N$ | $P^2$ | $G > P$ |
| Sorting | $N\log N$ | $N$ | $\approx P\log P$ | slightly $> P$ |
| Replicated-data codes | $N$ | $N\cdot P$ (each node stores everything) | $< P$ | $G < P$ |

Example, GEMM with $f = 0.01$, $P=1024$: $G = 1024^{1.5} = 32768$.
$$S_{SN} = \frac{0.01 + 0.99\cdot 32768}{0.01 + 0.99\cdot 32} = \frac{32440.3}{31.69} \approx 1023.7$$
Gustafson gives $1024 - 0.01\cdot1023 = 1013.8$. But Sun–Ni's run takes $31.69\times$ longer than the single-processor run, so it is not a fixed-time run.

## 4. When Gustafson overestimates

1. **$G(P) < P$:** memory runs out before time does (replicated data, ghost zones dominating small subdomains, per-process runtime overhead such as Python interpreters). The problem cannot grow $P$-fold, so the linear speedup is unreachable.
2. **Serial part grows with the problem:** sequential I/O of a $P$-times larger output, a global gather or a serial preprocessing step. Then $\alpha$ is not invariant.
3. **Communication grows with $P$:** collectives at $O(\log P)$, all-to-all at $O(P)$. This is not part of either model (Task 12).
4. **Superlinear work:** if $W \propto N^3$, scaling $N$ "with $P$" in a physical sense (finer grid) grows the work faster than $P$, so runtime grows.

## 5. When Gustafson underestimates

1. **$G(P) > P$ and the user accepts longer runs:** memory-bounded scaling of GEMM or N-body produces far more total work (more science per run) than the fixed-time model counts. Gustafson's fixed-time $S$ understates the **throughput of the capability run**.
2. **Cache/memory effects:** as per-node data shrinks (GEMM under fixed time, Task 16) it may fit in cache, producing superlinear effects that a linear law cannot express.

## 6. Conclusion
Sun–Ni is the general case. Amdahl and Gustafson are its two boundary instances ($G=1$ and $G=P$). For any algorithm, the right question is how its work grows with its memory. Gustafson is exact only when $W \propto M$, as in stencils, explicit PDE solvers and Monte Carlo methods.

## Reference
X.-H. Sun, L. M. Ni, "Scalable Problems and Memory-Bounded Speedup," *J. Parallel and Distributed Computing* 19(1):27–37, 1993.
