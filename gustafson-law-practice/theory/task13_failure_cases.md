# Task 13 — Boundary Case: When Gustafson Fails

Gustafson's premise: if the problem grows with $P$, the parallel run time stays constant. That needs (i) work growing linearly with problem size, (ii) serial time independent of problem size, and (iii) per-rank communication independent of $P$. The two algorithms below break these conditions.

---

## Failure case 1 — Iterative Krylov / relaxation solvers for elliptic PDEs (CG, Jacobi on a Poisson problem)

**Setting.** Solve $\nabla^2 u = f$ on a 3D grid with $N^3$ points using Conjugate Gradient, $n_0^3$ points per rank, $N = n_0 P^{1/3}$.

**Breakdown.**
1. **Iteration count grows with problem size.** The condition number of the discrete Laplacian is $\kappa = O(h^{-2}) = O(N^2)$. CG needs $O(\sqrt{\kappa}) = O(N)$ iterations, and Jacobi needs $O(N^2)$.
$$\text{iterations}(P) \propto N = n_0 P^{1/3}$$
Work per iteration per rank is constant (local SpMV plus halo exchange), but the **number of iterations** grows, so
$$T_P \propto P^{1/3}\quad(\text{CG}), \qquad T_P \propto P^{2/3}\quad(\text{Jacobi})$$
Going 1 → 4096 ranks makes CG 16× slower and Jacobi 256× slower at constant per-rank size.
2. **Global synchronisation every iteration.** Each CG iteration contains 2 dot products, each an allreduce costing $O(\log P)$ latency. At scale these are latency-bound, and they multiply with the growing iteration count:
$$T_{sync} \propto P^{1/3}\log P$$
3. The equivalent serial work grows faster than $P$ (total work $\propto N^3\cdot N = N^4 \propto P^{4/3}$), so the constant-time premise fails.

**Mitigation:** multigrid preconditioning restores $O(1)$ iterations, but the coarse grid levels are almost serial and add $O(\log P)$ or more. Pipelined/communication-avoiding CG hides the allreduce latency.

---

## Failure case 2 — Parallel sample sort / all-to-all data redistribution (also distributed hash join, 3D FFT transpose)

**Setting.** Each rank holds $n_0$ keys, total $N = P n_0$. Sample sort: local sort, sample selection, splitter broadcast, **all-to-all exchange**, local merge.

**Breakdown.**
1. **Work is superlinear.** Comparison sorting is $O(N\log N)$. Per rank: $O(n_0\log(Pn_0))$, growing with $\log P$ even at fixed $n_0$.
2. **Splitter selection is serial and grows with $P$.** Rank 0 sorts $P\cdot(P-1)$ samples (oversampling $\sim P$ per rank): $O(P^2\log P)$ serial work, then broadcasts $P-1$ splitters. At $P = 4096$ that is about 16.7M samples on a single node, a serial fraction that **grows** with $P$, the exact opposite of Gustafson's assumption.
3. **All-to-all communication.** Each rank sends a message to every other rank: $P-1$ messages per rank, so the latency term is $O(P)$, and total network traffic $\approx N$ crosses the bisection. The bisection bandwidth of real networks grows more slowly than $P$ (e.g. $P^{2/3}$ on a 3D torus), so
$$T_{comm} \propto \frac{N}{\text{bisection}} \propto \frac{P n_0}{P^{2/3}} = n_0 P^{1/3}$$
4. **Data-dependent load imbalance.** Skewed key distributions send more keys to some ranks, and the slowest rank sets the time.

**Result:** $T_P$ grows like $O(\log P + P\cdot t_{latency} + P^{1/3})$; constant time is impossible.

---

## Common pattern

| Gustafson requirement | CG / Jacobi | Sample sort / all-to-all |
|---|---|---|
| Work ∝ problem size | ✗ iterations grow with $N$ | ✗ $N\log N$ |
| Serial time constant | ~ (coarse grid in MG) | ✗ splitters $O(P^2)$ |
| Per-rank comm constant | ✗ allreduce $O(\log P)$ each iteration | ✗ all-to-all $O(P)$ |

The constant-runtime premise breaks whenever growing the problem changes the algorithm's convergence behaviour, its asymptotic complexity, or turns local communication into global communication. In these cases the **isoefficiency function** (Task 14) is superlinear, and Gustafson's line becomes an upper bound rather than a prediction.
