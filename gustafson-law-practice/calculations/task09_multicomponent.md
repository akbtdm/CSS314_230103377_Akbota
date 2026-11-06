# Task 09 — Multi-Component Workload Partitioning

## Phase breakdown (fractions of total parallel execution time)

| Phase | Share | Classification | Reason |
|---|---|---|---|
| A — Disk I/O + parameter broadcast | 2% | Serial | One reader (rank 0) reads input and broadcasts; other ranks wait. |
| B — Boundary message-passing synchronization | 6% | Serial (non-scalable overhead) | Communication and barriers do no useful work that grows with $P$; time spent here is not multiplied by $P$ in the equivalent serial run. |
| C — Core domain flux computation | 92% | Parallel | Every rank updates its own subdomain. |

$$\alpha = 0.02 + 0.06 = 0.08, \qquad 1-\alpha = 0.92$$

*Why B counts as serial:* in Gustafson's derivation, a phase is "parallel" only if a uniprocessor would need $P$ times as long for it. A uniprocessor has no message passing, so phase B contributes nothing to $T_1$. It behaves like serial time: it is paid on the parallel machine and produces no scaled work.

## Scaled speedup

$$S(P) = P - \alpha(P-1)$$

**P = 512:** $S = 512 - 0.08\cdot 511 = 512 - 40.88 = 471.12$, so $E = 92.02\%$.

**P = 2048:** $S = 2048 - 0.08\cdot 2047 = 2048 - 163.76 = 1884.24$, so $E = 92.004\%$.

## Composite analysis

| | P = 512 | P = 2048 |
|---|---|---|
| Gustafson $S$ | 471.12 | 1,884.24 |
| Lost to I/O (A): $0.02(P-1)$ | 10.22 | 40.94 |
| Lost to sync (B): $0.06(P-1)$ | 30.66 | 122.82 |
| Amdahl with $s = 0.08$ (for contrast) | 12.23 | 12.43 |

* Phase B causes 75% of the lost speedup; it is the first optimisation target (overlap communication with computation, fewer barriers).
* Removing phase B entirely ($\alpha = 0.02$) would give $S(2048) = 2007.06$, i.e. +122.8 effective cores.
* **Realism check:** the model assumes the 6% stays constant as $P$ grows. Boundary exchange per rank is constant under weak scaling, but synchronization (barriers, allreduce) grows as $O(\log P)$. At 2048 ranks the real B share is likely larger than 6%, so 1884× is an upper bound (see Task 12).
