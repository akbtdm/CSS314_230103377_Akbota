# Task 12 — Communication Overhead Penalty Integration

## 1. Problem with the classic law
Classic Gustafson assumes the parallel run time $s + p = 1$ for **every** $P$. In distributed systems an extra term appears whose cost grows with $P$: collectives (allreduce, barrier) at $O(\log P)$, and 2D-mesh or row/column exchanges at $O(\sqrt{P})$.

## 2. Model extension

Let the run on $P$ processors take
$$T_P = s + p + C(P), \qquad s + p = 1,\quad C(1) = 0$$
with the scaled serial run unchanged, since communication does not exist on one processor:
$$T_1 = s + P\,p$$

**Modified scaled speedup:**
$$\boxed{S_C(P) = \frac{s + P\,p}{1 + C(P)} = \frac{P - \alpha(P-1)}{1 + C(P)}}$$

**Efficiency:**
$$E_C(P) = \frac{1 - \alpha + \alpha/P}{1 + C(P)}$$

### Communication models
| Pattern | $C(P)$ | Typical source |
|---|---|---|
| Tree collective | $c\,\log_2 P$ | allreduce of residuals, dot products in CG |
| 2D mesh / transpose | $c\,(\sqrt{P} - 1)$ | row/column broadcasts, SUMMA |
| Linear (naive gather) | $c\,(P-1)$ | rank 0 collecting results |
| Nearest neighbour (weak scaling) | $c$ (constant) | halo exchange (Task 17); absorbed into $\alpha$ |

Here $c$ is the cost of one communication stage relative to the normalised run time (e.g. $c = t_{msg}/T_{P}$).

## 3. Derived properties

1. **Reduces to Gustafson** when $c = 0$.
2. **Asymptotics:**
   * $C = c\log_2 P$: $S_C \sim \dfrac{(1-\alpha)P}{c\log_2 P} \to \infty$, but efficiency $E_C \to 0$ like $1/\log P$.
   * $C = c\sqrt{P}$: $S_C \sim \dfrac{(1-\alpha)}{c}\sqrt{P}$. Speedup grows only as $\sqrt P$.
   * $C = cP$: $S_C \to \dfrac{1-\alpha}{c}$, a hard ceiling similar to Amdahl's.
3. **Effective serial fraction.** Writing $S_C = P - \alpha_{eff}(P-1)$ and solving:
$$\alpha_{eff}(P) = \frac{P - S_C(P)}{P-1}$$
It grows with $P$, which is the same signature that an **increasing Karp–Flatt metric** shows in measurements.

## 4. Numerical illustration ($\alpha = 0.04$, $c = 0.005$)

| $P$ | Gustafson | $\log_2 P$ model | $\sqrt P$ model |
|---:|---:|---:|---:|
| 16 | 15.40 | 15.10 (E 94.4%) | 15.17 |
| 256 | 245.80 | 236.35 (E 92.3%) | 228.65 |
| 1,024 | 983.08 | 936.27 (E 91.4%) | 851.15 |
| 16,384 | 15,728.68 | 14,699.70 (E 89.7%) | 9,619.99 |

Here $C_{log} = 0.005\log_2 P$ and $C_{sqrt} = 0.005(\sqrt P - 1)$. (Reproduce with `python3 benchmarks/simulator.py --comm-model log --comm-c 0.005` and `--comm-model sqrt`.)

## 5. Design implications
* Prefer algorithms whose per-rank communication is constant under weak scaling (nearest neighbour).
* Hide $O(\log P)$ collectives by overlapping them with computation (non-blocking MPI_Iallreduce) or reduce their frequency (communication-avoiding Krylov methods).
* Avoid any $O(P)$ pattern at rank 0; it turns weak scaling back into Amdahl-type saturation.
