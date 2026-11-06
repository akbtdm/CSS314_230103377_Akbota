# Task 19 — Monte Carlo Path Simulation: The Ideal Weak Scaling Model

## 1. Why Monte Carlo is close to ideal
Every rank simulates $n_0$ independent paths and needs no data from other ranks until the very end. Work per rank is constant and communication during the computation is zero, so Gustafson's premise holds almost exactly. But $\alpha$ is not zero.

## 2. Sources of non-zero serial fraction

| Source | What happens | Cost vs. $P$ |
|---|---|---|
| **PRNG seed / stream management** | Rank 0 generates and distributes $P$ independent seeds or streams. Naive: sequential seeding with skip-ahead, $O(P)$ on rank 0. Good: counter-based PRNGs (Philox, Threefry) or `SeedSequence.spawn`, so each rank derives its stream from (seed, rank) with no communication. | $O(P)$ naive, $O(1)$ good |
| **Stream correctness** | Overlapping streams (e.g. `seed = rank` with an LCG) cause correlated paths, so the statistical error does not drop as $1/\sqrt{N}$. This costs accuracy rather than time. | — |
| **Start-up** | Job launch, process spawn, library import, broadcast of parameters. | $O(\log P)$ broadcast, launch often $O(P)$ |
| **Reduction of results** | Combine $P$ partial sums (mean, variance, histograms). | $O(P)$ linear, $O(\log P)$ tree |
| **Output serialization** | Writing paths or statistics through rank 0. | $O(P\cdot\text{output size})$ |
| **Load imbalance** | Variable path lengths (absorbing boundaries, early termination); the slowest rank sets the time. | grows with variance and $\log P$ (expected maximum) |

## 3. Quantifying the reduction

Normalise compute time per rank to $t_c$, and let one reduction step (message latency + addition) cost $t_r$.

**Linear (gather to rank 0):** $T_{red} = (P-1)\,t_r$
$$S_{lin}(P) = \frac{P\,t_c}{t_c + (P-1)t_r} \;\xrightarrow{P\to\infty}\; \frac{t_c}{t_r}$$
This saturates: the linear reduction turns Monte Carlo into an Amdahl-bounded workload.

**Binary tree (MPI_Reduce):** $T_{red} = \lceil\log_2 P\rceil\,t_r$
$$S_{tree}(P) = \frac{P\,t_c}{t_c + \lceil\log_2 P\rceil\,t_r}$$
This grows almost linearly; efficiency is $E = 1/(1 + \log_2 P\cdot t_r/t_c)$.

In Gustafson form, $\alpha_{eff}(P) \approx \dfrac{\log_2 P\,t_r}{t_c + \log_2 P\, t_r}$, which drifts slowly with $P$.

**Example:** $t_c = 1$ s of path simulation per rank, $t_r = 10\,\mu$s.

| $P$ | Linear: $S$ | Linear: $E$ | Tree: $S$ | Tree: $E$ |
|---:|---:|---:|---:|---:|
| 1,024 | 1,013.6 | 98.99% | 1,023.9 | 99.990% |
| 16,384 | 14,077.7 | 85.9% | 16,381.7 | 99.986% |
| 131,072 | 56,723.7 | 43.3% | 131,049.7 | 99.983% |

The tree reduction keeps efficiency above 99.98% at 131K ranks, while the linear gather halves the efficiency. For Monte Carlo the reduction **algorithm**, not the arithmetic, decides whether the code scales.

## 4. Recommendations
1. Use counter-based or spawned PRNG streams; never `seed = rank` with simple generators.
2. Use tree or recursive-doubling reductions (`MPI_Reduce`/`MPI_Allreduce`); reduce once at the end, not per path.
3. Reduce sufficient statistics (count, sum, sum of squares, or Welford/Chan pairwise merges) instead of raw samples.
4. Write output in parallel (MPI-IO, per-rank files) or aggregate only summaries.
5. Use dynamic work distribution (work stealing, small batches) when path cost varies.

Our own benchmark (Task 22) uses numerical integration with the same structure: independent slices plus one final reduction. Its measured $\alpha$ is dominated by process start-up, not by arithmetic.
