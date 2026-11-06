# Task 18 — N-Body Gravitational Problem Scaling

## Setup
Requirement: constant total time per iteration as $P$ grows, so **work per processor per step must stay constant**.

## 1. Direct all-pairs method, $O(N^2)$

**Problem growth:**
$$\frac{N^2}{P} = \text{const} \;\Rightarrow\; N \propto P^{1/2}$$
1000× more processors allow only about 32× more particles.

**Per-rank costs** (particle decomposition, $N/P$ particles per rank, ring or all-gather of positions):
| Component | Cost per rank per step |
|---|---|
| Force computation | $(N/P)\cdot N = N^2/P$, constant |
| Communication: every rank needs all $N$ positions | $O(N) = O(\sqrt{P})$ data, plus $O(\log P)$ or $O(P)$ messages |
| Serial part (time-step control, energy reduction) | $O(\log P)$ |

$$\frac{T_{comm}}{T_{comp}} \propto \frac{N}{N^2/P} = \frac{P}{N} \propto \sqrt{P}$$

**Serial/parallel ratio:** the non-scalable fraction **grows as $\sqrt P$**, because a constant amount of computation per rank must be fed with a growing ($\propto N$) amount of data. Gustafson's $\alpha$ is not invariant. (2D/"$N$-body force decomposition" reduces data per rank to $O(N/\sqrt{P}) = O(1)$, which restores weak scaling at the cost of extra reductions.)

## 2. Barnes–Hut tree code, $O(N\log N)$

**Problem growth:**
$$\frac{N\log N}{P} = \text{const} \;\Rightarrow\; N \approx \frac{P}{\log P}\cdot\text{const}$$
i.e. almost linear: with $10^6$ particles per processor as the baseline, 1000× more processors allow about 679× more particles.

**Per-rank costs** (spatial decomposition, $N/P$ particles per rank):
| Component | Cost per rank per step |
|---|---|
| Force computation (tree walk) | $(N/P)\log N$, constant by construction |
| Locally essential tree exchange (boundary cells) | $\approx (N/P)^{2/3}\log N$, surface-like, nearly constant |
| Global tree top / domain decomposition (Morton keys, sort, load balancing) | $O(\log P)$ collectives plus a top tree of $O(P)$ nodes replicated on each rank |
| Serial part | Building/broadcasting the top-level tree: grows slowly with $P$ |

$$\frac{T_{comm}}{T_{comp}} \propto \frac{(N/P)^{2/3}\log N}{(N/P)\log N} = \left(\frac{N}{P}\right)^{-1/3} \approx \text{const}$$
(because $N/P$ is roughly constant).

**Serial/parallel ratio:** roughly constant, with weak growth from the $O(\log P)$ global steps and the $O(P)$ replicated top tree.

## 3. Comparison

| | Direct $O(N^2)$ | Barnes–Hut $O(N\log N)$ |
|---|---|---|
| $N$ vs. $P$ at constant time | $N\propto P^{1/2}$ | $N\propto P/\log P$ |
| Comm/comp ratio vs. $P$ | $\propto \sqrt{P}$ (grows) | $\approx$ const |
| Effective $\alpha(P)$ | increases | nearly invariant, $O(\log P)$ drift |
| Fit to Gustafson | poor | good |
| Main bottleneck | all-to-all data volume | load imbalance from clustering, tree build |

## 4. Insight
The asymptotic complexity of the algorithm determines how "weak" weak scaling can be. A more efficient algorithm (Barnes–Hut) is also the more *scalable* one: its work and its communication have the same surface/volume structure as a stencil. The direct method has computational intensity that grows with $N$, but needs global data, so its serial/parallel ratio drifts by $\sqrt{P}$.

Barnes–Hut is harder to balance: gravitational clustering makes the work per particle uneven over time, so dynamic load balancing (cost zones, ORB) is required. That rebalancing is an additional, growing serial cost.
