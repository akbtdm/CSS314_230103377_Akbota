# Task 20 — Parallel Graph Traversal & Irregular Workloads

## 1. Why dense linear algebra fits Gustafson

Dense GEMM or a stencil has the properties weak scaling needs:
* **Predictable work:** each rank's block has the same flop count; static partitioning gives perfect balance.
* **High arithmetic intensity:** $O(n^3)$ flops on $O(n^2)$ data, so caches and vector units are used efficiently.
* **Regular, local communication:** fixed messages to known neighbours, computable at compile time; surface-to-volume ratio is constant under weak scaling (Task 17).

## 2. Why BFS and PageRank on power-law graphs do not

### (a) Irregular memory access
* Each edge visit is an indirect load `dist[adj[e]]` to an essentially random address. Spatial and temporal locality are poor, so most accesses miss cache and wait on DRAM latency (~100 ns).
* Arithmetic intensity is about 1 operation per 8–16 bytes. Performance is set by **memory latency and bandwidth**, not by cores. Adding processes on one node quickly saturates the memory system (the same effect as in our Karp–Flatt study), so per-rank work cannot grow at constant time.

### (b) Power-law degree distribution → load imbalance
* A few hub vertices have degrees in the millions, while most have degree 1–10. With a 1D vertex partition, the rank that owns a hub processes far more edges.
* Imbalance grows with graph size: in a scale-free graph the maximum degree grows like $N^{1/(\gamma-1)}$, so weak-scaling the graph makes the worst rank **slower**. $T_P$ is set by the maximum, not the average.
* Mitigations: 2D edge partitioning, vertex delegates for hubs, degree-aware partitioning, all of which add communication.

### (c) Frontier imbalance (BFS)
* BFS proceeds level by level. Frontier size varies enormously: tiny at the start, huge in the middle (small-world graphs reach most vertices in 4–6 levels), tiny at the end.
* In small levels most ranks are idle, but all must still synchronise. These levels behave like **serial phases** whose count does not shrink when the graph grows.
* Direction-optimising BFS (top-down/bottom-up switching) helps, but makes the work data-dependent.

### (d) Data-dependent, global synchronisation
* Each BFS level ends with a global barrier/allreduce ("is the frontier empty?"), costing $O(\log P)$ per level.
* Edges cross partitions randomly: in a random graph a fraction $\approx (P-1)/P$ of edges are **remote**. Communication becomes **all-to-all** with volume proportional to the number of edges, not a surface. The surface-to-volume argument that saves stencils does not exist for graphs with no geometric embedding.
* PageRank needs one global exchange of all rank contributions per iteration, and the number of iterations depends on convergence, not on $P$.

## 3. Consequence in Gustafson terms

| Property | Dense LA / stencil | BFS / PageRank |
|---|---|---|
| Work per rank at fixed share of $N$ | constant | varies (hubs, frontier) |
| Comm per rank | surface, constant | all-to-all, ∝ edges/rank × $(P-1)/P$, plus $O(P)$ messages |
| Serial-like phases | few | one sync per level/iteration |
| Bound | compute | memory latency + network |
| Effective $\alpha(P)$ | ≈ constant | grows with $P$ and graph size |

Graph500 results show this: even the best supercomputers reach only a small fraction of their peak compute in TEPS (traversed edges per second), and efficiency per node drops as node count grows.

## 4. Conclusion
Gustafson's law assumes the extra work from a bigger problem is "more of the same, local" work. In graph analytics a bigger graph means more global connectivity, worse imbalance and more remote memory accesses. The serial/overhead fraction grows with scale, so weak scaling falls well below $P - \alpha(P-1)$.
