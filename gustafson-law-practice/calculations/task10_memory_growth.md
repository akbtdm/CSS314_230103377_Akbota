# Task 10 — Memory Growth Rate vs. Scaled Speedup

## Formulation

* Weak scaling: $W(P) = P\cdot W(1)$, memory $M(P) = c\cdot P$.
* Per-node cap: $m = 32$ GiB $= 32\cdot 2^{30} = 2^{35}$ bytes.
* Aggregate capacity on $n$ nodes: $M_{max}(n) = n\cdot m$.
* 3D grid of $N\times N\times N$ points, $b$ bytes per point: $M = b\,N^3$.

Constraint:
$$b\,N^3 \le n\,m \quad\Rightarrow\quad N_{max} = \left\lfloor \left(\frac{n\,m}{b}\right)^{1/3}\right\rfloor$$

So $N_{max} \propto n^{1/3}$: 8× more nodes allow only a 2× larger grid edge.

## Capacities

| Nodes $n$ | Aggregate memory |
|---|---|
| 64 | $64 \cdot 32 = 2048$ GiB $= 2^{41}$ B |
| 512 | $512\cdot 32 = 16384$ GiB $= 2^{44}$ B |

## Maximum grid dimension

**Base case, one double per point ($b = 8$ B, $2^3$):**
* 64 nodes: $N^3 \le 2^{41}/2^3 = 2^{38}$, so $N \le 2^{38/3} = 6501.9$, giving **$N_{max} = 6501$** (≈ $2.75\times 10^{11}$ points).
* 512 nodes: $N^3 \le 2^{44}/2^3 = 2^{41}$, so $N \le 2^{41/3} = 13003.8$, giving **$N_{max} = 13003$** (≈ $2.2\times10^{12}$ points).

**Realistic solvers store several fields per point:**

| Bytes per point $b$ | Example | $N_{max}$, 64 nodes | $N_{max}$, 512 nodes |
|---:|---|---:|---:|
| 8 | 1 double field | 6,501 | 13,003 |
| 40 | 5 doubles (ρ, u, v, w, E) | 3,802 | 7,604 |
| 64 | 8 doubles (fields + workspace) | 3,250 | 6,501 |

(Using 100% of memory; leaving ~20% for the OS and buffers reduces $N$ by a factor of $0.8^{1/3} = 0.928$.)

## Link to scaled speedup

* From 64 → 512 nodes (8×), memory and $N^3$ grow 8×, but the edge $N$ grows only 2×.
* If the solver cost is $O(N^3)$ per time step, work grows 8× (matches $P$), and Gustafson's constant time holds per step.
* If the number of time steps must also grow with resolution (CFL condition: $\Delta t \propto 1/N$), total work grows as $N^4$, i.e. $8^{4/3} = 16\times$ for 8× nodes. **Run time doubles**, so memory-filling scaling breaks Gustafson's fixed-time assumption (see Tasks 11 and 13).
