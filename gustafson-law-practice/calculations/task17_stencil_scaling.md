# Task 17 — 3D Stencil Hydrodynamics Simulation

## Model
* Global domain $N\times N\times N$, decomposed into $P$ cubic subdomains of edge $n = N/P^{1/3}$.
* 7-point stencil: about $c_1 n^3$ operations per step per rank (interior update).
* Halo exchange: 6 faces of $n^2$ points, so $c_2 \cdot 6n^2$ data per rank per step.

Communication-to-computation ratio per rank:
$$R = \frac{T_{comm}}{T_{comp}} = \frac{6 c_2 n^2}{c_1 n^3} = \frac{6c_2}{c_1}\cdot\frac{1}{n}$$

This is the surface-to-volume ratio of one subdomain.

## Weak scaling (Gustafson): domain volume grows with P

Keep the per-rank subdomain fixed: $n = n_0$, so the global $N^3 = P n_0^3$, i.e. $N = n_0 P^{1/3}$.
$$R_{weak}(P) = \frac{6c_2}{c_1 n_0} = \text{constant}$$

**Proof:** $R$ depends only on $n$, and $n = n_0$ does not depend on $P$. $\blacksquare$

* Per rank, computation and communication are both constant, so step time is constant. This is exactly Gustafson's premise.
* Globally, total communication $6Pn_0^2$ and total work $Pn_0^3$ both grow linearly in $P$, so the global ratio is also constant.
* Example: $n_0 = 128$ gives $6/128 = 4.7\%$ halo points per interior point, for any $P$.

## Strong scaling (contrast): fixed N
$$n = \frac{N}{P^{1/3}} \;\Rightarrow\; R_{strong}(P) = \frac{6c_2}{c_1}\cdot\frac{P^{1/3}}{N}$$
The ratio **degrades** as $P^{1/3}$: 8× more ranks double the relative communication.

| $P$ | $R_{strong}$ with $N=1024$ (×$6c_2/c_1$) | $R_{weak}$ with $n_0 = 128$ (×$6c_2/c_1$) |
|---:|---:|---:|
| 8 | 1/512 = 0.00195 | 1/128 = 0.0078 |
| 512 | 1/128 = 0.0078 | 0.0078 |
| 32,768 | 1/32 = 0.0313 | 0.0078 |

## Gustafson's law in this setting

The serial-like fraction on the parallel machine is $\alpha \approx R/(1+R)$ (plus global I/O). Under weak scaling $R$ is constant, so $\alpha$ is constant and $S(P) = P - \alpha(P-1)$ applies directly. This is why stencil codes were the showcase of the nCUBE study (Task 05).

## What still degrades
* **Latency terms:** each halo message also costs a start-up latency, constant per rank in weak scaling, so no growth.
* **Global reductions** (time-step control, convergence checks) cost $O(\log P)$ and slowly increase $\alpha$ (Task 12).
* **Network contention / topology:** on a torus or fat-tree, nearest-neighbour traffic is mostly scale-free; random rank placement breaks this.
