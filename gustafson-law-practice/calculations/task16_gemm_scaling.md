# Task 16 — Dense Matrix Multiplication Scaling (GEMM)

## Model
* Work: $W(N) = 2N^3$ flops (one multiply-add per $(i,j,k)$).
* Memory: $M(N) = 3N^2$ elements ($A$, $B$, $C$).
* Gustafson: parallel time stays constant on $P$ processors, each processor with rate $r$.

## Proof: $N$ must scale as $P^{1/3}$

Constant time means constant work per processor:
$$T_P = \frac{2N_P^3}{P\,r} = T_1 = \frac{2N_1^3}{r}$$
$$\Rightarrow\; N_P^3 = P\,N_1^3 \;\Rightarrow\; N_P = N_1\,P^{1/3} \qquad \blacksquare$$

(Ignoring the serial fraction, which only lowers the exponent slightly.)

## Consequences

* **Memory per processor shrinks:**
$$\frac{M(N_P)}{P} = \frac{3N_1^2 P^{2/3}}{P} = 3N_1^2\,P^{-1/3}$$
Under fixed-time scaling GEMM uses *less* memory per node as $P$ grows; memory is not the binding constraint.
* **If instead we fill memory** (Sun–Ni memory-bounded scaling, Task 11): $N^2 \propto P$, so $N \propto P^{1/2}$ and work $\propto P^{3/2}$. Time per processor then grows as $P^{1/2}$, which violates fixed time.
* **Communication per processor** (SUMMA/Cannon on a $\sqrt{P}\times\sqrt{P}$ grid): $O(N^2/\sqrt{P}) = O(N_1^2 P^{2/3}/P^{1/2}) = O(N_1^2 P^{1/6})$. It grows slowly with $P$, while computation per processor stays constant, so efficiency decays gently.

## Scaling targets (baseline $N_1 = 4096$ on one processor)

| $P$ | $P^{1/3}$ | Fixed-time $N_P = 4096\,P^{1/3}$ | Memory-bounded $N = 4096\,P^{1/2}$ | Memory/proc (fixed-time), rel. to $P=1$ |
|---:|---:|---:|---:|---:|
| 1 | 1 | 4,096 | 4,096 | 1.000 |
| 8 | 2 | 8,192 | 11,585 | 0.500 |
| 64 | 4 | 16,384 | 32,768 | 0.250 |
| 512 | 8 | 32,768 | 92,682 | 0.125 |
| 4,096 | 16 | 65,536 | 262,144 | 0.0625 |
| 32,768 | 32 | 131,072 | 741,455 | 0.03125 |

At $P = 1$: $2\cdot 4096^3 = 1.37\times10^{11}$ flops and $3\cdot4096^2\cdot 8$ B $= 402$ MB in double precision.

**Takeaway:** for an $O(N^3)$ algorithm, 1000× more processors buy only a 10× larger matrix at constant time. "Scaling the problem with $P$" means scaling the **work** linearly; the problem **dimension** grows only with the cube root.
