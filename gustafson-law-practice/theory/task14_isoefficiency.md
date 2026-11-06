# Task 14 — Isoefficiency Relation with Gustafson Scaling

## 1. Isoefficiency (Grama, Gupta, Kumar)

Let $W$ be the problem size measured as serial work, $T_P$ the parallel time, and the **total overhead**
$$T_o(W,P) = P\,T_P - W$$
(all processor-time not spent on useful work: idle time in serial phases, communication, redundancy). Then
$$E = \frac{W}{P\,T_P} = \frac{W}{W + T_o} = \frac{1}{1 + T_o/W}$$
To keep $E$ constant:
$$W = \frac{E}{1-E}\,T_o(W,P) = K\cdot T_o(W,P)$$
The solution $W = f_E(P)$ is the **isoefficiency function**: how fast work must grow with $P$ to keep efficiency fixed. Smaller growth means more scalable.

## 2. Overhead of a Gustafson-compliant program

Use Gustafson's model: on $P$ processors the run has serial time $a$ (constant) and parallel time $b$ with $a+b = T_P$.

* Useful work: $W = a + P\,b$.
* Processor-time paid: $P\,T_P = P(a+b) = Pa + Pb$.
* Overhead: $T_o = Pa + Pb - (a + Pb) = a(P-1)$.

So **the overhead of a Gustafson program is exactly the idle time of $P-1$ processors during the serial phase**, and it does not depend on $W$.

## 3. Isoefficiency function

$$W = K\,T_o = K\,a\,(P-1) \;\Rightarrow\; \boxed{f_E(P) = \Theta(P)}$$

**Proof that weak scaling achieves it.** Gustafson's scaling sets $W(P) = a + Pb$ with $a, b$ constant, so $W = \Theta(P)$. Substituting:
$$E = \frac{a + Pb}{P(a+b)} = 1 - \alpha\frac{P-1}{P}, \qquad \alpha = \frac{a}{a+b}$$
which converges to $1-\alpha$, a constant (Task 04). Hence linear work growth is sufficient to hold efficiency constant. It is also necessary up to constants: with $W = o(P)$ the ratio $T_o/W = a(P-1)/W \to \infty$ and $E\to 0$. $\blacksquare$

**$\Theta(P)$ is the best possible isoefficiency.** Any parallel program needs at least $\Omega(P)$ work to keep $P$ processors busy (each must do at least one unit). A Gustafson-compliant program is therefore **ideally scalable** in the isoefficiency sense.

## 4. Critique: what the result hides

1. **The overhead is assumed independent of $P$ per processor.** With communication $C(P)$ per rank (Task 12):
$$T_o = a(P-1) + P\,C(P)$$
| $C(P)$ | isoefficiency |
|---|---|
| constant (halo exchange) | $\Theta(P)$ |
| $\log P$ (collectives) | $\Theta(P\log P)$ |
| $\sqrt{P}$ (2D mesh) | $\Theta(P^{3/2})$ |
| $P$ (all-to-all) | $\Theta(P^2)$ |
Only programs whose per-rank overhead is constant are truly Gustafson-compliant.
2. **"Problem size" is not "problem dimension".** Isoefficiency counts work $W$. For GEMM, $W \propto N^3$, so $\Theta(P)$ isoefficiency means $N \propto P^{1/3}$ (Task 16). Memory may limit this first (Sun–Ni, Task 11).
3. **Gustafson fixes time; isoefficiency fixes efficiency.** With $T_o = \Theta(P)$ they coincide. With growing $C(P)$, holding efficiency constant requires $W$ to grow faster than $P$, so per-rank work and run time must grow. **Constant time and constant efficiency then become incompatible goals.**
4. **Isoefficiency is more informative** in practice: it gives the growth rate needed, while Gustafson's law silently assumes the best case $\Theta(P)$.
