# Task 04 — Asymptotic Analysis of Parallel Efficiency

## 1. Gustafson efficiency

$$E_G(P) = \frac{S_G(P)}{P} = \frac{P - \alpha(P-1)}{P} = 1 - \alpha + \frac{\alpha}{P}$$

**Limit.**
$$\lim_{P\to\infty} E_G(P) = \lim_{P\to\infty}\left(1-\alpha+\frac{\alpha}{P}\right) = 1-\alpha$$

*Proof ($\varepsilon$–$N$):* $|E_G(P) - (1-\alpha)| = \alpha/P$. For any $\varepsilon > 0$, choose $N > \alpha/\varepsilon$; then for all $P > N$, $\alpha/P < \varepsilon$. $\blacksquare$

**Properties.**
* $E_G$ is monotonically decreasing: $dE_G/dP = -\alpha/P^2 < 0$.
* It is bounded below by $1-\alpha$, which is a **positive constant**. Efficiency never collapses; with $\alpha = 4\%$ it stays above 96% for any number of processors (Task 06).
* Most of the drop happens at small $P$. Going from $P=1$ to $P=2$ already loses $\alpha/2$, half of the total loss.

## 2. Amdahl efficiency

$$E_A(P) = \frac{S_A(P)}{P} = \frac{1}{P\left(s + \frac{1-s}{P}\right)} = \frac{1}{sP + 1 - s}$$

**Limit.**
$$\lim_{P\to\infty} E_A(P) = \lim_{P\to\infty}\frac{1}{sP + 1 - s} = 0 \quad (s>0)$$

*Proof:* the denominator grows without bound for $s>0$. For any $\varepsilon>0$, $E_A < \varepsilon$ whenever $P > \frac{1/\varepsilon - (1-s)}{s}$. $\blacksquare$

Asymptotically $E_A(P) \sim \dfrac{1}{sP}$: efficiency decays like $1/P$, because the speedup saturates at $1/s$ while the processor count keeps growing.

## 3. Comparison

| | Gustafson | Amdahl |
|---|---|---|
| $S(P)$ as $P\to\infty$ | $\sim (1-\alpha)P$, unbounded | $\to 1/s$, bounded |
| $E(P)$ as $P\to\infty$ | $\to 1-\alpha > 0$ | $\to 0$, like $1/(sP)$ |
| Half-efficiency point $E = 0.5$ | Never reached if $\alpha < 0.5$ | $P_{1/2} = 1 + 1/s$ |

Example, $\alpha = s = 0.04$:
* Gustafson: $E(16384) = 0.960002$.
* Amdahl: $E(16384) = 1/(0.04\cdot 16384 + 0.96) \approx 0.0015$, and $E = 0.5$ already at $P = 26$.

## 4. Deduction

The two limits differ because the denominators differ. In Amdahl, the work is fixed, so the serial part is paid by an ever-growing number of idle processors: idle processor-time grows as $s(P-1)$ while useful work is fixed. In Gustafson, useful work grows as $(1-\alpha)P$ at the same rate as idle time $\alpha(P-1)$, so their ratio converges to a constant. **Weak scaling converts a vanishing efficiency into a constant efficiency.** Real systems fall between the two because communication cost grows with $P$ (Task 12), which makes $E_G \to 0$ again, just much more slowly.
