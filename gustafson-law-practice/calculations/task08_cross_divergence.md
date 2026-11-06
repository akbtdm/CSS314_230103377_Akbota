# Task 08 — Cross-Over Point Determination (Amdahl vs. Gustafson)

**Given:** $s = \alpha = 0.08$, $P = 128$.

## Amdahl (fixed size)
$$S_A = \frac{1}{s + \frac{1-s}{P}} = \frac{1}{0.08 + \frac{0.92}{128}}$$
1. $0.92 / 128 = 0.0071875$
2. $0.08 + 0.0071875 = 0.0871875$
3. $1 / 0.0871875 = 11.4695$

**$S_A \approx 11.47$**, efficiency $11.47/128 = 8.96\%$.

## Gustafson (scaled size)
$$S_G = P - \alpha(P-1) = 128 - 0.08\cdot 127$$
1. $0.08 \cdot 127 = 10.16$
2. $128 - 10.16 = 117.84$

**$S_G = 117.84$**, efficiency $117.84/128 = 92.06\%$.

## Discrepancy

| Measure | Value |
|---|---|
| Ratio $S_G/S_A$ | $117.84 / 11.4695 = 10.27\times$ |
| Difference relative to Amdahl | $(117.84 - 11.47)/11.47 = +927.4\%$ |
| Difference relative to Gustafson | $(117.84 - 11.47)/117.84 = 90.27\%$ |
| Absolute gap | $106.37$ |

## Why they diverge as P grows

$$S_G - S_A = \underbrace{(1-\alpha)P + \alpha}_{\text{grows linearly}} - \underbrace{\frac{1}{s + (1-s)/P}}_{\to 1/s = 12.5}$$

* Amdahl's curve saturates at $1/0.08 = 12.5$; at $P=128$ it is already at 92% of that ceiling.
* Gustafson's curve keeps growing with slope $0.92$.
* So the ratio $S_G/S_A \to s(1-s)P$ grows **linearly** with $P$: it is $10.3\times$ at 128, about $76\times$ at 1024 ($942.2/12.36$), and unbounded.
* The two models agree only at $P = 1$. For any $0 < s < 1$ and $P > 1$, $S_G > S_A$, because they use the same number for different quantities (Task 03). The parallel-frame 8% corresponds to an Amdahl-frame serial fraction of only $0.08/117.84 = 0.068\%$ for the scaled problem.
