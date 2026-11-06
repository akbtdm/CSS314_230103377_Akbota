# Task 07 — Reverse Engineering Maximum Tolerable Serial Overhead

**Given:** $P = 8192$, requirement $S(P) \ge 7500$.

## Derivation

$$S(P) = P - \alpha(P-1) \ge S_{req}$$
$$P - S_{req} \ge \alpha(P-1)$$
Since $P-1 > 0$, dividing keeps the inequality direction:
$$\alpha \le \alpha_{max} = \frac{P - S_{req}}{P - 1}$$

## Calculation

$$\alpha_{max} = \frac{8192 - 7500}{8192 - 1} = \frac{692}{8191} = 0.0844830\ldots$$

**$\alpha_{max} \approx 0.08448$, i.e. 8.448% of the parallel run time.**

## Verification
$S = 8192 - 0.0844830 \cdot 8191 = 8192 - 692.000 = 7500.0$ ✓

Equivalent efficiency target: $E = 7500/8192 = 91.55\%$.

## Interpretation

* In a run that takes 100 s on the cluster, at most about 8.45 s may be serial (I/O, setup, reductions), and 91.55 s must be perfectly parallel.
* $\alpha_{max} \approx 1 - S_{req}/P$ for large $P$: the tolerable serial fraction equals the efficiency you are allowed to lose.
* Contrast with Amdahl: $S_A \ge 7500$ would need $s \le \dfrac{1/7500 - 1/8192}{1 - 1/8192} \approx 1.13\times 10^{-5}$, about **7,500× stricter**, which is unachievable for most real codes.
