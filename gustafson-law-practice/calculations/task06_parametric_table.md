# Task 06 — Parametric Speedup Across Processor Densities

**Given:** $\alpha = 0.04$.

**Formulas:**
$$S(P) = P - \alpha(P-1), \qquad E(P) = \frac{S(P)}{P} = 1 - \alpha\frac{P-1}{P}$$

## Worked example (P = 16)
$S = 16 - 0.04\cdot 15 = 16 - 0.60 = 15.40$, and $E = 15.40/16 = 0.9625$.

## Table

| $P$ | $\alpha(P-1)$ | $S(P)$ | $E(P)$ | Lost cores $P - S$ |
|---:|---:|---:|---:|---:|
| 16 | 0.60 | 15.40 | 0.962500 | 0.60 |
| 64 | 2.52 | 61.48 | 0.960625 | 2.52 |
| 256 | 10.20 | 245.80 | 0.960156 | 10.20 |
| 1,024 | 40.92 | 983.08 | 0.960039 | 40.92 |
| 4,096 | 163.80 | 3,932.20 | 0.960010 | 163.80 |
| 16,384 | 655.32 | 15,728.68 | 0.960002 | 655.32 |

Numeric log (Python, `benchmarks/simulator.py --alpha 0.04`):
```
16 15.4 0.9625
64 61.48 0.960625
256 245.8 0.96015625
1024 983.08 0.9600390625
4096 3932.2 0.960009765625
16384 15728.68 0.96000244140625
```

## Evaluation

* Processor count grows 1024× (16 → 16,384), yet efficiency falls only from 96.25% to 96.0002%, a total drop of 0.25 percentage points.
* $E(P)$ converges to the limit $1-\alpha = 0.96$ (Task 04). The gap above the limit is exactly $\alpha/P$, so it shrinks 4× for every 4× step in $P$.
* Speedup is linear with slope $0.96$. The absolute number of "lost" cores $\alpha(P-1)$ grows linearly too, but always remains 4% of the machine.
* Under Amdahl with the same 4% the speedup would be capped at 25×, and $E(16384) \approx 0.15\%$. This is the practical case for designing large systems around weak scaling.
* **Caveat:** these are ideal numbers with $\alpha$ constant. On real machines communication grows with $P$ (Task 12), so measured efficiency at 16K cores would be lower.
