# Task 02 — Formal Derivation of Scaled Speedup

**Goal:** derive $S(P) = s + p\cdot P = P - \alpha(P-1)$.

## Setup

| Symbol | Meaning |
|---|---|
| $P$ | number of processors |
| $s$ | time spent in serial code **on the parallel machine** |
| $p$ | time spent in parallel code **on the parallel machine** (all $P$ processors busy) |
| $\alpha$ | serial fraction of the parallel run, $\alpha = s/(s+p)$ |

## Derivation

**Step 1 — Normalise the parallel run.**
$$T_P = s + p = 1$$
*Justification:* we measure everything on the parallel machine and choose the time unit so the observed run takes 1. Then $\alpha = s/(s+p) = s$, so $s=\alpha$ and $p = 1-\alpha$.

**Step 2 — Count the work done in the parallel run.**
During the serial phase one processor works for time $s$: work $= s$.
During the parallel phase $P$ processors each work for time $p$: work $= P\cdot p$.
$$W = s + P\,p$$
*Justification:* work is measured in processor-time units; parallel time $p$ is spent simultaneously on $P$ processors.

**Step 3 — Equivalent serial workload.**
A single processor running the *same scaled problem* must execute all of that work one piece after another:
$$T_1 = s + P\,p$$
*Justification:* the single processor has no other processors to share the $P$ pieces of parallel work with, so it runs them one after another. This is the key move: $T_1$ is a **hypothetical** serial run of the large problem, not a measured run of the small one.

**Step 4 — Scaled speedup.**
$$S(P) = \frac{T_1}{T_P} = \frac{s + P\,p}{s + p} = \frac{s + P\,p}{1} = s + p\cdot P$$
*Justification:* definition of speedup and Step 1.

**Step 5 — Substitute $s=\alpha$, $p = 1-\alpha$.**
$$S(P) = \alpha + (1-\alpha)P = \alpha + P - \alpha P$$

**Step 6 — Rearrange.**
$$S(P) = P - \alpha P + \alpha = P - \alpha(P-1) \qquad \blacksquare$$

## Checks

* $P = 1$: $S = 1 - \alpha\cdot 0 = 1$. Correct, a processor is not faster than itself.
* $\alpha = 0$: $S = P$. A perfectly parallel program gets ideal linear speedup.
* $\alpha = 1$: $S = 1$. A fully serial program gets no speedup.
* $dS/dP = 1-\alpha$: a straight line with slope $1-\alpha$, never saturating.

## Efficiency (used later)
$$E(P) = \frac{S(P)}{P} = 1 - \alpha\,\frac{P-1}{P}$$
