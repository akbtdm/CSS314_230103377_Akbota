# Task 03 — The Serial Fraction Invariance Paradox

**Memo topic:** the frame of reference in which overhead is measured.

## 1. The apparent contradiction

The same program can be called "5% serial" and still achieve near-linear speedup on 1000 processors. Amdahl says the speedup can never exceed $1/0.05 = 20$. Gustafson says it is about $1000 - 0.05\cdot 999 \approx 950$. Both use "5%", so how can both be true?

**Resolution:** the two 5% are **different numbers measured in different frames.**

* Amdahl's $s$ is a fraction of the **uniprocessor time** $T_1$.
* Gustafson's $\alpha$ is a fraction of the **parallel time** $T_P$.

## 2. Converting between the frames

Take one concrete run on $P$ processors with serial time $a$ and parallel time $b$ (seconds, measured on the parallel machine).

* Parallel frame: $T_P = a + b$, so $\alpha = \dfrac{a}{a+b}$.
* The same work on one processor: $T_1 = a + Pb$, so Amdahl's fraction of that *same* run is
$$s = \frac{a}{a + Pb} = \frac{\alpha}{\alpha + (1-\alpha)P}$$

Now substitute this $s$ into Amdahl's formula:
$$S_A = \frac{1}{s + \frac{1-s}{P}}
      = \frac{\alpha + (1-\alpha)P}{\alpha + (1-\alpha)} = \alpha + (1-\alpha)P = S_G$$

**For the same run, the two laws give exactly the same speedup.** They are one identity written with two different serial fractions. The "contradiction" only appears if you plug the parallel-frame $\alpha$ into Amdahl's uniprocessor-frame formula.

## 3. Why the serial fraction shrinks when the problem grows

From $s = \alpha / (\alpha + (1-\alpha)P)$: with $\alpha$ fixed at 5%,

| $P$ | Amdahl-frame $s$ of the scaled problem |
|---|---|
| 1 | 5.00% |
| 16 | 0.33% |
| 1024 | 0.0051% |

In absolute time the serial part stays the same (setup, I/O, a global reduction), but it is diluted by a parallel part that grows with $P$. Measured on the uniprocessor for the **scaled** problem, $s \to 0$, so Amdahl's ceiling $1/s \to \infty$.

The nCUBE data shows exactly this (Task 05): fixed-size serial component $s \approx 0.0006$–$0.001$, while for the scaled problems the equivalent serial component dropped to a few parts per million.

## 4. Critical assessment

* **Measuring on the parallel machine is honest:** it is the only frame in which both parts are actually observed. A uniprocessor run of a problem that needs 1024 nodes' memory usually cannot be executed at all, so $T_1$ is extrapolated anyway.
* **Hidden assumption:** Gustafson needs the *absolute* serial time $a$ to stay constant as the problem grows. If the serial part grows with problem size (global sort, sequential I/O of a growing output, all-to-all communication), $\alpha$ is not invariant and the linear law over-predicts (Tasks 12–13).
* **Lesson:** a serial fraction is meaningless unless you state the frame (uniprocessor or parallel) and the problem size it was measured at.
