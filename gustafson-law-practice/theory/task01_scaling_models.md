# Task 01 — Strong Scaling vs. Weak Scaling Foundations

## 1. The question each law asks

| | Amdahl (1967) | Gustafson (1988) |
|---|---|---|
| Question | *How much faster can I solve **this** problem?* | *How much **bigger** a problem can I solve in the same time?* |
| Fixed quantity | Total problem size $W$ | Wall-clock time $T$ |
| What grows with $P$ | Nothing; work per processor shrinks as $W/P$ | Total work $W(P) \approx P \cdot W(1)$; work per processor is constant |
| Serial fraction measured on | A uniprocessor (fraction of $T_1$) | The parallel machine (fraction of $T_P$) |
| Speedup | $S_A(P) = \dfrac{1}{s + \frac{1-s}{P}}$ | $S_G(P) = s + (1-s)P = P - \alpha(P-1)$ |
| Limit as $P\to\infty$ | $S_A \to 1/s$ (hard ceiling) | $S_G \to \infty$, linear with slope $1-\alpha$ |
| Scaling regime | Strong scaling | Weak scaling |

## 2. Amdahl's assumptions

1. The program is a fixed set of operations: a serial part $s$ and a perfectly parallel part $1-s$, with $s+(1-s)=1$ measured on one processor.
2. Adding processors only divides the parallel part: $T_P = s + (1-s)/P$.
3. The serial part is **independent of $P$** and of the problem size, because the problem size never changes.

**Problem dimension as $P \to \infty$:** constant. The per-processor share $W/P \to 0$, so every processor eventually spends almost all its time waiting for the serial part. That is why the speedup saturates at $1/s$: with $s = 5\%$ you never exceed $20\times$, even with a million cores.

## 3. Gustafson's assumptions

1. Users fix the **time they are willing to wait** and choose the largest problem that fits in it (finer grids, more particles, more time steps).
2. On the parallel machine the run is normalised to $s' + p' = 1$. The serial part $s'$ (I/O, setup, global reductions) does **not** grow with the problem; the parallel part grows with $P$.
3. The same job on one processor would take $s' + P\,p'$.

**Problem dimension as $P \to \infty$:** grows without bound, $W(P) = s' + P p'$, so per-processor work stays constant. The serial part becomes a smaller and smaller fraction of the *equivalent serial* work, and speedup grows linearly.

## 4. Why "fixed-size" vs "fixed-time"

* **Amdahl = fixed-size.** The invariant is $W$. Time $T_P$ is the dependent variable that we try to shrink. More processors mean less work each and a growing relative weight of serial and communication costs.
* **Gustafson = fixed-time.** The invariant is $T_P$ (normalised to 1). Work $W$ is the dependent variable that we try to grow. More processors mean more total work done in the same time.

## 5. Philosophical and physical divergence

* **Different definitions of the problem.** Amdahl treats the problem as given; Gustafson treats it as a parameter the scientist tunes to the machine (resolution, accuracy). Gustafson argued that in practice nobody runs a fixed-size problem on a 1024-node machine except for academic benchmarking.
* **Different behaviour of the serial fraction.** Amdahl assumes $s$ is a property of the program. Gustafson observes that $s$ depends on problem size: setup and I/O usually grow more slowly than the computation ($O(N^2)$ surface vs. $O(N^3)$ volume), so the serial *fraction* shrinks as the problem grows.
* **Both are correct inside their own assumptions.** They are the same algebra seen from two reference frames (see Task 03). Strong scaling answers latency questions ("finish this forecast sooner"); weak scaling answers capability questions ("run a finer forecast in the same 6 hours").
* **Physical limits neither law models:** memory capacity (Sun–Ni, Task 11), communication cost growing with $P$ (Task 12), and algorithms whose work grows superlinearly with problem size (Tasks 13, 16, 18).
