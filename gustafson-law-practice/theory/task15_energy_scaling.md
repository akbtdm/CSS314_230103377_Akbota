# Task 15 — Energy and Power Proportionality under Scaled Speedup

## 1. Power model

* Each node draws static power $P_{idle}$ whenever it is powered on.
* When computing, it draws an additional dynamic power $P_{dyn} = P_{active} - P_{idle}$.
* Gustafson run on $P$ nodes, wall-clock $T$ (normalised $s + p = 1$, so $T_s = \alpha T$, $T_p = (1-\alpha)T$):
  * Serial phase: 1 node active, $P-1$ nodes idle.
  * Parallel phase: all $P$ nodes active.

## 2. Energy of the weak-scaling run

$$E_P = \underbrace{P\,P_{idle}\,T}_{\text{static, all nodes, all the time}} + \underbrace{P_{dyn}\left[\alpha T + P(1-\alpha)T\right]}_{\text{dynamic, only while computing}}$$
Using $S(P) = \alpha + (1-\alpha)P$:
$$\boxed{E_P = T\left[P\,P_{idle} + P_{dyn}\,S(P)\right]}$$

## 3. Comparison with 1 core, identical wall-clock

**Same time $T$, one node:** $E_1 = T(P_{idle} + P_{dyn})$. That node completes 1 unit of work; the $P$-node run completes $S(P)$ units.

$$\frac{E_P}{E_1} = \frac{P\,P_{idle} + P_{dyn}\,S(P)}{P_{idle} + P_{dyn}}$$

**Energy per unit of work (the fair comparison):**
$$e_1 = T(P_{idle} + P_{dyn}), \qquad e_P = \frac{E_P}{S(P)} = T\left[P_{dyn} + P_{idle}\frac{P}{S(P)}\right] = T\left[P_{dyn} + \frac{P_{idle}}{E(P)}\right]$$

$$e_P - e_1 = T\,P_{idle}\left(\frac{1}{E(P)} - 1\right) \ge 0$$

**Interpretation:** the parallel run does $S(P)$ times more work in the same time, but every unit of work costs extra static energy, because idle nodes during the serial phase still burn $P_{idle}$. Dynamic energy per unit of work is the same in both cases; only static energy is wasted, in proportion to $1/E - 1$.

**Same total work instead:** doing the scaled problem on 1 node takes $T_1 = S(P)\,T$, so $E_1' = S(P)T(P_{idle} + P_{dyn})$ and
$$E_P - E_1' = T\,P_{idle}\,(P - S(P)) = T\,P_{idle}\,\alpha(P-1)$$
The extra energy is exactly the static energy of the $\alpha(P-1)$ "lost" processors.

## 4. Numerical example

$P_{idle} = 50$ W, $P_{active} = 150$ W ($P_{dyn} = 100$ W), $T = 1$ h, $\alpha = 0.04$, $P = 1024$:

* $S = 983.08$, $E = 0.96004$.
* $E_P = 1\,\text{h}\,[1024\cdot 50 + 100\cdot 983.08] = 149{,}508$ Wh $\approx 149.5$ kWh.
* 1 node for 1 h: $150$ Wh for 1 unit of work.
* Energy per unit of work: $149508/983.08 = 152.08$ Wh, i.e. +1.4% vs. 150 Wh.
* Same work on 1 node: $983.08\cdot150 = 147{,}462$ Wh. The $P$-node run costs $2{,}046$ Wh more $= 50\cdot 0.04\cdot 1023$ ✓, but finishes 983× sooner.

## 5. Conclusions
* Weak scaling is **nearly energy-proportional**: energy per unit work rises only by the factor $P_{idle}(1/E-1)$, which stays bounded because $E\to 1-\alpha$.
* Under strong scaling (Amdahl), $E\to 0$, so energy per unit work grows without bound. Fixed-size runs on huge machines waste energy.
* Gains: lower $P_{idle}$ (power gating idle cores during serial phases, DVFS), and a smaller $\alpha$.
* The model ignores network/interconnect power and cooling (PUE), which grow with $P$ and make large runs slightly worse in practice.
