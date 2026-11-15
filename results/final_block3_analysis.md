# Block 3 Scaling Analysis (T10, B2)
> Submission-ready: YES

Method: CP3 binary assignment MILP, B1–B4, CPU/memory/zone constraints; HiGHS 120s. SA-transfer and GA-repair preserve their CP4 parameters, 100,000 evaluations, seeds 0–9. LP-rounded start: process VMs by LP confidence and place each on the feasible allowed server with highest fractional y; break ties by least additional latency. The rounded assignment is checked with evaluate_B. Both methods receive it through solve(start=...).

## M=100, S=29
HiGHS: incumbent 9.176682; solver gap 0.0000%; elapsed 19.40 s; proven optimal (solver bound meets incumbent).
SA transfer, own: best 9.176682, gap to best known 0.0000%, valid runs 10.
SA transfer, LP-rounded: best 9.221847, gap to best known 0.4922%, valid runs 10.
GA repair, own: best 9.193057, gap to best known 0.1784%, valid runs 10.
GA repair, LP-rounded: best 9.215408, gap to best known 0.4220%, valid runs 10.

## M=200, S=57
HiGHS: incumbent 24.928666; solver gap 0.2257%; elapsed 120.07 s; feasible incumbent; optimality not proven.
SA transfer, own: best 25.176733, gap to best known 0.9951%, valid runs 10.
SA transfer, LP-rounded: best 25.160631, gap to best known 0.9305%, valid runs 10.
GA repair, own: best 25.023959, gap to best known 0.3823%, valid runs 10.
GA repair, LP-rounded: best 25.077799, gap to best known 0.5982%, valid runs 10.
No approximate method in these runs beat the HiGHS incumbent.

## M=400, S=114
HiGHS: incumbent 61.199431; solver gap 10.4199%; elapsed 120.31 s; feasible incumbent; optimality not proven.
SA transfer, own: best 55.580900, gap to best known 0.0000%, valid runs 10.
SA transfer, LP-rounded: best 55.645485, gap to best known 0.1162%, valid runs 10.
GA repair, own: best 55.604876, gap to best known 0.0431%, valid runs 10.
GA repair, LP-rounded: best 55.657386, gap to best known 0.1376%, valid runs 10.
Approximate method(s) beating the HiGHS incumbent: SA transfer (own), SA transfer (LP-rounded), GA repair (own), GA repair (LP-rounded).

## Cross-size comparison
First size without a proof of optimality in these measurements: M=200. A missing solver bound or incumbent must be reported as such.
First size at which an approximate run beat the unproven HiGHS incumbent: M=400, SA transfer (own) cost 55.580900 versus exact incumbent 61.199431.
LP-rounded effect at M=100, SA transfer: worsened best cost by 0.045165 relative to own start (rounded − own = +0.045165).
LP-rounded effect at M=100, GA repair: worsened best cost by 0.022351 relative to own start (rounded − own = +0.022351).
LP-rounded effect at M=200, SA transfer: improved best cost by 0.016102 relative to own start (rounded − own = -0.016102).
LP-rounded effect at M=200, GA repair: worsened best cost by 0.053840 relative to own start (rounded − own = +0.053840).
LP-rounded effect at M=400, SA transfer: worsened best cost by 0.064585 relative to own start (rounded − own = +0.064585).
LP-rounded effect at M=400, GA repair: worsened best cost by 0.052510 relative to own start (rounded − own = +0.052510).
## Interpretation and limits
Compare each method to the feasible incumbent and the solver best bound; a solver gap is NOT gapLP. An exact method that reaches the 120-second limit without a matching bound has NOT proven optimality. Three selected sizes cannot establish a general complexity law or a crossover threshold outside this range; additional intermediate sizes and repeated solver timings would test stability of the crossover and solve times.
Use only source data from results/final_*.csv; the manuscript must not invent other numbers.
