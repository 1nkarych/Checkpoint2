# Block 1. From the Checkpoint 1 model to the reference model

Team T10, storyline B (cloud resource allocation), variant 2; instance `cp34_instance(10, "B", 2)`: S = 11 servers, M = 50 VMs, 7 premium VMs with a zone condition, CPU fill 233/504 = 46.2 %, memory fill 871/1725 = 50.5 %.
Reference values (HiGHS, rebuilt in `final/src/final_block1.py` and independently checked): z_LP = 8.339506, z_ref = 8.382826 (optimum proven, solver gap 0).
Every number below comes from a file in `results/`; commands are in the README, section "Final".

**Source of the Checkpoint 1 model.** The Checkpoint 1 column of Table 1 is the model as recorded in the team's working notes: binary x[i,j]; U_j^cpu and U_j^mem the CPU and memory utilisation of server j; U_j = max(U_j^cpu, U_j^mem); objective min Σ_j (U_j + U_j²); constraints: one server per VM, CPU capacity, memory capacity, premium VM only in its required zone, integrality. This column was **not re-compared line by line with the Checkpoint 1 PDF** when this text was written, and the available files do not say which team member's individual Checkpoint 1 model it is; the text therefore names no author. All statements about "my Checkpoint 1 objective" below refer to this objective, `cp1_objective` in `final/src/final_block1.py`.

## Table 1. Element by element

| Element | Checkpoint 1 model | Reference model (Checkpoint 3 on) | Status | What the change does to the meaning of the answer |
|---|---|---|---|---|
| Assignment variables | x[i,j] ∈ {0,1} | y[k,l] ∈ {0,1}; [0,1] in the LP relaxation | kept | Same decision (which server hosts which VM); the relaxation only gives z_LP. |
| Utilisation variables | U_j^cpu, U_j^mem, U_j = max(·,·), determined by x | not variables; only the CPU utilisation u_l = Σ_k cpu_req[k] y[k,l] / cpu_cap[l] appears in the latency rows | changed | The memory part of the load is no longer measured by the cost. |
| Latency variables | none (latency is a function of x) | t[l] ≥ 0, one per server (11) | changed (added) | Makes the nonlinear latency expressible linearly; the decision itself is not touched. |
| Objective | min Σ_j (U_j + U_j²) | min Σ_l t[l] | changed | A different function of a different argument (peak of CPU and memory against CPU only): the two objectives can order solutions differently. |
| Latency function | L(U) = U + U², U = peak utilisation (assumed) | u/(1−u), replaced by the maximum of 4 tangents at u = 0, 0.4, 0.7, 0.9 (`latency_B`) | changed | The reference penalises high load much more steeply (tangent slopes 1, 2.78, 11.1, 100; U + U² has slope ≤ 3 on [0,1]), so it prefers more balanced CPU loads. |
| Constraint group 1: one server per VM | Σ_j x[i,j] = 1 | Σ_l y[k,l] = 1 (50 rows) | kept | Every VM is placed exactly once in both models. |
| Constraint group 2: CPU capacity | Σ_i r^cpu_i x[i,j] ≤ c^cpu_j | same (11 rows) | kept | Same feasible set with respect to CPU. |
| Constraint group 3: memory capacity | Σ_i r^mem_i x[i,j] ≤ c^mem_j | same (11 rows) | kept | Same feasible set; in the reference, memory is a constraint only and does not enter the cost. |
| Constraint group 4: premium zone | x[i,j] = 0 if the server zone differs from the required zone | variables fixed to 0 for forbidden pairs (51) | kept | Hard in both models; the two feasible sets coincide. |
| Latency rows | none | t[l] ≥ tangent_m(u_l), 44 rows | changed (added) | They replace the explicit nonlinear objective; z_ref is optimal for the piecewise-linear cost, not for u/(1−u). |
| Integrality | x[i,j] ∈ {0,1} | y[k,l] ∈ {0,1} | kept | Same. |
| Assumption 1: latency from utilisation, no latency matrix | yes | yes | kept | Latency is a function of the server load, not of the VM–server pair. |
| Assumption 2: form of the latency | named M/M/1, but U + U² used | u/(1−u) (the M/M/1 form) by tangents | changed | The reference uses the form that Checkpoint 1 only named; the shape at high load differs. |
| Assumption 3: load = peak of two resources | yes | CPU only | dropped | In the reference a server with full memory and low CPU has low latency. |
| Assumption 4: static, centralised allocation of all 50 VMs | yes | yes | kept | Same planning horizon, all VMs placed. |
| Problem class | nonlinear integer program | mixed-integer linear program | changed | The reference can be solved exactly with HiGHS, so z_ref and its proof exist. |

Counts of the reference model (11 latency variables, 50 + 22 + 44 = 116 rows, 51 forbidden pairs, 550 + 11 = 561 variables) were checked by rebuilding the model (`final_block1.py --resolve-reference`) and by an independent second build; both give z_LP = 8.339506 and z_ref = 8.382826. Only the elements recorded for Checkpoint 1 are listed; no alternative that Checkpoint 1 may have considered and rejected is added.

## The Checkpoint 1 objective on reference solutions

The Checkpoint 1 objective has no part that the reference model lacks (same data, same constraints; no violation of the Checkpoint 1 constraints in any solution below), so it was evaluated exactly on every solution; nothing had to be replaced. Script: `final/src/final_block1.py`; results: `results/final_block1_eval.csv`, `final_block1_solutions.json`, `final_block1_decomposition.csv`, `final_block1_reference_resolve.json`, `final_block1_optimal_set.json`.

| Solution | Reference cost | Checkpoint 1 objective | Rank, reference | Rank, Checkpoint 1 |
|---|---|---|---|---|
| Reference optimum, assignment returned by HiGHS and stored in `final_block1_solutions.json` | 8.382826 | 11.294574 | 1 (tie with GA) | 2 |
| Best approximate solution, GA, seed 4 | 8.382826 (equal to z_ref up to 1e-9) | 10.688779 | 1 (tie with the optimum) | 1 |
| Best approximate solution, SA, seed 3 | 8.391877 (+0.108 % over z_ref) | 11.426841 | 2 | 3 |
| *Second HiGHS solve of the reference model (another optimal assignment)* | 8.382826 | 10.966209 | 1 | 2 |

Reproduction: `python final/src/final_block1.py` re-evaluates the stored assignments (control values 11.294574, 10.688779, 11.426841; the script stops if a difference exceeds 5·10⁻⁷); `--resolve-reference` re-solves the reference model (about 12 s) and gives the last row.

**What the numbers say.**
1. *The best GA solution is not an approximation of the optimum: its reference cost equals z_ref, so it is another optimal assignment of the reference model* (it differs from the stored optimum in 39 of 50 VMs). The reference model therefore ties the two solutions; the order "optimum vs best approximate solution" is not defined by it. Under the Checkpoint 1 objective the tie is resolved in favour of the GA assignment by 0.6058 (5.4 % of the value of the stored optimum).
2. The only solution that the reference model ranks strictly lower, the best SA solution (0.108 % above z_ref), also ranks last under the Checkpoint 1 objective (+0.1323, +1.17 % over the stored optimum). For the stored optimum and for the second HiGHS optimum the order "optimal assignment < SA" is therefore kept.
3. The tie of the two optimal assignments comes only from memory. Both have the same CPU load on every server (CPU-only part Σ(u + u²) = 7.151254), so the reference cost cannot tell them apart; the Checkpoint 1 objective also counts memory, which is the larger of the two utilisations on 9 of 11 servers of the stored optimum, 8 of the GA solution and 6 of the SA solution (`final_block1_decomposition.csv`).

**The reference optimum is not unique.** A second HiGHS solve returns the same z_ref = 8.382826 (proven) but another assignment with Checkpoint 1 value 10.966209 instead of 11.294574. The first row of the table thus depends on which optimal assignment the solver returns: it is a comparison of *these stored assignments*, not of "the" optimum, and not of the methods in general (a single GA and a single SA run). The value 0.6058 is a difference between two particular assignments, not a sampling difference; no significance statement is made.

**How large is the freedom among reference-optimal assignments?** Two MILPs (`--optimal-set`, reference cost fixed to ≤ z_ref + 10⁻⁶, U + U² modelled by 100 linear pieces, 600 s limit each, true values re-evaluated with `evaluate_B` and the Checkpoint 1 objective) found reference-optimal assignments with Checkpoint 1 values:

| | attained value (true, evaluated) | proven bound of the MILP | solver gap |
|---|---|---|---|
| minimum | 8.267809 | ≥ 8.264509 | 0.04 % (time limit) |
| maximum | 14.428043 | ≤ 14.702334 | 3.1 % (time limit) |

Both assignments have reference cost 8.382826 and the same CPU loads per server as the stored optimum (checked for the two assignments; it was not proved for the whole optimal set). So among assignments that the reference model considers equally optimal, the Checkpoint 1 objective takes at least all values from 8.2678 to 14.4280, a width of at least 6.16; the value 14.428 is a lower estimate of the true maximum. The difference 0.6058 above is below 10 % of this width.

## Conclusion

Under the reference model the stored optimum and the best GA solution are tied (both 8.382826), and the best SA solution is slightly worse (8.391877); under the Checkpoint 1 objective the stored optimum (11.2946) is beaten by the GA solution (10.6888) and beats the SA solution (11.4268). So the order is **kept for the SA solution and not defined by the reference model for the tied pair**: the Checkpoint 1 objective breaks the tie, in favour of the GA assignment, through the memory load that the reference model does not measure. We do not call this a reversal and do not treat it as evidence that the GA is better than the exact solver: a second exact solve gives an optimum at 10.9662, between the GA and SA values, so the sign of the difference depends on which optimal assignment the solver returned.

What does follow from the numbers is the answer to the question: **yes, the team would have chosen another solution had it kept its own model.** Among assignments with reference cost 8.382826 there is one with Checkpoint 1 value 8.267809, 26.8 % below the stored optimum and 22.6 % below the GA solution, so none of the compared solutions (the HiGHS pick, the GA or the SA solution) would be chosen under the Checkpoint 1 objective; and the reference model cannot distinguish an assignment with value 8.27 from one with value 14.43.

**Limits.** (i) 8.2678 was found by a MILP with a piecewise-linear model of U + U² under a time limit; it is an attained value (lower bound 8.2645), not a proved optimum, and the true optimum of the Checkpoint 1 model is not restricted to reference-optimal assignments and may be lower. (ii) The seed spread in Checkpoint 1 units was not computed, because the team's GA/SA code and per-seed solutions are not part of the repository used here (`final_block1.py --seeds` needs them), so nothing is claimed about seed variability under the Checkpoint 1 objective. (iii) The Checkpoint 1 column of Table 1 rests on the team's notes and was not re-checked against the Checkpoint 1 PDF.
