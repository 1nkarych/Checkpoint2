## GA-P part (Abdimazhit Inkar): representation, operators, repair, role test, repair experiment

**Files of this role:** `src/meta/ga_problem.py`, `tests/test_GA-P.py`, `experiments/exp_repair.py`, `results/abdimazhit_exp.csv`, `report/prediction_GA-P.txt`.

**Representation.** An individual is an integer array of length M = 50; `ind[k]` is the server of VM k. A VM with a required zone may only use servers of that zone (allowed servers are precomputed per VM).

**Operators** (all random numbers come from the `rng` passed in by `solve`):
- `random_individual(d, rng, repair=True)`: random allowed server for every VM; repaired when `repair=True`, raw when `False`. The parameter hides the function `repair` inside the function, so the module keeps the alias `_repair_fn = repair` and calls that.
- `evaluate(d, ind, rho=None)`: returns `(fitness, cost, feasible)`. The cost is the piecewise-linear latency of the checker (tangents 0, 0.4, 0.7, 0.9); `feasible` means no CPU or memory overload and no zone violation. `violation(d, ind)` is the sum of the relative CPU and memory overloads plus the number of zone violations. `ga.py` forms the fitness `cost + rho * violation` itself.
- `crossover`: uniform, one child. `mutate`: moves 1 or 2 VMs to another allowed server.
- `repair(d, ind, rng)`: (1) a VM on a forbidden server goes to a random allowed server; (2) while some server is overloaded in CPU or memory, one overloaded server is chosen at random, and among its VMs that have a target with room the smallest one that removes the overload moves (if none removes it, the largest one moves). The target is the allowed server with room in both resources and the lowest CPU utilisation after the move. Only loads and capacities are read, so repair costs no evaluations. **Limitation:** if no VM can be moved, repair returns the individual with its remaining violation (there is no fallback). On this instance (fill about 46% CPU and 50% memory) this did not happen in my experiment (0 repair failures).

**Role test** (`python tests/test_GA-P.py`): `evaluate` equals the checker `evaluate_B` on 1000 individuals, `crossover` and `mutate` return valid individuals on 1000 pairs, the repair output passes the checker, a feasible individual is returned unchanged and repair does not call the cost function, and the penalty formula is checked on an example.

**Repair experiment** (`python experiments/exp_repair.py`, writes `results/abdimazhit_exp.csv`): full GA runs, budget 100 000, seeds 100-104, repair on versus repair off with penalty `rho` = 1, 10, 100. The share of feasible offspring is stored in the `params` column.

| Setting | Mean final cost | Min - max | Mean share of feasible offspring | Runs ending infeasible |
|---|---|---|---|---|
| repair on | 8.4213 | 8.4088 - 8.4380 | 100% | 0 of 5 |
| penalty, rho = 1 | 8.4807 | 8.4460 - 8.5726 | 34.5% | 0 of 5 |
| penalty, rho = 10 | 8.4469 | 8.4361 - 8.4771 | 61.7% | 0 of 5 |
| penalty, rho = 100 | 8.4866 | 8.4698 - 8.5001 | 64.7% | 0 of 5 |

With five seeds the ranges overlap (the best penalty run with rho = 10 is better than the worst repair run), so the advantage of repair is a tendency, not a proven difference. The prediction (`report/prediction_GA-P.txt`) was finalised after I had seen earlier results; it is a retrospective prediction.

**Authorship note.** `src/meta/ga_problem.py` first appeared in commit `a65bc2a` (Diar Telgarin, 07.11) as a temporary stand-in so that `ga.py` could be run. The implementation of this role, its fixes and the `random_individual` repair switch were committed by me in `57dc71d`, `e81beb8` and `b2044ed`; the role test and the experiment script were first committed by me (`57dc71d`, `70c73a9`). The history was not rewritten.

### Declaration (Abdimazhit Inkar)
| Tool or source | Used for | Files | What I checked myself |
|---|---|---|---|
| AI assistant (Claude) | drafts of the operators and the repair rule, the role test, the repair experiment script, the `random_individual` repair switch, explanations of git steps and of the analysis | `src/meta/ga_problem.py`, `tests/test_GA-P.py`, `experiments/exp_repair.py` | ran the role test and the test of the partner, ran the whole experiment myself (20 runs) and compared the results with a reference run; [ADD here, truthfully, what you read, changed or can explain yourself] |
| Python libraries (`numpy`) | arrays, random numbers | `src/meta/ga_problem.py` | - |
