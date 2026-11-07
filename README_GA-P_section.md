## Team T10 -- Checkpoint 4 (storyline B, variant B2, instance cp34_instance(10, 'B', 2))
| Student | Config | Role | Files |
|---|---|---|---|
| Telgarin Diar | GA, repair | GA-A | src/meta/ga.py, experiments/run_final.py use, results/telgarin_runs.csv, random-search baseline |
| Abdimazhit Inkar | GA, repair | GA-P | src/meta/ga_problem.py, tests/test_GA-P.py, experiments/exp_repair.py, results/abdimazhit_exp.csv |

Agreed interface GA-P <-> GA-A (numpy int arrays of length M, numpy Generator `rng`): prepare(d), random_individual(ctx, rng),
repair(ctx, ind, rng), evaluate(ctx, ind, rho=None) -> (fitness, cost, feasible), crossover(ctx, p1, p2, rng), mutate(ctx, ind, rng).
Repair rule: zone fix, then move the smallest sufficient VM from the most overloaded server to the allowed server with room in CPU and memory and lowest CPU utilisation; fallback FFD rebuild. Reads data only (free).
Reproduce: `python tests/test_GA-P.py`;  `python experiments/exp_repair.py --out results/abdimazhit_exp.csv`;
final runs (GA-A): `python experiments/run_final.py --module src.meta.ga --instance-seed 10 --story B --variant 2 --team T10 --role GA-A --configuration "GA, repair" --out results/telgarin_runs.csv`

### Declaration (Abdimazhit Inkar)
| Tool or source | Used for | Files | What I checked myself |
|---|---|---|---|
| (fill in, e.g. AI assistant) | (e.g. draft of repair rule, tests) | src/meta/ga_problem.py, tests/test_GA-P.py | (e.g. evaluate vs evaluate_B on 1000 individuals; read and rewrote every function) |
