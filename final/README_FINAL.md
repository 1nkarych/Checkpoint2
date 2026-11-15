# Final starter additions (Block 3: scaling)

Copy these files into the repository of your team (the same layout as in the Checkpoint 4 kit):

    checkpoint4_generator_PUBLIC.py     REPLACES the old file: identical, but scaled_instance(story, size, variant) works for all variants
    check_env.py                        REPLACES the old file; new option --scaled
    experiments/run_scaling.py          runs solve (your methods) on a larger instance, seeds 0-9, writes a CSV
    experiments/exact_helper.py         solve_milp(c, constraints, bounds, integrality) for scipy models, solve_pulp(prob) for PuLP models: HiGHS, limit 120 s; both return objective, bound, solver_gap, proven, seconds

1. Install once, only if your model is a PuLP problem:  pip install pulp highspy   (scipy >= 1.9 is enough for the matrix route)
2. Check the generator: python check_env.py --scaled 100 --story A --variant 2   (storyline B: --scaled 200 --story B)
   It prints the size of the scaled instance: N, V, total demand (A) or S, M (B). Use it to be sure that the new generator is in place.
3. Approximate methods:
   python experiments/run_scaling.py --module src.meta.sa --story A --variant 2 --size 100 --team T05 --configuration "SA, own start" --out results/scaling_runs.csv --append
   Run again with --start-file rounded_A_100.json for the start from the rounded LP solution (format in the header of run_scaling.py).
4. Exact method: build YOUR Checkpoint 3 model for the scaled instance (the function that builds it must take the instance d as a parameter), then
   matrices + scipy:  from exact_helper import solve_milp;  res = solve_milp(c, constraints, bounds, integrality, limit=120)
   PuLP model:        from exact_helper import solve_pulp;  res = solve_pulp(prob, limit=120)
   Do not trust the word 'Optimal': use res['proven'] and res['solver_gap'] (it is not gap_LP of the report).
5. Sizes: storyline A  N = 60, 100, 150;  storyline B  M = 100, 200, 400.
6. Put the exact commands that reproduce Table 3 and Figure F2 into the README of your repository.
