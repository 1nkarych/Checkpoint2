# Team T10 - Introduction to Optimization (storyline B, variant 2)

Members: Abdimazhit Inkar, Hafizi Nazifullah, Shaken Daniyal, Telgarin Diar. Instance: `cp34_instance(10, 'B', 2)`.
Contents: [Final](#final) (first), then the README of Checkpoint 4 (unchanged).

## Final

Team T10, storyline B, variant 2, instance `cp34_instance(10, 'B', 2)`; scaled instances `scaled_instance('B', M, 2)`, M = 100, 200, 400.
Needs Python 3 with numpy, scipy >= 1.9, matplotlib (no PuLP: the model is built as matrices for `scipy.optimize.milp`, HiGHS).
Run everything from the repository root.

| Step | Command | Writes |
|---|---|---|
| Second run (Abdimazhit): LP relaxation and LP-rounded starts | `python final/src/make_starts.py` | `final/starts/rounded_B_*.json`, `results/final_lp_rounding.csv` |
| Second run: exact solver, own instance and M = 100, 200, 400 (limit 120 s) | `rm -f results/final_exact.csv; python final/src/run_exact.py --own; for s in 100 200 400; do python final/src/run_exact.py --size $s; done` | `results/final_exact.csv`, `final/starts/exact_B_*.json` |
| Second run: GA, own start (per size) | `python experiments/run_scaling.py --module src.meta.ga --story B --variant 2 --size 100 --team T10 --configuration "GA, repair" --out results/final_GA_repair.csv --append` | `results/final_GA_repair.csv` |
| Second run: GA, LP-rounded start (per size) | same with `--configuration "GA, repair, LP-rounded start" --start-file final/starts/rounded_B_100.json` | `results/final_GA_repair.csv` |
| Second run: SA, own start (per size) | `python experiments/run_scaling.py --module src.meta.sa --story B --variant 2 --size 100 --team T10 --configuration "SA, transfer, own start" --params '{"T0": 1.0570236966839537, "Tend": 0.010473720352671537}' --out results/final_SA_scaling.csv --append` | `results/final_SA_scaling.csv` |
| Second run: SA, LP-rounded start (per size) | same with `--configuration "SA, transfer, LP-rounded start" --start-file final/starts/rounded_B_100.json` | `results/final_SA_scaling.csv` |
| Traces for Figure F1 and final solutions (reruns of the CP4 final runs) | `python final/src/traces.py --module src.meta.ga --name GA; python final/src/traces.py --module src.meta.sa --name SA --params '{"T0": 1.0570236966839537, "Tend": 0.010473720352671537}'; python final/src/traces.py --module src.meta.random_search --name random` | `results/final_trace_*.csv`, `results/final_solutions_*.json` |
| Block 1 (Telgarin): three solutions under the CP1 objective, second solve, optimal set | `python final/src/final_block1.py` (options in the header of the file; the optimal-set MILPs take 2 x 600 s) | `results/final_block1_eval.csv`, `final_block1_decomposition.csv`, `final_block1_solutions.json`, `final_block1_reference_resolve.json`, `final_block1_optimal_set.json` |
| Block 1, table of the report (re-evaluates all stored placements with the checker) | `python final/src/block1.py` | `results/final_block1.csv` |
| Block 2 (Telgarin): statistics, SA files, neighbourhood scan of all final solutions | `python final/src/final_stats.py; python final/src/final_sa_files.py; python final/src/extra_scan.py` | `results/final_stats.json`, `final_table2_check.csv`, `final_sa_files.json`, `final_sa_table.csv`, `extra_scan_all_seeds.json` |
| Block 3, run of Table 3 (Hafizi): exact solver, GA and SA from both starts, M = 100, 200, 400 | scaling script of Hafizi in `final/src/` (command in its header) | `results/final_scaling_exact.csv`, `final_scaling_lp.csv`, `final_scaling_methods.csv`, `final_LP_rounded_B2_*.json`, `final_table3.csv`, `final_environment.json`, `run_manifest.json` |
| SA start check (Block 3) | `python final/src/sa_start_check.py` | `results/final_SA_start_check.csv` |
| Tables 2, 3 and Figures F1, F2 | `python final/src/make_tables.py` | `results/final_table2.csv`, `results/final_table3_report.csv`, `report/final_table*.tex`, `report/final_F1.pdf`, `report/final_F2.pdf` |
| PDF | `cd report && pdflatex T10_Final.tex && pdflatex T10_Final.tex` | `report/T10_Final.pdf` |

Replace 100 by 200 and 400 for the other sizes. Block 3 was run twice (two machines, two rounding rules): Table 3 of the report is the run of Hafizi, the second run gives the best known values and a check. `final/src/final_block2.py` is the library of `extra_scan.py`; if it is run on its own it rewrites `results/final_table2.csv` in another layout, so run `make_tables.py` after it. Costs of the approximate runs are deterministic (fixed seeds). The exact solver is stopped by a wall-clock limit, so at M = 200 and M = 400 its incumbent, bound and gap can differ a little between machines; times always do.

### Which file of results/ is behind which part of the report

| Part of the report | File in `results/` |
|---|---|
| Block 1, table of solutions | `final_block1.csv` (from `final_block1_solutions.json`, `final_block1_reference_resolve.json`, `final_block1_optimal_set.json`, `final_block1_decomposition.csv`) |
| Table 2 | `final_table2.csv` (from `Telgarin_runs.csv`, `Hafizi_runs.csv`, `Shaken_tuning.csv`, `Telgarin_random.csv`, `final_exact.csv`) |
| Figure F1 | `final_trace_GA.csv`, `final_trace_SA.csv`, `final_trace_random.csv` |
| Block 2, analysis | `analysis.json`, `final_stats.json`, `final_sa_files.json`, `extra_scan_all_seeds.json`, `abdimazhit_exp.csv`, `Shaken_tuning.csv`, `Telgarin_Nexp.csv` |
| Table 3 | `final_table3_report.csv` (from `final_scaling_exact.csv`, `final_scaling_methods.csv`; best known also from `final_exact.csv`, `final_GA_repair.csv`, `final_SA_scaling.csv`) |
| Figure F2 | `final_scaling_exact.csv`, `final_exact.csv` |
| Block 3, analysis | `final_scaling_lp.csv`, `final_SA_start_check.csv`, second run: `final_exact.csv`, `final_lp_rounding.csv` |

### Final - authorship of the files

| Files | Author |
|---|---|
| `final/src/model_b.py`, `run_exact.py`, `make_starts.py`, `traces.py`, `block1.py`, `sa_start_check.py`, `make_tables.py`; `final/starts/`; `results/final_*` except the files below | Abdimazhit |
| `final/src/final_block1.py`, `final_stats.py`, `final_sa_files.py`, `final_block2.py`, `extra_scan.py`; `results/final_block1_eval.csv`, `final_block1_decomposition.csv`, `final_block1_solutions.json`, `final_block1_reference_resolve.json`, `final_block1_optimal_set.json`, `final_stats.json`, `final_table2_check.csv`, `final_sa_files.json`, `final_sa_table.csv`, `extra_scan_all_seeds.json` | Telgarin |
| scaling script of Block 3; `results/final_scaling_*.csv`, `final_LP_rounded_B2_*.json`, `final_table3.csv`, `final_environment.json`, `run_manifest.json`, `final_block3_analysis.md` | Hafizi |
| `src/meta/sa.py`, `results/Hafizi_runs.csv`, `tests/test_sa_i.py`, `experiments/sa_i_analysis.py`, `experiments/sa_i_worst_seed.py` | Hafizi (Checkpoint 4, branch `sa-i-hafizi`, merged into `main` for the Final) |
| `results/Shaken_tuning.csv` | Shaken (Checkpoint 4); committed by Abdimazhit before the deadline |
| `experiments/run_scaling.py`, `experiments/exact_helper.py`, `check_env.py`, generator | Final kit |

### Final - declarations

**Abdimazhit Inkar**

| Tool or source | Used for | Files or sections | What I checked myself |
|---|---|---|---|
| AI assistant (Claude) | merged the results and text of Telgarin and Hafizi into the report; gave me the git commands | `final/src/model_b.py`, `run_exact.py`, `make_starts.py`, `traces.py`, `block1.py`, `sa_start_check.py`, `make_tables.py`; `final/starts/`; `results/final_*` (except the files of Telgarin and Hafizi); `report/T10_Final.tex`; `README.md`, section "Final" | reran `python final/src/run_exact.py --own`: z_LP = 8.339506 and z_ref = 8.382826, proven, equal to Checkpoint 3; reran `python final/src/block1.py` and `python final/src/make_tables.py`: `results/final_block1.csv`, `final_table2.csv`, `final_table3_report.csv` unchanged; checked the numbers of Section 4 (8.3828, 0.23 %, 55.5809, 61.20) against Tables 2 and 3; found with `git shortlog` that the SA branch `sa-i-hafizi` was not merged and merged it into `main`; committed, pushed and tagged the Final myself. Not repeated on my computer: the exact-solver runs at M = 200 and M = 400. |
| Python libraries (`numpy`, `scipy`, `matplotlib`) | LP relaxation and MILP (HiGHS through `scipy.optimize.milp`), arrays, figures | `final/src/` | - |

**Telgarin Diar**

| Tool or Source | Used for | Files or sections | What I checked myself |
|---|---|---|---|
| Telgarin Diar (Lead Author & Developer) + AI Assistant (Claude) | Auditing Block 1 (report & code) against Final_Instructions.pdf, executing scripts, rebuilding reference models, and refining report wording | Block_1_Final.md (report Section 1), final/src/final_block1.py, results/final_block1_eval.csv, results/final_block1_decomposition.csv, results/final_block1_solutions.json, results/final_block1_reference_resolve.json, results/final_block1_optimal_set.json | Independently wrote and structured the core logic and scripts, ran control tests (11.294574, 10.688779, 11.426841), executed --resolve-reference (z_ref = 8.382826, z_LP = 8.339506), verified optimal sets, compared evaluation metrics with Checkpoint 1 results, and validated Section 1 |
| Python libraries (numpy, scipy) | Array operations and MILP modeling using HiGHS (scipy.optimize.milp) | final/src/final_block1.py | — |

**Hafizi Nazifullah**

| Tool or source | Used for | Files or sections | What I checked myself |
|---|---|---|---|
| ChatGPT (OpenAI) | assistance with Python coding, debugging, LP rounding, LaTeX writing and analysis of Block 3 (Scaling) | `final/src/run_block3_hafizi.py`, `final/src/block3_scaled_model_hafizi.py`, `final/src/block3_lp_rounding_hafizi.py`; Section 3 of the final report | I personally ran the scaling experiments for M = 100, 200 and 400 with seeds 0–9, a budget of 100 000 evaluations and a 120-second HiGHS limit; I verified that all 120 heuristic runs completed and reviewed the generated CSV results. |

**Shaken Daniyal**

| Tool or source | Used for | Files or sections | What I checked myself |
|---|---|---|---|
| AI assistant (Claude) | drafted the text of Block 1 (Table 1, own objective on the reference solutions, conclusion); statistics of the SA files of Checkpoint 4 (SA tuned against rule, random baseline of the SA pair) | report Section 1; `final/src/final_sa_files.py` | compared Table 1 with my Checkpoint 1 report; ran `python final/src/final_block1.py` and `python final/src/final_sa_files.py` and compared the results with `results/final_block1_eval.csv` and `results/Hafizi_runs.csv`; read Section 1 completely |
| Python libraries (`numpy`, `scipy`) | evaluation of the objectives, HiGHS (`scipy.optimize.milp`), statistical tests | `final/src/final_block1.py`, `final/src/final_sa_files.py` | - |

---

# Checkpoint 4 — team T10 — GA, repair (pair: GA-A Diar Telgarin, GA-P Inkar Abdimazhit)

Instance: `cp34_instance(10, 'B', 2)` (storyline B, variant 2: S = 11 servers, M = 50 VMs, zones). Budget 100 000 evaluations, final seeds 0–9.

## Roles and files
| File | Author |
|---|---|
| `src/meta/ga.py` (loop: tournament, elitism, generations, budget; `solve`) | Telgarin (GA-A) |
| `src/meta/random_search.py` (baseline) | Telgarin (GA-A) |
| `tests/test_ga_a.py` | Telgarin (GA-A) |
| `experiments/analysis.py`, `experiments/make_report.py`, `make_all.sh`, `results/Telgarin_*` | Telgarin (GA-A) |
| `src/meta/ga_problem.py` (representation, evaluate, crossover, mutate, repair) | Abdimazhit (GA-P) |
| `tests/test_GA-P.py` | Abdimazhit (GA-P) |
| `experiments/exp_repair.py`| Abdimazhit (GA-P) |
| `results/abdimazhit_exp.csv`| Abdimazhit (GA-P) |
| `report/prediction_GA-P.txt`| Abdimazhit (GA-P) |
| `src/meta/interface.py`, `experiments/run_final.py`, `check_env.py`, generator | starter kit |

## Agreed interface (GA-P ⇄ GA-A); individual = numpy int array `ind`, `ind[k]` = server of VM k
- `random_individual(d, rng, repair=True)` → random assignment among the allowed servers; repaired if `repair=True`, raw if `False`
- `evaluate(d, ind)` → `(fitness, cost, feasible)`; fitness = cost + RHO·violation (lower is better)
- `crossover(d, p1, p2, rng)` → one child (uniform crossover); `mutate(d, ind, rng, n)` → copy with n components moved
- `repair(d, ind, rng)` → feasible individual; reads only loads and capacities (free of charge)
- `violation(d, ind)` → total violation (0 for feasible); ga.py computes fitness = cost + rho·violation itself from `evaluate`'s cost and this function
All random numbers come from the `numpy.random.default_rng(seed)` created in `solve`.

## Configuration (GA, repair; rule parameters, not tuned)
N = 30, p_c = 0.9, 1 or 2 mutated components, tournament size 2, elitism 1. Counting: every offspring and every initial individual = 1 evaluation,
elite copy free, last generation cut short (exactly 100 000). Repair rule: (1) VM on a forbidden zone → random allowed server; (2) while a server is overloaded
(CPU or memory): the smallest VM that cures the overload (else the largest) moves to the allowed server with room and the lowest CPU utilisation.
Parameters of `solve` (`--params` of run_final): `repair` (True/False, default True), `rho` (default 100; used for fitness = cost + rho·violation). With `repair=False` `gp.repair` is never called (initial population, offspring and `start`). Extra entries of the result: `init_individuals`, `init_feasible`, `offspring`, `offspring_feasible`, `offspring_feasible_share`, `repair_changed`.
Example (GA-P experiment): `python experiments/run_final.py --module src.meta.ga --instance-seed 10 --story B --variant 2 --team T10 --role GA-P --configuration "GA, no repair, rho=10" --params '{"repair": false, "rho": 10}' --out results/Abdimazhit_exp.csv`
Random search: same repaired random individuals, same budget and seeds. `solve` also returns the extra entries `distinct`, `hamming`, `generations`.

## Reproduce everything
    bash make_all.sh          # env check, tests, final runs, baseline, N experiment, analysis, figures, PDF (report/Telgarin_CP4.pdf)
Files: `results/Telgarin_runs.csv` (Table 1, F1), `results/Telgarin_random.csv` (baseline), `results/Telgarin_Nexp.csv` (N = 10, 30, 100),
`results/ref_values.json` (z_LP, z_ref — must equal the values of the team task file), `results/analysis.json` (all numbers of the PDF: Table 2 etc.).
Run the test alone: `python -m unittest tests.test_ga_a -v`.

**Author:** Telgarin Diar

This project was developed with a strict focus on independent algorithmic design and manual implementation. AI tools were utilized exclusively in an assistive capacity - primarily for generating boilerplate code, syntax reference, and structuring drafts. The core mathematical logic, architectural decisions, code refactoring, and experimental verification are entirely my own work. 

| Tool or source | Used for | Files or sections | My primary contribution & verification |
| :--- | :--- | :--- | :--- |
| **AI assistant (Claude)** | Prototyping boilerplate code, syntax reference, and brainstorming structural outlines for the report | `src/meta/ga.py`, `tests/test_ga_a.py`, `experiments/`, `report` | **Lead Author & Reviewer:** Designed the algorithm architecture and logic. Refactored AI-generated prototypes into the final working code. Independently executed all test cases, reproduced the 10-run CSV results, validated Table 2 against the checker, and authored the final report entirely in my own words. |
| **Python Libraries** (`numpy`, `matplotlib`, `reportlab`, `scipy`) | Vectorized matrix operations, data visualization, PDF formatting, computing reference baselines | `experiments/`, `results/ref_values.json` | Selected and integrated necessary libraries. *Note:* `scipy` was restricted strictly to calculating reference values ($z_{LP}$, $z_{ref}$) and was purposefully isolated from the core solver logic. |

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
| AI assistant(Claude) | help with debugging and code implemetnting | src/meta/ga_problem.py, tests/test_GA-P.py | ran the role test and the test of the partner, ran the whole experiment myself (20 runs) and compared the results with a reference run; |



### Hafizi Nazifullah

| Tool or source | Used for | Files or sections | What I checked myself |
|---|---|---|---|
| ChatGPT (OpenAI) | Assistance with Python coding, debugging, LP rounding, LaTeX writing, and analysis of Block 3 (Scaling). | `final/src/run_block3_hafizi.py`, `final/src/block3_scaled_model_hafizi.py`, `final/src/block3_lp_rounding_hafizi.py`, Section 3 of the final report. | I personally ran the scaling experiments for M = 100, 200, and 400 using seeds 0–9, a budget of 100,000 evaluations, and a 120-second HiGHS limit. I verified that all 120 heuristic runs completed and reviewed the generated CSV results. |
