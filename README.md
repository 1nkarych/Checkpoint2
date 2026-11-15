# Team T10 - Introduction to Optimization (storyline B, variant 2)

Members: Abdimazhit Inkar, Hafizi Nazifullah, Shaken Daniyal, Telgarin Diar. Instance: `cp34_instance(10, 'B', 2)`.
Contents: [Final](#final) (first), then the README of Checkpoint 4 (unchanged).

## Final

Team T10, storyline B, variant 2, instance `cp34_instance(10, 'B', 2)`; scaled instances `scaled_instance('B', M, 2)`, M = 100, 200, 400.
Needs Python 3 with numpy, scipy >= 1.9, matplotlib (no PuLP: the model is built as matrices for `scipy.optimize.milp`, HiGHS).
Run everything from the repository root.

| Step | Command | Writes |
|---|---|---|
| LP relaxation and LP-rounded starts | `python final/src/make_starts.py` | `final/starts/rounded_B_*.json`, `results/final_lp_rounding.csv` |
| Exact solver, own instance and M = 100, 200, 400 (limit 120 s) | `rm -f results/final_exact.csv; python final/src/run_exact.py --own; for s in 100 200 400; do python final/src/run_exact.py --size $s; done` | `results/final_exact.csv`, `final/starts/exact_B_*.json` |
| GA, own start (per size) | `python experiments/run_scaling.py --module src.meta.ga --story B --variant 2 --size 100 --team T10 --configuration "GA, repair" --out results/final_GA_repair.csv --append` | `results/final_GA_repair.csv` |
| GA, LP-rounded start (per size) | same with `--configuration "GA, repair, LP-rounded start" --start-file final/starts/rounded_B_100.json` | `results/final_GA_repair.csv` |
| SA, own start (per size) | `python experiments/run_scaling.py --module src.meta.sa --story B --variant 2 --size 100 --team T10 --configuration "SA, transfer, own start" --params '{"T0": 1.0570236966839537, "Tend": 0.010473720352671537}' --out results/final_SA_scaling.csv --append` | `results/final_SA_scaling.csv` |
| SA, LP-rounded start (per size) | same with `--configuration "SA, transfer, LP-rounded start" --start-file final/starts/rounded_B_100.json` | `results/final_SA_scaling.csv` |
| Traces for Figure F1 and final solutions (reruns of the CP4 final runs) | `python final/src/traces.py --module src.meta.ga --name GA; python final/src/traces.py --module src.meta.sa --name SA --params '{"T0": 1.0570236966839537, "Tend": 0.010473720352671537}'; python final/src/traces.py --module src.meta.random_search --name random` | `results/final_trace_*.csv`, `results/final_solutions_*.json` |
| Block 1, three solutions under the CP1 objective (Telgarin) | `python final/src/final_block1.py` | `results/final_block1_eval.csv`, `results/final_block1_solutions.json` |
| Block 1, table of the report (both optimal placements, GA, SA, descents) | `python final/src/block1.py` | `results/final_block1.csv` |
| Block 2, neighbourhood scan of all final solutions (Telgarin; imports `final_block2.py`) | `python final/src/extra_scan.py` | `results/extra_scan_all_seeds.json` |
| SA start check (Block 3) | `python final/src/sa_start_check.py` | `results/final_SA_start_check.csv` |
| Tables 2, 3 and Figures F1, F2 | `python final/src/make_tables.py` | `results/final_table2.csv`, `results/final_table3.csv`, `report/final_table*.tex`, `report/final_F1.pdf`, `report/final_F2.pdf` |
| PDF | `cd report && pdflatex T10_Final.tex && pdflatex T10_Final.tex` | `report/T10_Final.pdf` |

Replace 100 by 200 and 400 for the other sizes. `final/src/final_block2.py` is the library of `extra_scan.py`; if it is run on its own it rewrites `results/final_table2.csv` in another layout, so run `make_tables.py` after it. Costs of the approximate runs are deterministic (fixed seeds). The exact solver is stopped by a wall-clock limit, so at M = 200 and M = 400 its incumbent, bound and gap can differ a little between machines; times always do.

### Which file of results/ is behind which part of the report

| Part of the report | File in `results/` |
|---|---|
| Block 1, table of solutions | `final_block1.csv`, `final_block1_eval.csv` |
| Table 2 | `final_table2.csv` (from `Telgarin_runs.csv`, `Hafizi_runs.csv`, `Shaken_tuning.csv`, `Telgarin_random.csv`, `final_exact.csv`) |
| Figure F1 | `final_trace_GA.csv`, `final_trace_SA.csv`, `final_trace_random.csv` |
| Block 2, analysis | `analysis.json`, `extra_scan_all_seeds.json`, `abdimazhit_exp.csv`, `Shaken_tuning.csv`, `Telgarin_Nexp.csv` |
| Table 3 | `final_table3.csv` (from `final_exact.csv`, `final_GA_repair.csv`, `final_SA_scaling.csv`) |
| Figure F2 | `final_exact.csv` |
| Block 3, analysis | `final_lp_rounding.csv`, `final_SA_start_check.csv` |

### Final - authorship of the files

| Files | Author |
|---|---|
| `final/src/model_b.py`, `run_exact.py`, `make_starts.py`, `traces.py`, `block1.py`, `sa_start_check.py`, `make_tables.py`; `final/starts/`; `results/final_*` except the files below | Abdimazhit |
| `final/src/final_block1.py`, `cp1_eval.py`, `final_block2.py`, `extra_scan.py`; `results/final_block1_eval.csv`, `final_block1_solutions.json`, `extra_scan_all_seeds.json` | Telgarin |
| `src/meta/sa.py`, `results/Hafizi_runs.csv`, `results/Shaken_tuning.csv` | Hafizi, Shaken (Checkpoint 4, SA pair); committed by Abdimazhit before the deadline |
| `experiments/run_scaling.py`, `experiments/exact_helper.py`, `check_env.py`, generator | Final kit |

### Final - declarations

**Abdimazhit Inkar**

| Tool or source | Used for | Files or sections | What I checked myself |
|---|---|---|---|
| AI assistant (Claude) | wrote the scripts of the Final listed under my name; ran the exact solver, the GA and SA scaling runs and the trace reruns in its own environment; built the tables and figures; drafted the text of the report and of this README section | `final/src/model_b.py`, `run_exact.py`, `make_starts.py`, `traces.py`, `block1.py`, `sa_start_check.py`, `make_tables.py`; `final/starts/`; `results/final_*`; `report/T10_Final.tex` | reran `python final/src/make_tables.py` on my computer: `results/final_table2.csv` and `results/final_table3.csv` came out unchanged; committed and pushed the files myself and checked the repository state with `git status` / `git log`; read the report and edited Section 4 (memo). The exact-solver runs at M = 200 and M = 400 were not repeated on my computer (their result depends on the machine, see the note above). |
| Python libraries (`numpy`, `scipy`, `matplotlib`) | LP relaxation and MILP (HiGHS through `scipy.optimize`), arrays, figures | `final/src/` | - |

**Telgarin Diar**

| Tool or source | Used for | Files or sections | What I checked myself |
|---|---|---|---|
| FILL IN | | | |

**Hafizi Nazifullah**

| Tool or source | Used for | Files or sections | What I checked myself |
|---|---|---|---|
| FILL IN | | | |

**Shaken Daniyal**

| Tool or source | Used for | Files or sections | What I checked myself |
|---|---|---|---|
| FILL IN | | | |

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