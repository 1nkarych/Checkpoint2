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
| `src/meta/interface.py`, `experiments/run_final.py`, `check_env.py`, generator | starter kit |

## Agreed interface (GA-P ⇄ GA-A); individual = numpy int array `ind`, `ind[k]` = server of VM k
- `random_individual(d, rng)` → feasible individual (random assignment + repair)
- `evaluate(d, ind)` → `(fitness, cost, feasible)`; fitness = cost + RHO·violation (lower is better)
- `crossover(d, p1, p2, rng)` → one child (uniform crossover); `mutate(d, ind, rng, n)` → copy with n components moved
- `repair(d, ind, rng)` → feasible individual; reads only loads and capacities (free of charge)
All random numbers come from the `numpy.random.default_rng(seed)` created in `solve`.

## Configuration (GA, repair; rule parameters, not tuned)
N = 30, p_c = 0.9, 1 or 2 mutated components, tournament size 2, elitism 1. Counting: every offspring and every initial individual = 1 evaluation,
elite copy free, last generation cut short (exactly 100 000). Repair rule: (1) VM on a forbidden zone → random allowed server; (2) while a server is overloaded
(CPU or memory): the smallest VM that cures the overload (else the largest) moves to the allowed server with room and the lowest CPU utilisation.
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