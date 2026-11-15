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
| Block 1 (own CP1 objective on the reference solutions) | `python final/src/block1.py` | `results/final_block1.csv` |
| SA start check (Block 3) | `python final/src/sa_start_check.py` | `results/final_SA_start_check.csv` |
| Tables 2, 3 and Figures F1, F2 | `python final/src/make_tables.py` | `results/final_table2.csv`, `results/final_table3.csv`, `report/final_table*.tex`, `report/final_F1.pdf`, `report/final_F2.pdf` |
| PDF | `cd report && pdflatex T10_Final.tex && pdflatex T10_Final.tex` | `report/T10_Final.pdf` |

Replace 100 by 200 and 400 for the other sizes. Costs of the approximate runs are deterministic (fixed seeds). The exact solver is stopped by a wall-clock limit, so at M = 200 and M = 400 its incumbent, bound and gap can differ a little between machines; times always do.

### Final - declarations

One table per author (fill in truthfully; format of Checkpoint 4).

**<Name>**

| Tool or source | Used for | Files or sections | What I checked myself |
|---|---|---|---|
| AI assistant (Claude) | <e.g. scripts of final/src, running the scaling experiment, draft of the report text> | <files / report sections> | <what you ran, read and can explain> |
