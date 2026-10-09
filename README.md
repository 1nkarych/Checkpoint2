# Checkpoint 2 — Team T10 (Introduction to Optimization)

Storyline B: balancing CPU load (softmax-parametrized quadratic latency), seed 10, variant 2.

## Files
- `checkpoint2_generator_PUBLIC.py` — public parameter generator (course-provided)
- `checkpoint2_solution.py` — f, grad, hess; GD / momentum / Adam / Newton / damped Newton; checks; writes `results_tables.tex`
- `report.tex` — LaTeX report (compile with `pdflatex report.tex`, twice if needed)
- `hand_traces.pdf` — signed scans of hand traces H1–H5 (appendix)

## Run
```
pip install -r requirements.txt
python checkpoint2_solution.py      # needs the Checkpoint 1 module with cloud_variant(seed, variant)
pdflatex report.tex
```
Edit the import in `load_data()` if your Checkpoint 1 module is not called `checkpoint1`.
If the module is missing, the script runs on DEMO DATA and prints a warning — do not use those numbers.
