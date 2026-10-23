# M2 — LP relaxation: solving and reading the solution

**Checkpoint 3 · team T10 · storyline B2 · seed 10**

| | |
|---|---|
| **Role** | M2 — Diar Telgarin |
| **Task** | Task 2 (the LP relaxation) |
| **Section** | S2 — Value and fractionality of the LP |
| **Hand task** | H2 |

---

## Files

| File | What it does |
|---|---|
| `src/lp/solve.py` | `solve_lp(data)` solves an LP in matrix form with HiGHS (`scipy.optimize.linprog`) and returns z_LP, x, dual prices, reduced costs, solver name and time |
| `src/lp/fractional.py` | Fractional `y` entries (strictly between 0 and 1, tolerance 1e-6) and VMs split across several servers |
| `src/lp/summary.py` | Table 1 numbers as a dict and CSV |
| `src/lp/run_lp.py` | Runs the LP for the team instance and writes `results/table1_lp.csv` |
| `tests/test_m2.py` | Reproduces the hand numbers of H2 and checks the dual sign convention |
| `tools/standin_model_B2.py` | Temporary stand-in for M1's `src/model`, used only to run M2's code alone |
| `results/table1_lp.csv` | z_LP, solver, time, fractional entries |
| `S2.md` | Signed analysis section S2 |

## Sign convention

Dual prices follow the handout: `pi_i = dz/db_i`.

- `<=` rows: `pi <= 0`
- `=` rows: `pi` free
- reduced cost: `r = c - A^T pi`

The convention was checked on a two-variable LP with a known answer (see `tests/test_m2.py`).

## How to run

```powershell
python -m pip install -r requirements_m2.txt
python -m pytest tests/test_m2.py
python src/lp/run_lp.py
```

`python tests/test_m2.py` also works if pytest is not installed.

## Results

| Quantity | Value |
|---|---|
| z_LP | **8.339506** |
| Solver | HiGHS via `scipy.optimize.linprog` |
| Fractional `y_kl` | 21 of 499 free entries |
| VMs split across servers | 9 of 50 |

The count of fractional entries depends on the solver, z_LP does not.

H2 (triangle problem, c = (6, 7, 5)): LP optimum (1/2, 1/2, 1/2), z_LP = 9, z* = 11, relative gap 2/11.

## Declarations (AI help)

`src/lp/` (`solve.py`, `fractional.py`, `summary.py`, `run_lp.py`), `tools/standin_model_B2.py`, `tests/test_m2.py` and the first draft of `S2.md` were written with the help of Claude (Anthropic's AI assistant) in a chat session.

Claude produced the code and the draft text. The numbers come from a run of the reference model (HiGHS via scipy, cross-checked with HiGHS dual simplex / interior point and CBC via PuLP 3.3.2) that Claude executed.

I read the code, ran the tests, and I am responsible for the content I sign. The hand task H2 is solved on paper (see the PDF appendix).
