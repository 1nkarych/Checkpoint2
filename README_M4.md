# Team T10 - Checkpoint 3 - M4

This package contains M4's sensitivity calculation, H4 test and S4 text.
It does not replace the other members' work or the team's final PDF.

## Files

- `src/sens/sensitivity.py`: B2 reference LP and three perturbation solves for B1, VM k=2 (zero-based indexing).
- `tests/test_M4.py`: four independent checks of H4 using the team costs (6,7,5).
- `checkpoint3_generator_PUBLIC.py`: unchanged instructor-supplied generator; keep it at the project root.
- `results/table3.csv`: left slope, original equality marginal, right slope.
- `results/sensitivity_runs.csv`: all three objective values, measured solve times and primal residuals.
- `S4.md`: report text to review, sign and include in the team's PDF.
- `requirements_m4.txt`: Python dependencies.

## Run in Windows / VS Code

Open the folder containing this README in VS Code. Extract the ZIP contents into your project folder; the `src` and `tests` folders should be directly inside it. Do not put this entire package inside `src`.

In Terminal > New Terminal, run these commands separately:

```powershell
python -m pip install -r requirements_m4.txt
python tests/test_M4.py
python src/sens/sensitivity.py
```

The test runs directly; pytest is not required. If pytest is already installed, it can also discover the same four tests with `python -m pytest tests/test_M4.py -v`.

The sensitivity script creates/updates the two CSV files automatically. Results are computed by the solver; they are not hard-coded in the analysis. Timings vary across computers.

## Expected results

- Instance: `cp34_instance(10, 'B', 2)`.
- Check values: 11 servers, 50 VMs, 561 variables, 116 constraint rows, 51 forbidden pairs fixed at zero.
- B1 RHS 0.99: zLP = 8.334568.
- B1 RHS 1.00: zLP = 8.339506.
- B1 RHS 1.01: zLP = 8.344444.
- Table 3: left slope = dual price = right slope = 0.493827 (six decimals).
- H4: z(1) = 9, slope = 4, formula valid for epsilon in [-1,1]; for increases only, [0,1].

The H4 price 4 belongs to the small triangle problem. It is different from the main B2 model's price 0.493827.

Validation environment: Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0. Both scripts passed there. Run both scripts on your own computer before committing.

## Group integration

A standalone reference assembly is included so M4 can run before M1/M2's files are available. It preserves the exact reference formulation: all forbidden variables remain with upper bound zero, all other y bounds are [0,1], t bounds are [0,infinity), and the four original tangent lines are used.

M5 can call the following from the project-root `experiments/run_all.py`:

```python
from src.sens.sensitivity import run_sensitivity, save_results

summary = run_sensitivity()
save_results(summary)
```

To use M1's matrices instead, call `run_sensitivity(lp=team_model, target_row=vm2_row_index)`. The dictionary must contain `c`, `A_eq`, `b_eq`, `A_ub`, `b_ub`, `bounds`, with inequalities represented as <= rows. Set `vm2_row_index` to the B1 row for VM k=2 in M1's equality ordering. The input matrices and right-hand sides are not modified. M4 reads the marginal from SciPy/HiGHS directly. A separate two-variable LP checks the equality-price sign convention.

Do not overwrite M1/M2's files when merging. Keep the existing group README and merge the relevant role/declaration notes into it. The course requests the existing private group repository.

## Report and commits

Include Table 3 and the signed S4 text in the team's PDF, and the signed H4 hand page in its appendix. Add your row to the group's signed contribution statement. The report assembler handles the final layout and page limit.

Commit your files with your own GitHub account. Use `[Sens]` for sensitivity code/results and `[Report]` for report changes, as required by the handout. Ensure M5 has incorporated the call into `experiments/run_all.py`; M4 alone does not regenerate Tables 1 and 2.

## Declarations

The public instance generator is supplied by the instructor and is included unchanged. ChatGPT assisted with drafting the M4 code, H4 checks and S4 text and with executing verification in a separate environment. Review and understand the code, run it locally, and disclose this assistance in the Declarations section of the group's README as required by the handout. Do not claim the generated code was written without assistance.
