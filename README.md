# Checkpoint 3 — Cloud Resource Allocation

## 1. Project Overview

This project implements and analyzes the reference optimization model for **Team T10, Storyline B — Cloud Resource Allocation, Variant 2**.

The project focuses on constructing the model, solving its linear programming (LP) relaxation, checking optimality conditions, analyzing dual prices, and studying the relationship between the LP bound and integer solutions.

## 2. Team Information

- **Team:** T10
- **Storyline:** B — Cloud Resource Allocation
- **Variant:** 2
- **Random seed:** 10

### Team Roles

| Member | Role | Responsibility |
|---|---|---|
| Inkar Abdimazhit | M1 — Model | Model construction, data, matrices, row/column bookkeeping, Task 1, S1, H1 |
| Diar Telgarin | M2 — LP | LP solver and solution analysis, Task 2, S2, H2 |
| Daniyal Shaken | M3 — KKT | Numerical optimality-condition checker, Task 3, S3, H3 |
| Nazifullah Hafizi | M4 — Sensitivity | RHS perturbations, dual-price analysis, Task 4, S4, H4 |

**Joint team work:** Task 5, S5, H5, experiment integration, result verification, and final report preparation.

## 3. Instance Configuration

The instance is generated using the provided `cp34_instance` function:

```python
d = cp34_instance(10, "B", 2)
```

Expected instance checks:

| Property | Expected value |
|---|---:|
| Physical servers | 11 |
| Virtual machines | 50 |
| Total CPU capacity | 504 |
| Total memory capacity | 1725 |
| Total CPU requirement | 233 |
| Total memory requirement | 871 |
| Forbidden assignments | 51 |
| Number of variables | 561 |
| Number of constraint rows | 116 |

These values are used to verify that the correct instance and reference formulation are implemented.

## 4. Project Structure

```text
Checkpoint2/
├── src/
│   ├── model/
│   │   └── model_builder.py
│   ├── lp/
│   ├── kkt/
│   └── sens/
├── experiments/
│   └── run_all.py
├── tests/
│   ├── test_model.py
│   ├── test_m1.py
│   ├── test_m2.py
│   ├── test_m3.py
│   └── test_m4.py
├── results/
├── checkpoint3_generator_PUBLIC.py
├── README.md
└── .gitignore
```

The structure above describes the intended project organization. Files should be listed here only when they exist in the repository.

## 5. Requirements and Setup

Python 3 and the following libraries are used:

- NumPy
- SciPy
- PuLP below version 4

Install the required packages:

```bash
python -m pip install numpy scipy "pulp<4"
```

Run the model tests:

```bash
python -m unittest discover -s tests -v
```

When the experiment runner is implemented, generate the report results using:

```bash
python -m experiments.run_all
```

The experiment runner is intended to reproduce the numerical results and Tables 1–3. The final results should be checked against the reference values before submission.

## 6. Expected Outputs

The completed project should provide:

- The reference model and its constraint matrices.
- The LP relaxation and its optimal solution.
- Numerical checks of the required optimality conditions.
- Sensitivity results for the designated constraint.
- CSV files containing reproducible experiment results.
- Tests reproducing the calculations from each member's hand task.
- The final report, `T10_Checkpoint3.pdf`.

## 7. Git Contribution Rules

Each member is responsible for authoring and committing the code associated with their role.

Commit messages should begin with the relevant prefix:

- `[Model]` — model implementation and tests.
- `[LP]` — LP solver and solution analysis.
- `[KKT]` — optimality-condition checks.
- `[Sens]` — sensitivity analysis.
- `[Report]` — report, experiment integration, and documentation.

The final submission must include the `checkpoint3` Git tag pointing to the final team commit.

## 8. Declarations

### AI Assistance

ChatGPT was used as an assistance tool for explaining assignment requirements, discussing code structure, troubleshooting implementation issues, and helping prepare documentation and test code.

The team is responsible for reviewing, understanding, testing, and verifying all submitted code, calculations, and written analysis. Any AI-assisted content has been checked by the team before submission.

### External Code and Data

The instance is generated using the course-provided `checkpoint3_generator_PUBLIC.py` file and the assigned parameters. Any additional external code, sources, or assistance used by the team must be documented here before submission.

No results should be reported as verified unless they have been reproduced by the project's code.

