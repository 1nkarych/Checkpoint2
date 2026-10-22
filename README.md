# Checkpoint 3 — Team T10 (Introduction to Optimization)

**Storyline:** B — Cloud Resource Allocation  
**Variant:** B2  
**Seed:** 10

## Project Structure

- `checkpoint3_generator_PUBLIC.py` — course-provided instance generator.
- `src/model/model_builder.py` — constructs the linear programming model in matrix form.
- `tests/test_model.py` — tests the model dimensions, constraints, and forbidden assignments.
- `src/lp/` — linear programming implementation.
- `src/kkt/` — KKT analysis.
- `src/sens/` — sensitivity analysis.
- `experiments/` — experiment scripts.
- `results/` — generated results.

## Run the Model

From the project root:

```powershell
python src/model/model_builder.py
```

## Run Tests

```powershell
python -m unittest discover -s tests -v
```

## AI Assistance Disclosure

AI assistance was used to help develop and review the M1 matrix formulation and its tests. The team should review, understand, and verify all generated code and explanations before submission.