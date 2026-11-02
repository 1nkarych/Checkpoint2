# M3 — KKT checker (Checkpoint 3, team T10, storyline B2, seed 10)

Role: **M3**, Daniyal Shaken. Task 3 (optimality conditions), section S3, hand task H3.

## What is here
- `src/kkt/checker.py` — `check_kkt(c, A, b, sense, lb, ub, x, pi)` returns the five numbers (a)–(e) of Task 3.
  It takes plain arrays, so it accepts the solution of any solver (sign convention: `pi_i = dz/db_i`,
  `<=` rows `pi <= 0`, `>=` rows `pi >= 0`, `=` rows free; reduced cost `r = c - A^T pi`).
  Variables with `lb == ub` (forbidden pairs fixed to 0) have no reduced-cost condition.
- `tests/test_M3.py` — reproduces the hand numbers of H3 (triangle problem, c = (6,7,5), pi = (4,3,2) and pi' = (5,2,1))
  to 1e-6, and tests the sign convention on small LPs with a known answer.
- `S3.md` — signed analysis section S3.

## How to run
```
pip install -r requirements_m3.txt
python -m pytest tests/test_M3.py
```

## Declarations (AI help)
`src/kkt/checker.py`, `tests/test_M3.py` and the first draft of `S3.md` were written with the help of Claude
(Anthropic's AI assistant) in a chat session: Claude produced the code and the draft text; the numbers in S3 come from
an independent run of the reference model (HiGHS via scipy, and CBC via PuLP 3.3.2) that Claude executed.
I read the code, ran the tests, and I am responsible for the content I sign. The hand task H3 is solved on paper (see the PDF appendix).
