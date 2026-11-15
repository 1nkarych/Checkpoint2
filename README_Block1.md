# Block 1 — from the Checkpoint 1 model to the reference model (T10, cp34_instance(10,"B",2))

Report: `Block_1_Final.md`. Code: `final/src/final_block1.py`. Data: `checkpoint4_generator_PUBLIC.py`. Results: `results/final_block1_*`.
Needs Python 3, numpy, scipy >= 1.9 (tested with scipy 1.18.1). Run from this folder.

```
python final/src/final_block1.py                                  # ~1 s: 3 stored solutions under the reference cost and the CP1 objective (exit 1 if control values differ by > 5e-7)
python final/src/final_block1.py --resolve-reference              # ~12 s: z_ref, z_LP, second optimal assignment (CP1 value 10.966209)
python final/src/final_block1.py --optimal-set --time-limit 600   # ~20 min, not re-run; stored result checked by evaluating the two stored assignments
```
`--seeds` needs the team GA/SA code (it is in the Block 2 package) and is not part of this package.
Limits: the Checkpoint 1 column of Table 1 was not compared with the Checkpoint 1 PDF; the author of that model is not named.
`legacy/` holds earlier versions of the script (history only).

<Telgarin Diar - AI Declaration> 
Tool or Source: Telgarin Diar (Lead Author & Developer) + AI Assistant (Claude)
Used for: Auditing Block 1 (report & code) against Final_Instructions.pdf, executing scripts, rebuilding reference models, and refining report wording.
Files or sections: Block_1_Final.md (report Section 1), final/src/final_block1.py, results/final_block1_eval.csv, results/final_block1_decomposition.csv, results/final_block1_solutions.json, results/final_block1_reference_resolve.json, results/final_block1_optimal_set.json
What I checked myself: Independently wrote and structured the core logic and scripts, ran control tests (11.294574, 10.688779, 11.426841), executed --resolve-reference (z_ref = 8.382826, z_LP = 8.339506), verified optimal sets, compared evaluation metrics with Checkpoint 1 results, and validated Section 1.

Tool or Source: Python libraries (numpy, scipy)
Used for: Array operations, MILP modeling with HiGHS (scipy.optimize.milp)
Files or sections: final/src/final_block1.py
What I checked myself: —

