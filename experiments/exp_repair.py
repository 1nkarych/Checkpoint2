import csv
import os
import sys
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from checkpoint4_generator_PUBLIC import cp34_instance, evaluate_B
from src.meta import ga_problem as gp

INSTANCE_SEED = 10
SEEDS = range(100, 105)
N_OFFSPRING = 1000

d = cp34_instance(INSTANCE_SEED, "B", 2)

os.makedirs("results", exist_ok=True)

columns = [
    "team", "role", "seed", "offspring",
    "feasible_before", "feasible_after",
    "share_before", "share_after",
    "best_cost_before", "best_cost_after",
    "repair_failures", "seconds"
]

rows = []

print("GA-P REPAIR EXPERIMENT")
print("Instance: T10, Story B, Variant B2")
print("Offspring per seed:", N_OFFSPRING)
print()

for seed in SEEDS:
    rng = np.random.default_rng(seed)
    before_count = 0
    after_count = 0
    failures = 0
    best_before = float("inf")
    best_after = float("inf")

    start_time = time.time()

    for i in range(N_OFFSPRING):
        # Generate two feasible parents
        p1 = gp.random_individual(d, rng)
        p2 = gp.random_individual(d, rng)

        # Create offspring
        child = gp.crossover(d, p1, p2, rng)
        n_mutations = int(rng.integers(1, 3))
        child = gp.mutate(d, child, rng, n_mutations)

        # Evaluate BEFORE repair using the official checker
        cost_before, problems_before = evaluate_B(d, child)
        feasible_before = len(problems_before) == 0

        if feasible_before:
            before_count += 1
            best_before = min(best_before, cost_before)

        # Apply repair
        repaired = gp.repair(d, child, rng)

        # Evaluate AFTER repair using the official checker
        cost_after, problems_after = evaluate_B(d, repaired)
        feasible_after = len(problems_after) == 0

        if feasible_after:
            after_count += 1
            best_after = min(best_after, cost_after)
        else:
            failures += 1

    elapsed = time.time() - start_time

    share_before = before_count / N_OFFSPRING
    share_after = after_count / N_OFFSPRING

    row = [
        "T10", "GA-P", seed, N_OFFSPRING,
        before_count, after_count,
        f"{share_before:.6f}",
        f"{share_after:.6f}",
        "" if best_before == float("inf") else f"{best_before:.9f}",
        "" if best_after == float("inf") else f"{best_after:.9f}",
        failures,
        f"{elapsed:.2f}"
    ]

    rows.append(row)

    print(f"Seed {seed}:")
    print(f"  Feasible before repair: {share_before:.2%}")
    print(f"  Feasible after repair:  {share_after:.2%}")
    print(f"  Repair failures:        {failures}")
    print(f"  Best cost before:       {best_before}")
    print(f"  Best cost after:        {best_after}")
    print()

output = "results/abdimazhit_exp.csv"

with open(output, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(columns)
    writer.writerows(rows)

print("EXPERIMENT FINISHED")
print("Results saved to:", output)
