
"""M5 integration runner for Checkpoint 3, Team T10, Storyline B2."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from src.model.model_builder import build_model
from src.lp.run_lp import build_lp_data
from src.lp.solve import solve_lp
from src.lp.summary import lp_summary, write_summary_csv
from src.kkt import check_kkt
from src.sens.sensitivity import run_sensitivity, save_results

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TOL = 1e-6


def write_csv(path, rows):
    """Write a list of dictionaries to CSV."""
    if not rows:
        raise ValueError(f"No rows to save to {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {path.relative_to(ROOT)}")


def run_tests():
    """Run the M2 and M3 hand-task checks."""
    for test_file in ("test_m2.py", "test_M3.py"):
        path = ROOT / "tests" / test_file
        if not path.exists():
            raise FileNotFoundError(f"Required test file not found: {path}")

        print(f"\nRunning {test_file}...")
        subprocess.run([sys.executable, str(path)], cwd=ROOT, check=True)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Checkpoint 3 - Team T10")
    print("Storyline B | Variant 2 | Seed 10")
    print("=" * 60)

    # M1: build and validate the official model.
    model = build_model()
    data, model_from_lp = build_lp_data()

    if not np.allclose(model["c"], data.c):
        raise RuntimeError("M1/M2 objective vectors do not match.")

    if not np.allclose(model["A_eq"], data.A_eq):
        raise RuntimeError("M1/M2 equality matrices do not match.")

    if not np.allclose(model["A_ub"], data.A_ub):
        raise RuntimeError("M1/M2 inequality matrices do not match.")

    print("\nM1 model checks passed.")
    print(f"Variables: {data.n_vars}")
    print(f"Constraint rows: {data.n_rows}")

    # M2: solve the LP once and save Table 1.
    result = solve_lp(data, repeat=5)
    d = model["data"]
    table1 = lp_summary(result, d["M"], d["S"])
    table1["n_variables"] = data.n_vars
    table1["n_rows"] = data.n_rows

    write_summary_csv(table1, str(RESULTS / "table1_lp.csv"))

    print("\nM2: LP summary")
    print(f"z_LP: {result.objective:.6f}")
    print(f"Fractional y variables: {table1['n_fractional_y']}")
    print(f"VMs split across servers: {table1['n_vms_split']}")

    # M3: evaluate the five KKT conditions on the actual M2 solution.
    A = np.vstack((data.A_eq, data.A_ub))
    b = np.concatenate((data.b_eq, data.b_ub))
    sense = (["="] * len(data.b_eq)) + (["<="] * len(data.b_ub))

    # SciPy's linprog marginals use the required dz/db sign convention.
    pi = np.concatenate((result.pi_eq, result.pi_ub))

    lower = np.array([bound[0] for bound in data.bounds], dtype=float)
    upper = np.array([
        np.inf if bound[1] is None else bound[1]
        for bound in data.bounds
    ], dtype=float)

    report = check_kkt(
        c=data.c,
        A=A,
        b=b,
        sense=sense,
        lb=lower,
        ub=upper,
        x=result.x,
        pi=pi,
    )

    names = [
        "(a) primal feasibility",
        "(b) dual sign",
        "(c) complementary slackness",
        "(d) reduced costs",
        "(e) strong duality",
    ]

    kkt_rows = []
    for name, value in zip(names, report.as_tuple()):
        kkt_rows.append({
            "condition": name,
            "value": f"{value:.12g}",
            "tolerance": TOL,
            "passed": value <= TOL,
        })

    kkt_rows.extend([
        {
            "condition": "primal objective",
            "value": f"{report.primal_obj:.12g}",
            "tolerance": "",
            "passed": "",
        },
        {
            "condition": "dual objective",
            "value": f"{report.dual_obj:.12g}",
            "tolerance": "",
            "passed": "",
        },
    ])

    write_csv(RESULTS / "table2_kkt.csv", kkt_rows)

    print("\nM3: KKT certificate")
    print(report)
    print(f"All five conditions pass (tolerance {TOL:g}): {report.ok(TOL)}")

    if not report.ok(TOL):
        raise RuntimeError(
            "KKT verification failed. Check results/table2_kkt.csv."
        )

    # M4: sensitivity analysis and Table 3.
    sensitivity = run_sensitivity(target_row=2, step=0.01)
    save_results(sensitivity, output_dir=RESULTS)

    print("\nM4: sensitivity results")
    for run in sensitivity["runs"]:
        print(
            f"RHS={run['b']:.2f} | "
            f"z_LP={run['z_LP']:.6f} | "
            f"runtime={run['solve_seconds']:.6f}s"
        )

    print("\nTable 3:")
    for name, value in sensitivity["table3"].items():
        print(f"{name}: {value:.6f}")

    if not sensitivity["interval_ok"]:
        raise RuntimeError("M4 dual interval check failed.")

    # Run the hand-task and sign-convention tests for M2 and M3.
    run_tests()

    # Save an overall machine-readable summary.
    summary = {
        "team": "T10",
        "storyline": "B",
        "variant": 2,
        "seed": 10,
        "model_variables": data.n_vars,
        "model_constraint_rows": data.n_rows,
        "z_LP": result.objective,
        "m2_fractional_y": table1["n_fractional_y"],
        "m2_vms_split": table1["n_vms_split"],
        "m3_kkt_values": dict(zip(names, report.as_tuple())),
        "m3_kkt_passed": report.ok(TOL),
        "m4_interval_ok": sensitivity["interval_ok"],
        "m4_table3": sensitivity["table3"],
    }

    summary_path = RESULTS / "integration_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"\nSaved: {summary_path.relative_to(ROOT)}")
    print("\nAll integration stages completed successfully.")


if __name__ == "__main__":
    main()
