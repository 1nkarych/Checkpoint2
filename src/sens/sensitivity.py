"""Team T10, M4: sensitivity of the B1 equality for VM k=2.

Run from the project root: python src/sens/sensitivity.py
run_sensitivity(lp, target_row=...) also accepts the team's LP matrices.
"""
from pathlib import Path
import csv
import importlib.util
from time import perf_counter

import numpy as np
from scipy.optimize import linprog

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOL = 1e-6

def load_team_data():
    """Use the original, unchanged public generator."""
    path = PROJECT_ROOT / "checkpoint3_generator_PUBLIC.py"
    if not path.is_file():
        raise FileNotFoundError(f"Place checkpoint3_generator_PUBLIC.py in {PROJECT_ROOT}")
    spec = importlib.util.spec_from_file_location("cp3_public_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.cp34_instance(10, "B", 2), module.TANGENT_POINTS

def build_reference_lp(data, tangent_points):
    """Standalone B2 reference model; use M1's matrices here when integrating.

    Variables are y[k,l] in VM-major order, followed by t[l].
    B4 is stored as a*CPU_load/C - t <= -b.
    Forbidden pairs remain as variables with bounds (0,0).
    """
    m, s = data["M"], data["S"]
    n = m * s + s
    c = np.r_[np.zeros(m * s), np.ones(s)]
    a_eq = np.zeros((m, n))
    b_eq = np.ones(m)
    for k in range(m):
        a_eq[k, k * s:(k + 1) * s] = 1.0

    a_ub = np.zeros(((2 + len(tangent_points)) * s, n))
    b_ub = np.zeros(a_ub.shape[0])
    row = 0
    for resource in ("cpu", "mem"):
        for l in range(s):
            a_ub[row, l:m * s:s] = data[f"{resource}_req"]
            b_ub[row] = data[f"{resource}_cap"][l]
            row += 1
    for l in range(s):
        for u in tangent_points:
            a = 1.0 / (1.0 - u) ** 2
            intercept = -(u ** 2) / (1.0 - u) ** 2
            a_ub[row, l:m * s:s] = a * data["cpu_req"] / data["cpu_cap"][l]
            a_ub[row, m * s + l] = -1.0
            b_ub[row] = -intercept
            row += 1

    bounds = []
    for k in range(m):
        for l in range(s):
            zone = data["vm_required_zone"][k]
            forbidden = zone != -1 and zone != data["zone"][l]
            bounds.append((0.0, 0.0 if forbidden else 1.0))
    bounds.extend([(0.0, None)] * s)
    return dict(c=c, A_eq=a_eq, b_eq=b_eq, A_ub=a_ub, b_ub=b_ub, bounds=bounds)


def check_team_instance(data, lp):
    """Stop if the generator or reference formulation differs from the packet."""
    actual = {
        "servers": data["S"], "VMs": data["M"],
        "cpu_cap_sum": int(np.sum(data["cpu_cap"])),
        "mem_cap_sum": int(np.sum(data["mem_cap"])),
        "cpu_req_sum": int(np.sum(data["cpu_req"])),
        "mem_req_sum": int(np.sum(data["mem_req"])),
        "forbidden_pairs": sum(hi == 0 for lo, hi in lp["bounds"]),
        "variables": len(lp["c"]),
        "constraint_rows": len(lp["b_eq"]) + len(lp["b_ub"]),
        "designated_VM": int(np.argmax(data["cpu_req"])),
    }
    expected = dict(servers=11, VMs=50, cpu_cap_sum=504, mem_cap_sum=1725,
                    cpu_req_sum=233, mem_req_sum=871, forbidden_pairs=51,
                    variables=561, constraint_rows=116, designated_VM=2)
    if actual != expected:
        raise ValueError(f"Team packet mismatch. Expected {expected}; got {actual}")
    return actual


def check_solver_convention():
    """min x+2y, x+y=b, x,y>=0 has value b and equality price +1."""
    results = [linprog([1., 2.], A_eq=[[1., 1.]], b_eq=[b],
                       bounds=(0, None), method="highs") for b in (1., 1.01)]
    if not all(r.success for r in results):
        raise RuntimeError("Solver convention example failed")
    price = float(results[0].eqlin.marginals[0])
    slope = (results[1].fun - results[0].fun) / 0.01
    if max(abs(price - 1), abs(slope - 1)) > TOL:
        raise RuntimeError("Unexpected sign convention for equality prices")


def solve_lp(lp, rhs):
    start = perf_counter()
    result = linprog(lp["c"], A_ub=lp["A_ub"], b_ub=lp["b_ub"],
                     A_eq=lp["A_eq"], b_eq=rhs, bounds=lp["bounds"],
                     method="highs", options={"primal_feasibility_tolerance": 1e-9,
                                               "dual_feasibility_tolerance": 1e-9})
    elapsed = perf_counter() - start
    if not result.success:
        raise RuntimeError(f"LP solve failed: {result.message}")
    x = result.x
    violations = [0., float(np.max(np.abs(lp["A_eq"] @ x - rhs))),
                  float(np.max(lp["A_ub"] @ x - lp["b_ub"]))]
    for value, (lo, hi) in zip(x, lp["bounds"]):
        if lo is not None:
            violations.append(lo - value)
        if hi is not None:
            violations.append(value - hi)
    violation = max(violations)
    if violation > TOL:
        raise RuntimeError(f"Primal feasibility violation: {violation}")
    return result, elapsed, violation


def run_sensitivity(lp=None, target_row=2, step=0.01):
    """Return Table 3 and run details without changing the input matrices.

    lp uses scipy.linprog keys: c, A_eq, b_eq, A_ub, b_ub, bounds.
    target_row must identify B1 for VM k=2 in the supplied equality ordering.
    """
    check_solver_convention()
    if lp is None:
        data, tangents = load_team_data()
        lp = build_reference_lp(data, tangents)
        check_team_instance(data, lp)
    if step <= 0 or not np.isfinite(step):
        raise ValueError("step must be positive and finite")
    rhs = np.asarray(lp["b_eq"], dtype=float).copy()
    base_b = float(rhs[target_row])
    runs, solutions = [], []
    for delta in (-step, 0., step):
        changed_rhs = rhs.copy()
        changed_rhs[target_row] = base_b + delta
        result, elapsed, violation = solve_lp(lp, changed_rhs)
        solutions.append(result)
        runs.append(dict(b=base_b + delta, z_LP=float(result.fun),
                         solve_seconds=elapsed, max_primal_violation=violation))

    left = float((solutions[1].fun - solutions[0].fun) / step)
    price = float(solutions[1].eqlin.marginals[target_row])
    right = float((solutions[2].fun - solutions[1].fun) / step)
    if not (left - TOL <= price <= right + TOL):
        raise RuntimeError(f"Dual interval check failed: {left}, {price}, {right}")
    table = dict(left_slope=left, dual_price=price, right_slope=right)
    return dict(table3=table, runs=runs, target_row=target_row,
                step=step, slope_difference=abs(right - left), interval_ok=True)


def save_results(summary, output_dir=None):
    directory = Path(output_dir) if output_dir is not None else PROJECT_ROOT / "results"
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows in (("table3.csv", [summary["table3"]]),
                       ("sensitivity_runs.csv", summary["runs"])):
        with (directory / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    summary = run_sensitivity()
    save_results(summary)
    print("Team T10 | B2 | seed=10 | B1 for VM k=2 | scipy/HiGHS")
    print("Instance checks passed: 561 variables, 116 rows, 51 forbidden pairs.")
    for run in summary["runs"]:
        print(f"b={run['b']:.2f}   z_LP={run['z_LP']:.6f}")
    for name, value in summary["table3"].items():
        print(f"{name}: {value:.6f}")
    print("Dual interval check: PASS (tolerance 1e-6)")
    if summary["slope_difference"] > TOL:
        print("Different finite-difference slopes: inspect smaller perturbations for a kink.")
    else:
        print("Left and right slopes agree to 1e-6.")
    print(f"Saved: {PROJECT_ROOT / 'results' / 'table3.csv'}")
    print(f"Saved: {PROJECT_ROOT / 'results' / 'sensitivity_runs.csv'}")


if __name__ == "__main__":
    main()
