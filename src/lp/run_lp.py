
"""Run M2's LP relaxation using M1's official model builder."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model.model_builder import build_model
from src.lp.solve import LPData, solve_lp
from src.lp.summary import lp_summary, write_summary_csv


def build_lp_data():
    """Convert M1's model dictionary into the LPData format."""
    model = build_model()

    data = LPData(
        c=model["c"],
        A_ub=model["A_ub"],
        b_ub=model["b_ub"],
        A_eq=model["A_eq"],
        b_eq=model["b_eq"],
        bounds=model["bounds"],
    )
    return data, model


def run_lp(seed: int = 10, repeat: int = 5) -> dict:
    """Solve the LP relaxation and return the Table 1 summary."""
    data, model = build_lp_data()
    result = solve_lp(data, repeat=repeat)

    d = model["data"]
    summary = lp_summary(result, d["M"], d["S"])
    summary["n_variables"] = data.n_vars
    summary["n_rows"] = data.n_rows

    return summary


if __name__ == "__main__":
    summary = run_lp()

    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    output_path = os.path.join(results_dir, "table1_lp.csv")
    write_summary_csv(summary, output_path)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print(f"\nSaved: {output_path}")
