"""Run M2's part: solve the LP relaxation of B2 and write results/table1_lp.csv.

Uses tools/standin_model_B2.py until M1's src/model is merged (see README_M2.md).
Run from the project root:  python src/lp/run_lp.py
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.lp import solve_lp, lp_summary, write_summary_csv  # noqa: E402
from tools.standin_model_B2 import build_B2  # noqa: E402


def run_lp(seed: int = 10, repeat: int = 5) -> dict:
    data, d = build_B2(seed)
    res = solve_lp(data, repeat=repeat)
    summary = lp_summary(res, d["M"], d["S"])
    summary["n_variables"] = data.n_vars
    summary["n_rows"] = data.n_rows
    return summary


if __name__ == "__main__":
    s = run_lp()
    write_summary_csv(s, os.path.join(ROOT, "results", "table1_lp.csv"))
    for k, v in s.items():
        print(f"{k}: {v}")
