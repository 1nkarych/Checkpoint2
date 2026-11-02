"""Table-1 numbers of Task 2 (z_LP, solver, time, fractional entries) as a dict / CSV."""
from __future__ import annotations

import csv
import os

import numpy as np

from .fractional import fractional_indices, split_vms
from .solve import LPResult


def lp_summary(res: LPResult, M: int, S: int) -> dict:
    """Storyline B summary. Integer variables = the first M*S entries (y_kl); t_l follow."""
    int_idx = np.arange(M * S)
    frac = fractional_indices(res.x, int_idx)
    split = split_vms(res.x, M, S)
    return {
        "z_LP": round(res.objective, 6),
        "solver": res.solver,
        "time_seconds": round(res.seconds, 4),
        "iterations": res.iterations,
        "n_fractional_y": int(len(frac)),
        "n_vms_split": int(len(split)),
        "split_vms": {int(k): [(int(l), round(v, 6)) for l, v in p] for k, p in split.items()},
    }


def write_summary_csv(summary: dict, path: str) -> None:
    """results/table1_lp.csv (one row of key,value pairs; split VMs written as text)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        for k, v in summary.items():
            w.writerow([k, v])
