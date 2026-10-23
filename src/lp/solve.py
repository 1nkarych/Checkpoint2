"""Solving a linear program given in matrix form and reading the solution.

Matrix form of the handout (Task 1):
    min  c^T x   s.t.   A_ub x <= b_ub,   A_eq x = b_eq,   lo <= x <= hi
(">=" rows are stored as "<=" rows after multiplying by -1; t_l >= 0 has hi = None).

Sign convention of the dual prices (the one of the handout, Task 3):
    pi_i = d z_LP / d b_i      (change of the optimal value per unit increase of b_i)
so pi_i <= 0 for "<=" rows of a minimisation problem and pi_i is free for "=" rows.
scipy/HiGHS reports exactly this quantity in `marginals`; the convention is checked on a
two-variable LP with a known answer in tests/test_m2.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog


@dataclass
class LPData:
    """An LP in matrix form. Rows of A_ub / A_eq may be None (no such rows)."""
    c: np.ndarray
    A_ub: Optional[np.ndarray] = None
    b_ub: Optional[np.ndarray] = None
    A_eq: Optional[np.ndarray] = None
    b_eq: Optional[np.ndarray] = None
    bounds: Sequence[Tuple[Optional[float], Optional[float]]] = field(default_factory=list)

    @property
    def n_vars(self) -> int:
        return len(self.c)

    @property
    def n_rows(self) -> int:
        n = 0
        if self.A_ub is not None:
            n += np.shape(self.A_ub)[0]
        if self.A_eq is not None:
            n += np.shape(self.A_eq)[0]
        return n


@dataclass
class LPResult:
    status: int                 # 0 = optimal (scipy convention)
    message: str
    objective: float            # z_LP
    x: np.ndarray               # optimal primal vector
    pi_ub: Optional[np.ndarray]  # dual prices of the "<=" rows (<= 0), d z / d b_ub
    pi_eq: Optional[np.ndarray]  # dual prices of the "=" rows (free), d z / d b_eq
    reduced_costs: np.ndarray   # r_j = c_j - sum_i pi_i A_ij  (lower + upper marginals)
    solver: str                 # solver name and version
    seconds: float              # wall time of the solver call
    iterations: Optional[int]

    @property
    def optimal(self) -> bool:
        return self.status == 0


def solve_lp(data: LPData, method: str = "highs", repeat: int = 1) -> LPResult:
    """Solve the LP with scipy.optimize.linprog (HiGHS) and read primal and dual information.

    repeat > 1 repeats the solve and reports the median time (the solution is the same).
    """
    times = []
    res = None
    for _ in range(max(1, repeat)):
        t0 = time.perf_counter()
        res = linprog(
            c=data.c,
            A_ub=data.A_ub, b_ub=data.b_ub,
            A_eq=data.A_eq, b_eq=data.b_eq,
            bounds=list(data.bounds) if len(data.bounds) else (0, None),
            method=method,
        )
        times.append(time.perf_counter() - t0)

    if res.status != 0:
        raise RuntimeError(f"LP not solved to optimality: status={res.status}, {res.message}")

    import scipy
    pi_ub = np.asarray(res.ineqlin.marginals) if data.A_ub is not None else None
    pi_eq = np.asarray(res.eqlin.marginals) if data.A_eq is not None else None
    # reduced cost of variable j = marginal of its lower bound + marginal of its upper bound
    # (lower >= 0, upper <= 0 for a minimisation problem); zero for variables strictly inside.
    r = np.asarray(res.lower.marginals) + np.asarray(res.upper.marginals)

    return LPResult(
        status=res.status,
        message=res.message,
        objective=float(res.fun),
        x=np.asarray(res.x, dtype=float),
        pi_ub=pi_ub,
        pi_eq=pi_eq,
        reduced_costs=r,
        solver=f"scipy {scipy.__version__} linprog(method='{method}') [HiGHS]",
        seconds=float(np.median(times)),
        iterations=getattr(res, "nit", None),
    )
