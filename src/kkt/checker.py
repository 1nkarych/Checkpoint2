"""M3: checker of the optimality conditions (a)-(e) for an LP (Checkpoint 3, Task 3).

The LP is   min c^T x   s.t.   rows  a_i^T x  (sense_i)  b_i ,   lb <= x <= ub.
sense_i is one of "<=", ">=", "=".

Sign convention (same as the handout):  pi_i = d z_LP / d b_i.
    "<=" row  -> pi_i <= 0
    ">=" row  -> pi_i >= 0
    "="  row  -> pi_i free
Reduced cost:  r_j = c_j - sum_i pi_i A_ij.

The function takes plain arrays, so it works with the output of ANY solver:
the caller only has to convert the solver's duals to this convention.
"""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class KKTReport:
    primal_feas: float      # (a) largest violation of a row or a bound
    dual_sign: float        # (b) largest sign violation of pi
    comp_slack: float       # (c) max |pi_i * slack_i|
    reduced_cost: float     # (d) largest violation of the reduced-cost conditions
    strong_duality: float   # (e) |c^T x - dual objective|
    primal_obj: float
    dual_obj: float

    def __post_init__(self):                # turn -0.0 into 0.0 for clean printing
        for f in ("primal_feas", "dual_sign", "comp_slack", "reduced_cost", "strong_duality"):
            setattr(self, f, float(getattr(self, f)) + 0.0)

    def as_tuple(self):
        return (self.primal_feas, self.dual_sign, self.comp_slack,
                self.reduced_cost, self.strong_duality)

    def ok(self, tol=1e-6):
        return all(v <= tol for v in self.as_tuple())

    def __str__(self):
        names = ["(a) primal feasibility", "(b) dual sign", "(c) compl. slackness",
                 "(d) reduced costs", "(e) strong duality gap"]
        lines = [f"{n:26s} {v:.3e}" for n, v in zip(names, self.as_tuple())]
        lines.append(f"primal obj {self.primal_obj:.6f}   dual obj {self.dual_obj:.6f}")
        return "\n".join(lines)


def check_kkt(c, A, b, sense, lb, ub, x, pi, tol_active=1e-9, tol_r=1e-6):
    """Return a KKTReport for the pair (x, pi).

    c (n,), A (m,n) dense or scipy.sparse, b (m,), sense list of "<=", ">=", "=",
    lb, ub (n,) (use np.inf for no upper bound), x (n,), pi (m,).
    tol_active: a value within tol_active of a bound counts as 'at the bound' in (d).
    """
    c = np.asarray(c, float)
    b = np.asarray(b, float)
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    x = np.asarray(x, float)
    pi = np.asarray(pi, float)
    A = sp.csr_matrix(A)
    sense = np.asarray(sense)

    le, ge, eq = sense == "<=", sense == ">=", sense == "="
    if not (le | ge | eq).all():
        raise ValueError("sense entries must be '<=', '>=' or '='")

    # (a) primal feasibility: rows and bounds
    Ax = A @ x
    row_viol = np.zeros(len(b))
    row_viol[le] = np.maximum(0.0, Ax[le] - b[le])
    row_viol[ge] = np.maximum(0.0, b[ge] - Ax[ge])
    row_viol[eq] = np.abs(Ax[eq] - b[eq])
    bound_viol = np.maximum(0.0, np.maximum(lb - x, x - ub))   # inf ub gives 0
    a = max(row_viol.max(initial=0.0), bound_viol.max(initial=0.0))

    # (b) dual feasibility: signs of pi
    sign_viol = np.zeros(len(b))
    sign_viol[le] = np.maximum(0.0, pi[le])
    sign_viol[ge] = np.maximum(0.0, -pi[ge])
    bb = sign_viol.max(initial=0.0)

    # (c) complementary slackness on rows: pi_i * (b_i - a_i^T x) = 0
    cc = np.abs(pi * (b - Ax)).max(initial=0.0)

    # (d) reduced costs r = c - A^T pi
    r = c - A.T @ pi
    viol = np.zeros(len(c))
    for j in range(len(c)):
        if ub[j] - lb[j] <= tol_active:          # fixed variable (e.g. forbidden pair): r free
            continue
        at_lb = x[j] - lb[j] <= tol_active
        at_ub = np.isfinite(ub[j]) and ub[j] - x[j] <= tol_active
        if at_lb:
            viol[j] = max(0.0, -r[j])            # x = lower bound  =>  r >= 0
        elif at_ub:
            viol[j] = max(0.0, r[j])             # x = upper bound  =>  r <= 0
        else:
            viol[j] = abs(r[j])                  # strictly inside   =>  r = 0
    dd = viol.max(initial=0.0)

    # (e) strong duality: c^T x = b^T pi + sum_j [min(0,r_j) ub_j + max(0,r_j) lb_j]
    primal = float(c @ x)
    dual = float(b @ pi)
    # A tiny negative r_j on a variable without upper bound (e.g. t_l, solver noise ~1e-8) would give
    # -inf; up to tol_r it is treated as 0 (its size is already reported in (d)).
    neg = np.where((r < 0) & ~((r >= -tol_r) & ~np.isfinite(ub)), r, 0.0)
    pos = np.where((r > 0) & ~((r <= tol_r) & ~np.isfinite(lb)), r, 0.0)
    with np.errstate(invalid="ignore"):
        dual += float(np.sum(neg[neg < 0] * ub[neg < 0]))
        dual += float(np.sum(pos[pos > 0] * lb[pos > 0]))
    ee = abs(primal - dual) if np.isfinite(dual) else np.inf

    return KKTReport(a, bb, cc, dd, ee, primal, dual)
