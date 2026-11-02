"""M3 tests: the KKT checker reproduces the hand task H3 (triangle problem of team T10)
and the sign convention is tested on tiny LPs with a known answer.

The triangle problem is DATA of this test, not part of the reference model.
Run:  python -m pytest tests/test_M3.py   (or just: python tests/test_M3.py)
"""
import os
import sys

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.kkt import check_kkt  # noqa: E402

TOL = 1e-6

# ---- triangle problem, team T10: c = (6, 7, 5), three ">=" rows ------------------------------
C = np.array([6.0, 7.0, 5.0])
A = np.array([[1, 1, 0],      # (T1) y1 + y2 >= 1
              [0, 1, 1],      # (T2) y2 + y3 >= 1
              [1, 0, 1]],     # (T3) y1 + y3 >= 1
             float)
B = np.ones(3)
SENSE = [">="] * 3
LB, UB = np.zeros(3), np.ones(3)
X_HALF = np.array([0.5, 0.5, 0.5])
PI = np.array([4.0, 3.0, 2.0])        # H3(a)
PI_PRIME = np.array([5.0, 2.0, 1.0])  # H3(b)


def test_H3a_all_five_conditions_hold():
    rep = check_kkt(C, A, B, SENSE, LB, UB, X_HALF, PI)
    assert np.allclose(rep.as_tuple(), 0.0, atol=TOL)
    assert abs(rep.primal_obj - 9.0) < TOL and abs(rep.dual_obj - 9.0) < TOL


def test_H3b_which_conditions_fail():
    rep = check_kkt(C, A, B, SENSE, LB, UB, X_HALF, PI_PRIME)
    a, b, c, d, e = rep.as_tuple()
    assert a < TOL and b < TOL and c < TOL      # feasibility, signs, slackness still hold
    assert abs(d - 2.0) < TOL                   # r3 = 5 - (2 + 1) = 2 but y3 = 1/2 needs r3 = 0
    assert abs(e - 1.0) < TOL                   # c^T x = 9 versus dual objective 8


def test_H3c_pi_prime_is_a_lower_bound():
    rep = check_kkt(C, A, B, SENSE, LB, UB, X_HALF, PI_PRIME)
    r = C - A.T @ PI_PRIME
    assert (r >= -TOL).all()                    # pi' is dual feasible (r >= 0, pi >= 0) ...
    assert abs(rep.dual_obj - 8.0) < TOL        # ... so b^T pi' + sum min(0, r) = 8 <= z_LP
    assert abs((9.0 - rep.dual_obj) - 1.0) < TOL  # one unit below the best bound z_LP = 9


def test_checker_accepts_solver_output_triangle():
    # HiGHS stores ">=" rows as -A y <= -b, so its marginals are d z / d(-b) = -pi: flip the sign.
    res = linprog(C, A_ub=-A, b_ub=-B, bounds=list(zip(LB, UB)), method="highs")
    pi = -res.ineqlin.marginals
    rep = check_kkt(C, A, B, SENSE, LB, UB, res.x, pi)
    assert rep.ok(TOL), str(rep)
    assert abs(res.fun - 9.0) < TOL


# ---- sign convention on LPs where the answer is known -----------------------------------------
def test_sign_convention_ge_row():
    # min 2x1 + 3x2  s.t. x1 + x2 >= 4, 0 <= x <= 10.  Optimum (4, 0), z = 8, dz/db = +2.
    c, A2, b2 = [2.0, 3.0], np.array([[1.0, 1.0]]), [4.0]
    res = linprog(c, A_ub=-A2, b_ub=[-4.0], bounds=[(0, 10)] * 2, method="highs")
    pi = -res.ineqlin.marginals
    assert abs(pi[0] - 2.0) < TOL                              # price of a ">=" row is >= 0
    assert check_kkt(c, A2, b2, [">="], [0, 0], [10, 10], res.x, pi).ok()
    # the wrong sign must be detected as a violation of (b)
    assert check_kkt(c, A2, b2, [">="], [0, 0], [10, 10], res.x, -pi).dual_sign > 1.0


def test_sign_convention_le_row():
    # max x1 + x2 = min -x1 - x2  s.t. x1 + x2 <= 4, 0 <= x <= 3.  z = -4, dz/db = -1.
    c, A2, b2 = [-1.0, -1.0], np.array([[1.0, 1.0]]), [4.0]
    res = linprog(c, A_ub=A2, b_ub=b2, bounds=[(0, 3)] * 2, method="highs")
    pi = res.ineqlin.marginals                                 # "<=" rows: already dz/db
    assert abs(pi[0] + 1.0) < TOL                              # price of a "<=" row is <= 0
    assert check_kkt(c, A2, b2, ["<="], [0, 0], [3, 3], res.x, pi).ok()


def test_equality_row_and_unbounded_variable():
    # min t  s.t.  t - 2 y = 0 , y = 1 (as equality), 0 <= y <= 1, t >= 0 (no upper bound)
    c = [1.0, 0.0]                                             # variables (t, y)
    A2 = np.array([[1.0, -2.0], [0.0, 1.0]])
    b2 = [0.0, 1.0]
    res = linprog(c, A_eq=A2, b_eq=b2, bounds=[(0, None), (0, 1)], method="highs")
    pi = res.eqlin.marginals
    rep = check_kkt(c, A2, b2, ["=", "="], [0, 0], [np.inf, 1], res.x, pi)
    assert rep.ok(TOL), str(rep)


def test_fixed_variable_has_free_reduced_cost():
    # a variable fixed to 0 (forbidden pair, ub = 0) may have any reduced-cost sign
    c = [1.0, 1.0]
    A2 = np.array([[1.0, 1.0]])
    res = linprog(c, A_ub=-A2, b_ub=[-1.0], bounds=[(0, 1), (0, 0)], method="highs")
    pi = -res.ineqlin.marginals
    assert check_kkt(c, A2, [1.0], [">="], [0, 0], [1, 0], res.x, pi).ok()


def test_solver_noise_on_variable_without_upper_bound():
    # min t s.t. t >= 1, t >= 0 (no upper bound). A solver (e.g. CBC) may return pi = 1 + 1e-8,
    # giving r = -1e-8 on t; this must not turn the dual objective into -infinity.
    rep = check_kkt([1.0], np.array([[1.0]]), [1.0], [">="], [0.0], [np.inf], [1.0], [1.0 + 1e-8])
    assert np.isfinite(rep.strong_duality) and rep.ok(TOL), str(rep)
    # a clearly wrong price must still fail
    assert not check_kkt([1.0], np.array([[1.0]]), [1.0], [">="], [0.0], [np.inf], [1.0], [1.5]).ok(TOL)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
