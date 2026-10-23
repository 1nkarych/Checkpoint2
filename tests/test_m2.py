"""M2 tests: reproduce the hand task H2 (triangle problem, team T10: c = (6, 7, 5)) with src/lp.

The triangle problem is data of this test, not part of the reference model.
Run:  python -m pytest tests/test_m2.py     (or: python tests/test_m2.py)
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.lp import LPData, solve_lp, fractional_indices  # noqa: E402

TOL = 1e-6
C = np.array([6.0, 7.0, 5.0])
# (T1) y1+y2>=1, (T2) y2+y3>=1, (T3) y1+y3>=1  ->  "<=" form: -A y <= -1
A = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=float)
B = np.ones(3)


def triangle_lp() -> LPData:
    return LPData(C, -A, -B, None, None, [(0, 1)] * 3)


def test_sign_convention_two_variable_lp():
    # min x1 + 2 x2  s.t. x1 + x2 >= 1, x >= 0 : optimum (1, 0), z = 1.
    # Raising the rhs 1 -> 1 + eps raises z by eps, so the price of the ">=" row is +1;
    # in "<=" form (-x1 - x2 <= -1) the same price is -1 (= d z / d b with b = -1).
    r = solve_lp(LPData(np.array([1.0, 2.0]), np.array([[-1.0, -1.0]]), np.array([-1.0]),
                        None, None, [(0, None), (0, None)]))
    assert abs(r.objective - 1.0) < TOL
    assert abs(r.pi_ub[0] - (-1.0)) < TOL
    assert abs(r.reduced_costs[1] - 1.0) < TOL and abs(r.reduced_costs[0]) < TOL


def test_h2a_half_point_feasible_and_cost():
    y = np.full(3, 0.5)
    assert np.all(A @ y >= B - TOL)
    assert abs(C @ y - 9.0) < TOL


def test_h2a_lp_optimum_is_the_half_point():
    r = solve_lp(triangle_lp())
    assert abs(r.objective - 9.0) < TOL                     # z_LP
    assert np.allclose(r.x, 0.5, atol=TOL)                  # unique optimum
    assert len(fractional_indices(r.x, range(3))) == 3      # all three entries fractional
    assert np.allclose(-r.pi_ub, [4.0, 3.0, 2.0], atol=TOL)  # prices of T1..T3 (>= form: +4,3,2)


def test_h2b_roundings():
    # round each coordinate of (1/2,1/2,1/2) up (1) or down (0): the 8 binary points
    feas, cost = {}, {}
    for y in itertools.product([0, 1], repeat=3):
        y = np.array(y, dtype=float)
        feas[tuple(int(v) for v in y)] = bool(np.all(A @ y >= B - TOL))
        cost[tuple(int(v) for v in y)] = float(C @ y)
    assert [p for p, f in feas.items() if f] == [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    assert cost[(1, 1, 1)] == 18.0 and cost[(0, 0, 0)] == 0.0 and not feas[(0, 0, 0)]
    assert (cost[(0, 1, 1)], cost[(1, 0, 1)], cost[(1, 1, 0)]) == (12.0, 11.0, 13.0)


def test_h2c_integer_optimum_and_gap():
    zs = min(float(C @ np.array(y)) for y in itertools.product([0, 1], repeat=3)
             if np.all(A @ np.array(y) >= B))
    zlp = solve_lp(triangle_lp()).objective
    assert abs(zs - 11.0) < TOL
    assert abs((zs - zlp) / zs - 2.0 / 11.0) < TOL          # relative gap 0.182
    # no integral point is LP-optimal: every feasible binary point costs >= 11 > 9 = z_LP
    assert all(float(C @ np.array(y)) >= 11.0 - TOL for y in itertools.product([0, 1], repeat=3)
               if np.all(A @ np.array(y) >= B))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
