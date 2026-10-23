"""Reproduce the T10 H4 hand calculations to absolute tolerance 1e-6.

Run directly: python tests/test_M4.py
Or: python -m pytest tests/test_M4.py -v
"""

import numpy as np
from scipy.optimize import linprog


TOL = 1e-6
COST = np.array([6., 7., 5.])
COVER = np.array([[1., 1., 0.], [0., 1., 1.], [1., 0., 1.]])


def solve_triangle(epsilon):
    # scipy requires <= rows, so multiply each covering row and RHS by -1.
    return linprog(COST, A_ub=-COVER,
                   b_ub=-np.array([1. + epsilon, 1., 1.]),
                   bounds=[(0., 1.)] * 3, method="highs")


def hand_point(epsilon):
    return np.array([(1. + epsilon) / 2., (1. + epsilon) / 2.,
                     (1. - epsilon) / 2.])


def test_h4_a_solution_and_cost():
    for epsilon in (-1., -0.5, -0.01, 0., 0.01, 0.5, 1.):
        result = solve_triangle(epsilon)
        assert result.success, result.message
        expected = hand_point(epsilon)
        np.testing.assert_allclose(result.x, expected, atol=TOL, rtol=0.)
        np.testing.assert_allclose(COVER @ result.x,
                                   [1. + epsilon, 1., 1.], atol=TOL, rtol=0.)
        assert abs(result.fun - (9. + 4. * epsilon)) <= TOL


def test_h4_a_slope_and_dual_price():
    low, base, high = [solve_triangle(e) for e in (-0.01, 0., 0.01)]
    assert all(r.success for r in (low, base, high))
    # Convert the <= row marginal back to the original >= row convention.
    dual_price = -float(base.ineqlin.marginals[0])
    left = (base.fun - low.fun) / 0.01
    right = (high.fun - base.fun) / 0.01
    np.testing.assert_allclose([low.fun, base.fun, high.fun],
                               [8.96, 9., 9.04], atol=TOL, rtol=0.)
    np.testing.assert_allclose([left, dual_price, right], [4., 4., 4.],
                               atol=TOL, rtol=0.)


def test_h4_b_valid_range():
    # At both endpoints the proposed point still satisfies its bounds.
    for epsilon in (-1., 1.):
        point = hand_point(epsilon)
        assert np.all(point >= 0.) and np.all(point <= 1.)
    # Immediately outside either endpoint, the formula violates a bound.
    for epsilon in (-1.01, 1.01):
        point = hand_point(epsilon)
        assert np.any(point < 0.) or np.any(point > 1.)
    assert solve_triangle(1.01).status == 2  # RHS > 2 is infeasible.
    below = solve_triangle(-1.01)
    assert below.success
    assert abs(below.fun - 5.) <= TOL
    assert abs(below.fun - (9. + 4. * -1.01)) > TOL


def test_h4_c_cost_change():
    base, changed = solve_triangle(0.), solve_triangle(0.1)
    assert base.success and changed.success
    assert abs((changed.fun - base.fun) - 4. * 0.1) <= TOL


if __name__ == "__main__":
    checks = [test_h4_a_solution_and_cost, test_h4_a_slope_and_dual_price,
              test_h4_b_valid_range, test_h4_c_cost_change]
    for check in checks:
        check()
        print(f"PASS: {check.__name__}")
    print("H4 verified: z(1)=9.000, slope=4.000, valid range [-1,1].")
