"""Checkpoint 2 -- public parameter generator (Introduction to Optimization, ItO2025).

Uses the SAME team seed as Checkpoint 1 (given in your team's task file).
It only produces the *parameters* of the test problems. You must write the
functions f, grad, hess yourselves (derive by hand first, then check_grad).

    p = checkpoint2_params(SEED)
    print(p)
"""
import numpy as np

X0_QUAD = np.array([1.3, 0.7])        # common start for the quadratic test problem


def checkpoint2_params(seed):
    rng = np.random.default_rng([seed, 2])          # separate stream: Checkpoint 1 data unchanged
    c = int(round(10 ** rng.uniform(2.0, 3.0)))     # condition number of the quadratic problem (100..1000)
    theta_deg = int(rng.integers(15, 76))           # rotation angle for the rotated copy (degrees)
    x0_rosen = np.array([round(float(rng.uniform(-2.0, -0.5)), 2),
                         round(float(rng.uniform(0.0, 3.0)), 2)])   # team start on Rosenbrock
    return dict(c=c, theta_deg=theta_deg, x0_rosen=x0_rosen)


def rotation(theta_deg):
    t = np.deg2rad(theta_deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

# Quadratic test problems (write f, grad, hess yourselves):
#   Q1:  f(x) = x1^2 + c*x2^2,           start  x0 = X0_QUAD
#   Q2:  g(x) = f(R^T x),  R = rotation(theta_deg),   start  R @ X0_QUAD
# (Q2 is Q1 rotated: same eigenvalues, same minimum value, same distance to it.)

# Example -- replace SEED with your team's value (given separately):
# print(checkpoint2_params(SEED))
