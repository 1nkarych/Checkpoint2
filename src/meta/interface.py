"""Interface of the pair's method + the wrapper of the checker (starter-kit file, shared by the pair).

solve(d, variant, seed, budget, params=None, start=None) -> dict with
    solution : assign (storyline B) or (centers, assign, vehicle) (storyline A)
    cost     : cost returned by the checker
    evals    : number of evaluations spent (== budget)
    feasible : True exactly when the checker reports no problems
    trace    : list of pairs (evaluations, best feasible cost so far), at least every 1000 evaluations
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import checkpoint4_generator_PUBLIC as gen  # noqa: E402


def check(d, variant, solution):
    """Apply the checker of the generator. Returns (cost, list_of_problems)."""
    if d['story'] == 'A':
        centers, assign, vehicle = solution
        return gen.evaluate_A(d, variant, centers, assign, vehicle)
    return gen.evaluate_B(d, solution)


def solve(d, variant, seed, budget, params=None, start=None):  # stub: the real one is in ga.py / sa.py
    raise NotImplementedError
