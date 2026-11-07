"""Random-search baseline (owned by GA-A): `budget` independent random candidates = random assignment of every VM to
an allowed server, made feasible by the same repair rule (ga_problem.random_individual); the best one is kept.
Same interface and same evaluation counting as ga.solve."""
import numpy as np

from . import ga_problem as gp
from .interface import check


def solve(d, variant, seed, budget, params=None, start=None):
    rng = np.random.default_rng(seed)
    best_cost, best, least_fit, least = float('inf'), None, float('inf'), None
    trace = []
    for e in range(1, budget + 1):
        ind = gp.random_individual(d, rng)
        fitness, cost, feasible = gp.evaluate(d, ind)
        if feasible and cost < best_cost:
            best_cost, best = cost, ind
        if fitness < least_fit:
            least_fit, least = fitness, ind
        if e % 100 == 0:
            trace.append((e, best_cost))
    if not trace or trace[-1][0] != budget:
        trace.append((budget, best_cost))
    sol = [int(v) for v in (best if best is not None else least)]
    cost, problems = check(d, variant, sol)
    return dict(solution=sol, cost=cost, evals=budget, feasible=len(problems) == 0, trace=trace)
