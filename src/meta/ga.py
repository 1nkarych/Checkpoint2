"""GA-A part: the loop of the genetic algorithm (tournament selection, elitism, generations, budget), solve(...).
Configuration: GA, repair. The problem-specific functions come from ga_problem.py (GA-P), see README.

Counting (Instructions, Section 3): every offspring and every member of the initial population = 1 evaluation;
the copy of the elite individual costs nothing; the last generation is cut short so that exactly `budget` is spent.
Random numbers: only numpy.random.default_rng(seed).
"""
import numpy as np

from . import ga_problem as gp
from .interface import check

DEFAULTS = dict(N=30, pc=0.9, n_mut_max=2, tournament=2, elite=1, trace_step=100)


def tournament(fitness, rng, k=2):
    """Index of the best (lowest fitness) of k distinct random entrants; never the worst of them (k >= 2)."""
    n = len(fitness)
    k = min(k, n)
    entrants = rng.choice(n, size=k, replace=False)
    return int(entrants[int(np.argmin(np.asarray(fitness)[entrants]))])


def next_generation(d, pop, fit, rng, p, n_offspring, on_offspring):
    """One generation. pop: list of individuals, fit: list of fitness values.
    Elitism: the `elite` best individuals are copied unchanged and free. Then `n_offspring` children are made:
    two tournament parents -> crossover with prob. pc (otherwise a copy of parent 1) -> mutation of 1..n_mut_max
    components -> repair -> evaluation (on_offspring returns the fitness and counts 1 evaluation)."""
    order = np.argsort(np.asarray(fit), kind='stable')
    new_pop = [pop[i].copy() for i in order[:p['elite']]]
    new_fit = [fit[i] for i in order[:p['elite']]]
    for _ in range(n_offspring):
        a = tournament(fit, rng, p['tournament'])
        b = tournament(fit, rng, p['tournament'])
        child = gp.crossover(d, pop[a], pop[b], rng) if rng.random() < p['pc'] else pop[a].copy()
        child = gp.mutate(d, child, rng, int(rng.integers(1, p['n_mut_max'] + 1)))
        child = gp.repair(d, child, rng)
        new_pop.append(child)
        new_fit.append(on_offspring(child))
    return new_pop, new_fit


def _distinct(pop):
    return len({np.asarray(x, dtype=np.int64).tobytes() for x in pop})


def _hamming(pop):
    P = np.asarray(pop)
    n = len(P)
    if n < 2:
        return 0.0
    return float((P[:, None, :] != P[None, :, :]).sum() / (n * (n - 1)))


def solve(d, variant, seed, budget, params=None, start=None):
    if d['story'] != 'B':
        raise NotImplementedError('this GA implements storyline B (virtual machines on servers)')
    p = dict(DEFAULTS)
    p.update(params or {})
    rng = np.random.default_rng(seed)
    st = dict(evals=0, best_cost=float('inf'), best_ind=None, least_fit=float('inf'), least_ind=None)
    trace = []

    def evaluate(ind):
        fitness, cost, feasible = gp.evaluate(d, ind)
        st['evals'] += 1
        if feasible and cost < st['best_cost']:
            st['best_cost'], st['best_ind'] = cost, ind.copy()
        if fitness < st['least_fit']:
            st['least_fit'], st['least_ind'] = fitness, ind.copy()
        if st['evals'] % p['trace_step'] == 0:
            trace.append((st['evals'], st['best_cost']))
        return fitness

    # initial population (every member counts; `start`, if given, is repaired and takes the first place)
    N0 = min(p['N'], budget)
    pop, fit = [], []
    for i in range(N0):
        if i == 0 and start is not None:
            ind = gp.repair(d, np.asarray(start, dtype=np.int64), rng)
        else:
            ind = gp.random_individual(d, rng)
        pop.append(ind)
        fit.append(evaluate(ind))

    distinct, hamming = [], []
    if len(pop) == p['N']:
        distinct.append(_distinct(pop)); hamming.append(_hamming(pop))
    while st['evals'] < budget and len(pop) == p['N']:
        n_off = min(p['N'] - p['elite'], budget - st['evals'])   # last generation is cut short
        pop, fit = next_generation(d, pop, fit, rng, p, n_off, evaluate)
        if len(pop) == p['N']:
            distinct.append(_distinct(pop)); hamming.append(_hamming(pop))
    if not trace or trace[-1][0] != st['evals']:
        trace.append((st['evals'], st['best_cost']))

    sol = st['best_ind'] if st['best_ind'] is not None else st['least_ind']
    solution = [int(v) for v in sol]
    cost, problems = check(d, variant, solution)
    return dict(solution=solution, cost=cost, evals=st['evals'], feasible=len(problems) == 0, trace=trace,
                distinct=distinct, hamming=hamming, generations=len(distinct))
