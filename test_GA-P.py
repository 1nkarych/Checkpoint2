"""Role test of GA-P (repair configuration), instance of team T10: cp34_instance(10, 'B', 2).
Run:  python tests/test_GA-P.py     (or: pytest tests/test_GA-P.py)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import checkpoint4_generator_PUBLIC as gen
from src.meta import ga_problem as P

D = gen.cp34_instance(10, 'B', 2)
CTX = P.prepare(D)
N = 1000


def _random_pairs(rng):
    return [(P.random_individual(CTX, rng), P.random_individual(CTX, rng)) for _ in range(N)]


def test_evaluate_matches_checker():
    """evaluate == evaluate_B on 1000 random individuals (feasible AND infeasible; cost to 1e-9, flag equal)."""
    rng = np.random.default_rng(1)
    n_inf = 0
    for i in range(N):
        ind = P.random_individual(CTX, rng) if i % 2 == 0 else P.repair(CTX, P.random_individual(CTX, rng), rng)
        fit, cost, feas = P.evaluate(CTX, ind)
        ref, problems = gen.evaluate_B(D, list(ind))
        assert abs(cost - ref) <= 1e-9, (i, cost, ref)
        assert feas == (len(problems) == 0), (i, feas, problems)
        assert fit == cost                       # repair configuration: fitness is the cost
        n_inf += (not feas)
    assert n_inf > 0, 'test must contain infeasible individuals too'


def test_crossover_mutation_valid():
    """crossover / mutation return valid individuals (length M, servers in range, zones respected) on 1000 pairs."""
    rng = np.random.default_rng(2)
    for p1, p2 in _random_pairs(rng):
        for ch in (P.crossover(CTX, p1, p2, rng), P.mutate(CTX, p1, rng)):
            assert len(ch) == D['M'] and ch.min() >= 0 and ch.max() < D['S']
            assert P.zone_ok(CTX, ch)
        assert 1 <= int((P.mutate(CTX, p1, rng) != p1).sum()) <= 2


def test_repair_output_feasible():
    """repair output passes the checker (also from crossover/mutation offspring and from extreme individuals)."""
    rng = np.random.default_rng(3)
    for p1, p2 in _random_pairs(rng):
        for ch in (p1, P.mutate(CTX, P.crossover(CTX, p1, p2, rng), rng)):
            r = P.repair(CTX, ch, rng)
            assert gen.evaluate_B(D, list(r))[1] == []
    for ch in (np.zeros(D['M'], int), np.full(D['M'], 3)):                         # all VMs on one server (zones violated too)
        assert gen.evaluate_B(D, list(P.repair(CTX, ch, rng)))[1] == []


def test_repair_keeps_feasible_unchanged_and_is_free():
    """A feasible individual is returned unchanged; repair never calls the cost function (data only)."""
    rng = np.random.default_rng(4)
    ind = P.repair(CTX, P.random_individual(CTX, rng), rng)
    assert np.array_equal(P.repair(CTX, ind, rng), ind)
    import inspect
    src = inspect.getsource(P.repair) + inspect.getsource(P._rebuild)
    assert 'cost_of' not in src and 'evaluate' not in src


def test_penalty_formula_hand_example():
    """Penalty mode (repair off): fitness = cost + rho * (sum of relative overloads) on a hand example."""
    ind = np.array([CTX['allowed'][k][0] for k in range(D['M'])])   # all on the first allowed server
    cpu, mem = P.loads(CTX, ind)
    manual = sum(max(0, cpu[l] - D['cpu_cap'][l]) / D['cpu_cap'][l] + max(0, mem[l] - D['mem_cap'][l]) / D['mem_cap'][l] for l in range(D['S']))
    fit, cost, feas = P.evaluate(CTX, ind, rho=10.0)
    assert manual > 0 and not feas
    assert abs(fit - (cost + 10.0 * manual)) < 1e-9


if __name__ == '__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_'):
            f(); print('ok', name)
