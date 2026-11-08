"""GA-A test: same seed -> same run; exactly the budget is spent; elitism keeps the best individual;
the tournament never returns the worst of its entrants. Run: python -m unittest tests.test_ga_a -v"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import checkpoint4_generator_PUBLIC as gen  # noqa: E402
from src.meta import ga, ga_problem as gp  # noqa: E402


class TestGAA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = gen.cp34_instance(10, 'B', 2)
        cls.d_other = gen.cp34_instance(3, 'B', 2)   # another instance of the same variant (like the hidden ones)

    def test_same_seed_same_run(self):
        a = ga.solve(self.d, 2, 7, 3000)
        b = ga.solve(self.d, 2, 7, 3000)
        self.assertEqual(a['solution'], b['solution'])
        self.assertEqual(a['cost'], b['cost'])
        self.assertEqual(a['trace'], b['trace'])
        self.assertEqual(a['distinct'], b['distinct'])
        c = ga.solve(self.d, 2, 8, 3000)
        self.assertNotEqual(a['solution'], c['solution'])   # another seed -> another run
        # the global random state is not used
        np.random.seed(1); x = ga.solve(self.d, 2, 7, 1000)
        np.random.seed(2); y = ga.solve(self.d, 2, 7, 1000)
        self.assertEqual(x['solution'], y['solution'])

    def test_exactly_the_budget(self):
        for d in (self.d, self.d_other):
            for budget in (1, 10, 30, 31, 59, 60, 1234, 5000):   # also budgets below / not a multiple of N-1
                r = ga.solve(d, 2, 1, budget)
                self.assertEqual(r['evals'], budget, f'budget {budget}')
                self.assertEqual(r['trace'][-1][0], budget)
                self.assertTrue(all(b[0] - a[0] <= 1000 for a, b in zip(r['trace'], r['trace'][1:])))
        # evaluations are counted at the source: count the calls of gp.evaluate
        calls = []
        orig = gp.evaluate
        gp.evaluate = lambda d, ind: (calls.append(1), orig(d, ind))[1]
        try:
            ga.solve(self.d, 2, 3, 2000)
        finally:
            gp.evaluate = orig
        self.assertEqual(len(calls), 2000)

    def test_honest_flag_and_checker_cost(self):
        r = ga.solve(self.d, 2, 2, 2000)
        cost, problems = gen.evaluate_B(self.d, r['solution'])
        self.assertEqual(r['cost'], cost)
        self.assertEqual(r['feasible'], len(problems) == 0)
        self.assertTrue(r['feasible'])
        trace_best = [c for _, c in r['trace']]
        self.assertTrue(all(b <= a for a, b in zip(trace_best, trace_best[1:])))   # best-so-far never increases
        self.assertAlmostEqual(trace_best[-1], cost, places=9)

    def test_elitism_keeps_the_best(self):
        p = dict(ga.DEFAULTS)
        rng = np.random.default_rng(5)
        pop = [gp.random_individual(self.d, rng) for _ in range(p['N'])]
        fit = [gp.evaluate(self.d, x)[0] for x in pop]
        for _ in range(60):   # generations
            best_before = min(fit)
            champion = pop[int(np.argmin(fit))].copy()
            pop, fit = ga.next_generation(self.d, pop, fit, rng, p, p['N'] - p['elite'],
                                          lambda x: gp.evaluate(self.d, x)[0])
            self.assertEqual(len(pop), p['N'])
            self.assertLessEqual(min(fit), best_before)                      # best fitness never gets worse
            self.assertTrue(any(np.array_equal(champion, x) for x in pop))  # the elite is still there, unchanged

    def test_tournament_never_returns_the_worst(self):
        rng = np.random.default_rng(0)
        for k in (2, 3, 5):
            for _ in range(2000):
                fit = rng.permutation(30).astype(float)   # distinct fitness values
                # replay the entrants: same rng state -> same draw as inside tournament
                state = rng.bit_generator.state
                winner = ga.tournament(fit, rng, k)
                rng.bit_generator.state = state
                entrants = rng.choice(30, size=k, replace=False)
                rng.bit_generator.state = state
                ga.tournament(fit, rng, k)
                self.assertIn(winner, entrants)
                self.assertEqual(fit[winner], fit[entrants].min())
                self.assertNotEqual(fit[winner], fit[entrants].max())
        # the best individual of the population wins whenever it is an entrant: its win rate is about 2/N for k=2
        fit = np.arange(30, dtype=float); wins = sum(ga.tournament(fit, rng, 2) == 0 for _ in range(6000))
        self.assertTrue(0.04 < wins / 6000 < 0.10)

    def test_start_is_accepted(self):
        base = ga.solve(self.d, 2, 0, 500)
        r = ga.solve(self.d, 2, 0, 500, start=base['solution'])
        self.assertEqual(r['evals'], 500)
        self.assertLessEqual(r['cost'], base['cost'] + 1e-12)   # the start is a member of the first population

    def test_repair_off_never_calls_repair(self):
        calls = []
        orig = gp.repair
        gp.repair = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
        try:
            r = ga.solve(self.d, 2, 4, 3000, params={'repair': False, 'rho': 10}, start=[0] * 50)
            self.assertEqual(len(calls), 0)
            calls.clear()
            ga.solve(self.d, 2, 4, 3000, params={'repair': True})
            self.assertGreater(len(calls), 0)
        finally:
            gp.repair = orig
        self.assertEqual(r['evals'], 3000)
        self.assertEqual(r['feasible'], len(gen.evaluate_B(self.d, r['solution'])[1]) == 0)

    def test_counters_and_rho(self):
        for rep, rho in ((True, 100), (False, 1), (False, 10), (False, 100)):
            r = ga.solve(self.d, 2, 6, 4000, params={'repair': rep, 'rho': rho})
            self.assertEqual(r['evals'], 4000)
            self.assertEqual(r['init_individuals'] + r['offspring'], 4000)
            self.assertLessEqual(r['offspring_feasible'], r['offspring'])
            self.assertLessEqual(r['init_feasible'], r['init_individuals'])
            if rep:
                self.assertEqual(r['offspring_feasible'], r['offspring'])   # repair delivers feasible offspring
            else:
                self.assertLess(r['offspring_feasible'], r['offspring'])    # raw offspring do violate sometimes
            self.assertEqual(r['rho'], rho); self.assertEqual(r['repair'], rep)
        a = ga.solve(self.d, 2, 6, 4000, params={'repair': False, 'rho': 1})
        b = ga.solve(self.d, 2, 6, 4000, params={'repair': False, 'rho': 100})
        self.assertNotEqual(a['trace'], b['trace'])   # rho changes the selection, hence the run

    def test_default_equals_repair_true(self):
        a = ga.solve(self.d, 2, 3, 3000)
        b = ga.solve(self.d, 2, 3, 3000, params={'repair': True, 'rho': 100})
        self.assertEqual(a['solution'], b['solution']); self.assertEqual(a['trace'], b['trace'])


if __name__ == '__main__':
    unittest.main()
