import math
import random
import unittest

from checkpoint4_generator_PUBLIC import cp34_instance, evaluate_B
from src.meta.sa import initial_solution, transfer_move, incremental_cost, solve


class TestSA(unittest.TestCase):

    def setUp(self):
        self.d = cp34_instance(10, "B", 2)

    def test_incremental_cost(self):
        rng = random.Random(0)
        current = initial_solution(self.d, rng)
        cost, problems = evaluate_B(self.d, current)
        self.assertEqual(problems, [])

        for _ in range(1000):
            candidate = transfer_move(self.d, current, rng)

            if candidate is None:
                continue

            new_cost = incremental_cost(self.d, current, candidate, cost)
            checker_cost, problems = evaluate_B(self.d, candidate)

            self.assertEqual(problems, [])
            self.assertTrue(math.isclose(new_cost, checker_cost, abs_tol=1e-8))

            current = candidate
            cost = new_cost

    def test_reproducibility(self):
        params = {"T0": 1.0, "Tend": 0.01}
        first = solve(self.d, 2, 0, 500, params=params)
        second = solve(self.d, 2, 0, 500, params=params)

        self.assertEqual(first["solution"], second["solution"])
        self.assertEqual(first["cost"], second["cost"])
        self.assertEqual(first["trace"], second["trace"])

    def test_budget(self):
        result = solve(self.d, 2, 0, 500)
        self.assertEqual(result["evals"], 500)

    def test_trace(self):
        result = solve(self.d, 2, 0, 500)
        costs = [cost for _, cost in result["trace"]]

        self.assertTrue(
            all(costs[i] >= costs[i + 1] for i in range(len(costs) - 1))
        )

    def test_feasibility(self):
        result = solve(self.d, 2, 0, 500)
        cost, problems = evaluate_B(self.d, result["solution"])

        self.assertEqual(problems, [])
        self.assertTrue(result["feasible"])
        self.assertTrue(math.isclose(result["cost"], cost, abs_tol=1e-8))


if __name__ == "__main__":
    unittest.main()