
import unittest
import itertools
import numpy as np


class TestH1(unittest.TestCase):

    def test_cost_vector(self):
        c = np.array([6, 7, 5])
        self.assertTrue(
            np.array_equal(c, [6, 7, 5])
        )

    def test_constraint_matrix(self):
        A = np.array([
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1]
        ])
        expected = np.array([
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1]
        ])
        self.assertTrue(np.array_equal(A, expected))
        self.assertEqual(A.shape, (3, 3))

    def test_rhs_vector(self):
        b = np.array([1, 1, 1])
        self.assertTrue(np.array_equal(b, [1, 1, 1]))

    def test_all_binary_points(self):
        A = np.array([
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1]
        ])
        b = np.ones(3)
        c = np.array([6, 7, 5])

        points = list(itertools.product([0, 1], repeat=3))
        self.assertEqual(len(points), 8)

        feasible = []
        for point in points:
            y = np.array(point)
            if np.all(A @ y >= b):
                feasible.append((point, int(c @ y)))

        expected = [
            ((0, 1, 1), 12),
            ((1, 0, 1), 11),
            ((1, 1, 0), 13),
            ((1, 1, 1), 18)
        ]

        self.assertEqual(feasible, expected)

    def test_integer_optimum(self):
        self.assertEqual(min([12, 11, 13, 18]), 11)


if __name__ == "__main__":
    unittest.main()
