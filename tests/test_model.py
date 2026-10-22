
"""Tests for the M1 model: Team T10, Storyline B, Variant 2."""


import unittest
import numpy as np
import sys
from pathlib import Path

# Add the project root to Python's import path.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model.model_builder import build_model


class TestModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = build_model()

    def test_matrix_dimensions(self):
        model = self.model

        self.assertEqual(model["c"].shape, (561,))
        self.assertEqual(model["A_eq"].shape, (50, 561))
        self.assertEqual(model["b_eq"].shape, (50,))
        self.assertEqual(model["A_ub"].shape, (66, 561))
        self.assertEqual(model["b_ub"].shape, (66,))

    def test_constraint_counts(self):
        model = self.model

        self.assertEqual(len(model["eq_row_names"]), 50)
        self.assertEqual(len(model["ub_row_names"]), 66)
        self.assertEqual(len(model["row_names"]), 116)

    def test_forbidden_assignments(self):
        model = self.model
        bounds = model["bounds"]
        y_index = model["y_index"]

        self.assertEqual(len(model["forbidden_pairs"]), 51)

        for k, l in model["forbidden_pairs"]:
            self.assertEqual(bounds[y_index(k, l)], (0.0, 0.0))

    def test_assignment_constraints(self):
        model = self.model

        # Every VM has one equality row with S coefficients equal to 1.
        self.assertTrue(
            np.allclose(model["A_eq"].sum(axis=1), 11)
        )
        self.assertTrue(
            np.allclose(model["b_eq"], 1.0)
        )

    def test_generated_instance(self):
        data = self.model["data"]

        self.assertEqual(data["S"], 11)
        self.assertEqual(data["M"], 50)
        self.assertEqual(int(data["cpu_cap"].sum()), 504)
        self.assertEqual(int(data["mem_cap"].sum()), 1725)
        self.assertEqual(int(data["cpu_req"].sum()), 233)
        self.assertEqual(int(data["mem_req"].sum()), 871)


if __name__ == "__main__":
    unittest.main(verbosity=2)
