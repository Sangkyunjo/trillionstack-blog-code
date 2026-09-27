from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "examples"


def load_example(topic: str, name: str):
    path = ROOT / topic / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TimeSeriesTests(unittest.TestCase):
    def test_ar2_recursion_matches_companion_matrix(self):
        example = load_example("time-series", "ar2_impulse_response")
        phi1, phi2 = 0.5, 0.2
        companion = np.array([[phi1, phi2], [1.0, 0.0]])
        e1 = np.array([1.0, 0.0])
        expected = [e1 @ np.linalg.matrix_power(companion, j) @ e1 for j in range(8)]
        np.testing.assert_allclose(example.impulse_response(phi1, phi2, 8), expected)
        self.assertTrue(np.all(np.abs(example.companion_eigenvalues(phi1, phi2)) < 1))
        with self.assertRaises(ValueError):
            example.impulse_response(phi1, phi2, 0)

    def test_arma_forecast_error_variance_uses_future_shocks_only(self):
        example = load_example("time-series", "arma11_forecast_error")
        weights = example.psi_weights(0.5, 0.3, 4)
        np.testing.assert_allclose(weights, [1.0, 0.8, 0.4, 0.2])
        self.assertAlmostEqual(example.forecast_error_variance(0.5, 0.3, 2.0, 4), 3.68)
        self.assertAlmostEqual(example.forecast_error_variance(0.5, 0.3, 2.0, 1), 2.0)
        with self.assertRaises(ValueError):
            example.forecast_error_variance(0.5, 0.3, -1.0, 2)


class StatisticsTests(unittest.TestCase):
    def test_ols_recovers_line_and_rejects_unidentifiable_slope(self):
        example = load_example("statistics", "ols_lstsq")
        coefficients, r_squared = example.fit_line([0, 1, 2, 3], [2, 5, 8, 11])
        np.testing.assert_allclose(coefficients, [2.0, 3.0])
        self.assertAlmostEqual(r_squared, 1.0)
        with self.assertRaises(ValueError):
            example.fit_line([1, 1, 1], [2, 3, 4])

    def test_information_identity_and_zero_support(self):
        example = load_example("statistics", "information_theory")
        p = np.array([0.7, 0.2, 0.1])
        q = np.array([0.6, 0.3, 0.1])
        self.assertAlmostEqual(example.cross_entropy(p, q), example.entropy(p) + example.kl_divergence(p, q))
        self.assertGreater(example.kl_divergence(p, q), 0)
        self.assertEqual(example.entropy([1.0, 0.0]), 0.0)
        self.assertEqual(example.cross_entropy([1.0, 0.0], [0.0, 1.0]), float("inf"))


if __name__ == "__main__":
    unittest.main()
