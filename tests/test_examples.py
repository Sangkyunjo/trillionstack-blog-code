from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "graph-neural-networks"


def load_module(name: str):
    path = EXAMPLES / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class ExampleTests(unittest.TestCase):
    def test_anonymous_walk(self) -> None:
        module = load_module("anonymous_walk_example")
        self.assertEqual((0, 1, 0, 2, 1), module.anonymize_walk([7, 2, 7, 5, 2]))

    def test_gcn_normalization_is_symmetric(self) -> None:
        module = load_module("gcn_normalization")
        adjacency = np.array([[0.0, 1.0], [1.0, 0.0]])
        normalized = module.normalized_adjacency(adjacency)
        np.testing.assert_allclose(normalized, normalized.T)
        np.testing.assert_allclose(normalized, np.full((2, 2), 0.5))

    def test_permutation_equivariance(self) -> None:
        module = load_module("permutation_equivariance")
        adjacency = np.array([[0.0, 1.0], [1.0, 0.0]])
        features = np.array([[1.0], [3.0]])
        permutation = np.array([[0.0, 1.0], [1.0, 0.0]])
        expected = permutation @ module.aggregate(adjacency, features)
        actual = module.aggregate(
            permutation @ adjacency @ permutation.T,
            permutation @ features,
        )
        np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    unittest.main()
