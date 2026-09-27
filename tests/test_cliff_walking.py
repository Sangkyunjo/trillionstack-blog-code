import sys
import unittest
from pathlib import Path

import numpy as np

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "reinforcement-learning" / "cliff-walking"
sys.path.insert(0, str(EXAMPLE_DIR))

from algorithms import (
    ALGORITHMS,
    first_visit_mc_control,
    off_policy_actor_critic,
    q_learning,
    reinforce_with_baseline,
    sarsa,
    weighted_importance_sampling_mc_control,
)
from cliff_walking import (
    CliffWalkingEnv,
    epsilon_greedy_probs,
    grad_log_softmax,
    softmax,
)


class EnvironmentTests(unittest.TestCase):
    def test_cliff_returns_to_start_without_termination(self):
        env = CliffWalkingEnv()
        transition = env.step(env.start_state, 3)
        self.assertEqual(transition.next_state, env.start_state)
        self.assertEqual(transition.reward, -100.0)
        self.assertFalse(transition.terminated)
        self.assertTrue(transition.fell_from_cliff)

    def test_goal_terminates(self):
        env = CliffWalkingEnv()
        transition = env.step(9, 1)
        self.assertEqual(transition.next_state, env.goal_state)
        self.assertEqual(transition.reward, 10.0)
        self.assertTrue(transition.terminated)


class ProbabilityTests(unittest.TestCase):
    def test_epsilon_greedy_splits_ties(self):
        probs = epsilon_greedy_probs(np.array([2.0, 2.0, 0.0, 0.0]), 0.2)
        np.testing.assert_allclose(probs, [0.45, 0.45, 0.05, 0.05])

    def test_softmax_gradient_sums_to_zero(self):
        logits = np.array([1.0, -1.0, 0.5, 0.0])
        probs = softmax(logits)
        gradient = grad_log_softmax(probs, 2)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0)
        self.assertAlmostEqual(float(np.sum(gradient)), 0.0)
        self.assertGreater(gradient[2], 0.0)
        self.assertTrue(np.all(gradient[[0, 1, 3]] < 0.0))

        finite_difference = np.zeros_like(logits)
        step = 1e-6
        for index in range(logits.size):
            plus = logits.copy()
            minus = logits.copy()
            plus[index] += step
            minus[index] -= step
            finite_difference[index] = (
                np.log(softmax(plus)[2]) - np.log(softmax(minus)[2])
            ) / (2 * step)
        np.testing.assert_allclose(gradient, finite_difference, atol=1e-7)


class AlgorithmTests(unittest.TestCase):
    def test_all_algorithms_return_finite_tables(self):
        for name, algorithm in ALGORITHMS.items():
            with self.subTest(name=name):
                result = algorithm(episodes=30, seed=7, max_steps=100)
                table = np.asarray(result["table"])
                self.assertEqual(table.shape, (15, 4))
                self.assertTrue(np.all(np.isfinite(table)))
                self.assertEqual(len(result["returns"]), 30)

    def test_seed_reproducibility(self):
        for algorithm in (
            first_visit_mc_control,
            sarsa,
            q_learning,
            reinforce_with_baseline,
            weighted_importance_sampling_mc_control,
            off_policy_actor_critic,
        ):
            with self.subTest(algorithm=algorithm.__name__):
                first = algorithm(episodes=20, seed=11, max_steps=100)
                second = algorithm(episodes=20, seed=11, max_steps=100)
                np.testing.assert_allclose(first["table"], second["table"])
                self.assertEqual(first["returns"], second["returns"])


if __name__ == "__main__":
    unittest.main()
