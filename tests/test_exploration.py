import unittest

import numpy as np

from src.utils import compute_epsilon, compute_training_epsilon


class ExplorationTest(unittest.TestCase):
    def test_epsilon_starts_at_maximum_and_decays_toward_minimum(self):
        self.assertEqual(compute_epsilon(0.05, 1, 0, 0.1), 1)
        self.assertGreater(compute_epsilon(0.05, 1, 1, 0.1), 0.05)
        self.assertAlmostEqual(compute_epsilon(0.05, 1, 1_000, 0.1), 0.05)

    def test_training_epsilon_decays_after_warmup(self):
        epsilon = compute_training_epsilon(0.05, 1, 125, 1 / 100, 25)

        self.assertEqual(compute_training_epsilon(0.05, 1, 24, 1 / 100, 25), 1)
        self.assertEqual(compute_training_epsilon(0.05, 1, 25, 1 / 100, 25), 1)
        self.assertAlmostEqual(epsilon, 0.05 + 0.95 / np.e)


if __name__ == "__main__":
    unittest.main()
