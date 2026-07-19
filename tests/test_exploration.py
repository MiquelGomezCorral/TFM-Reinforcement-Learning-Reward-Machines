import unittest

from src.utils import compute_epsilon


class ExplorationTest(unittest.TestCase):
    def test_epsilon_starts_at_maximum_and_decays_toward_minimum(self):
        self.assertEqual(compute_epsilon(0.05, 1, 0, 0.1), 1)
        self.assertGreater(compute_epsilon(0.05, 1, 1, 0.1), 0.05)
        self.assertAlmostEqual(compute_epsilon(0.05, 1, 1_000, 0.1), 0.05)


if __name__ == "__main__":
    unittest.main()
