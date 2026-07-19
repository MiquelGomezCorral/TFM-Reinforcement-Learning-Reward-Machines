import unittest

import numpy as np

from src.utils import next_episode_seed, seed_training


class SeedTest(unittest.TestCase):
    def test_episode_seeds_are_deterministic_and_in_range(self):
        first = np.random.default_rng(42)
        second = np.random.default_rng(42)

        first_seeds = [next_episode_seed(first) for _ in range(3)]
        second_seeds = [next_episode_seed(second) for _ in range(3)]

        self.assertEqual(first_seeds, second_seeds)
        self.assertTrue(all(0 <= seed < 2**31 for seed in first_seeds))

    def test_training_seed_initializes_action_space_and_generator(self):
        class ActionSpace:
            def seed(self, seed):
                self.value = seed

        action_space = ActionSpace()

        generator = seed_training(42, action_space)

        self.assertEqual(action_space.value, 42)
        self.assertEqual(next_episode_seed(generator), next_episode_seed(np.random.default_rng(42)))


if __name__ == "__main__":
    unittest.main()
