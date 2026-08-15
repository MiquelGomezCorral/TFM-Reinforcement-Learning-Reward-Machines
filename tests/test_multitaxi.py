import unittest
from types import SimpleNamespace

import numpy as np

from src.envs import create_environment
from src.envs.taxi_big_env import MultiTaxiEnv


class MultiTaxiTest(unittest.TestCase):
    def test_relative_observation_encodes_deltas_and_status(self):
        env = MultiTaxiEnv(num_passengers=2, observation_mode="relative")
        env.state = [2, 1, 1, 2, 4, 3]

        observation = env._observation()

        self.assertEqual(observation.shape, (14,))
        self.assertTrue(env.observation_space.contains(observation))
        np.testing.assert_array_equal(observation[:4], np.array([-2, 3, 2, -1]))
        np.testing.assert_array_equal(observation[4:7], np.array([1, 0, 0]))
        np.testing.assert_array_equal(observation[7:11], np.array([0, 0, 2, 2]))
        np.testing.assert_array_equal(observation[11:14], np.array([0, 1, 0]))
    def test_relative_reset_preserves_discrete_raw_state(self):
        env = MultiTaxiEnv(num_passengers=3, observation_mode="relative")

        observation, info = env.reset(seed=1)

        self.assertEqual(observation.shape, (21,))
        self.assertEqual(info["raw_state"], env.encode(env.state))

    def test_factored_observation_one_hot_encodes_full_state(self):
        env = MultiTaxiEnv(num_passengers=2, observation_mode="factored")
        env.state = [0, 4, 1, 2, 4, 3]

        observation = env._observation()

        self.assertEqual(observation.shape, (28,))
        self.assertEqual(observation.sum(), 6)
        self.assertEqual(observation[0], 1)
        self.assertEqual(observation[9], 1)

    def test_factory_uses_raw_states_for_multitaxi_propositions(self):
        env, get_propositions = create_environment(
            SimpleNamespace(
                gym_id="MultiTaxi-v0",
                multitaxi_grid_size=5,
                multitaxi_num_passengers=2,
                multitaxi_observation_mode="relative",
                multitaxi_reward_shaping=False,
                multitaxi_non_terminal_reward=-2,
            )
        )
        state, info = env.reset(seed=1)
        new_state, _, _, _, new_info = env.step(0)

        propositions = get_propositions(env, info["raw_state"], 0, new_info["raw_state"])

        self.assertEqual(state.shape, new_state.shape)
        self.assertIsInstance(propositions, list)
        self.assertFalse(env.reward_shaping)
        self.assertEqual(env.non_terminal_reward, -2)

    def test_progress_rewards_scale_with_passengers(self):
        env = MultiTaxiEnv(num_passengers=2, observation_mode="relative")
        env.state = [0, 0, 0, 1, 0, 2]

        _, pickup_reward, _, _, _ = env.step(4)
        env.state = [0, 4, 4, 1, 4, 1]
        _, dropoff_reward, terminated, _, _ = env.step(5)

        self.assertEqual(pickup_reward, 10)
        self.assertEqual(dropoff_reward, 40)
        self.assertTrue(terminated)

    def test_sparse_rewards_only_on_completion(self):
        env = MultiTaxiEnv(num_passengers=2, reward_shaping=False)
        env.state = [0, 0, 0, 1, 0, 1]

        _, invalid_reward, terminated, _, _ = env.step(1)
        _, pickup_reward, terminated, _, _ = env.step(4)
        env.state[:2] = [0, 4]
        _, completion_reward, terminated, _, _ = env.step(5)

        self.assertEqual(invalid_reward, -1)
        self.assertEqual(pickup_reward, -1)
        self.assertEqual(completion_reward, 50)
        self.assertTrue(terminated)

    def test_invalid_actions_are_reported(self):
        env = MultiTaxiEnv(grid_size=5, num_passengers=2)
        env.reset(seed=1)

        _, reward, _, _, info = env.step(5)

        self.assertEqual(reward, -10)
        self.assertTrue(info["invalid_action"])

        env.state[:2] = [0, 0]
        _, reward, _, _, info = env.step(1)

        self.assertEqual(reward, -1)
        self.assertTrue(info["invalid_action"])


if __name__ == "__main__":
    unittest.main()
