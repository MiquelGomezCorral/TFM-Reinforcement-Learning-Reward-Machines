import unittest

import numpy as np

from src.envs import create_environment
from src.envs.taxi_big_env import MultiTaxiEnv


class MultiTaxiTest(unittest.TestCase):
    def test_relative_observation_encodes_deltas_and_status(self):
        env = MultiTaxiEnv(num_passengers=2, observation_mode="relative")
        env.state = [0, 0, 1, 2, 4, 3]

        observation = env._observation()

        np.testing.assert_array_equal(
            observation,
            np.array([0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0.75, 0, 1, 0], dtype=np.float32),
        )

    def test_relative_reset_preserves_discrete_raw_state(self):
        env = MultiTaxiEnv(num_passengers=3, observation_mode="relative")

        observation, info = env.reset(seed=1)

        self.assertEqual(observation.shape, (21,))
        self.assertEqual(info["raw_state"], env.encode(env.state))

    def test_factory_uses_raw_states_for_multitaxi_propositions(self):
        env, get_propositions = create_environment(
            "MultiTaxi-v0",
            multitaxi_observation_mode="relative",
        )
        state, info = env.reset(seed=1)
        new_state, _, _, _, new_info = env.step(0)

        propositions = get_propositions(env, info["raw_state"], 0, new_info["raw_state"])

        self.assertEqual(state.shape, new_state.shape)
        self.assertIsInstance(propositions, list)


if __name__ == "__main__":
    unittest.main()
