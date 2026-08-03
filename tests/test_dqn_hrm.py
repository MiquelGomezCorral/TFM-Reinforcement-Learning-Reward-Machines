import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.models.DQNHRM import DQNHRM
from src.models.train_dqn_hrm import train_dqn_hrm


class ActionSpace:
    def __init__(self, n=1):
        self.n = n

    def seed(self, _seed):
        pass

    def sample(self):
        return 0


def config(models_path, **overrides):
    values = {
        "MODELS_PATH": models_path,
        "seed": 1,
        "n_training_episodes": 1,
        "max_steps": 3,
        "min_epsilon": 0,
        "max_epsilon": 0,
        "dqn_epsilon_decay_steps": 10,
        "dqn_optimize_interval": 1,
        "dqn_learning_starts": 0,
        "dqn_checkpoint_interval": 100,
        "dqn_validation_episodes": 1,
        "dqn_validation_seed_base": 7,
        "gamma": 0.5,
        "hrm_r_plus": 1,
        "hrm_r_minus": -1,
        "dqn_batch_size": 100,
        "dqn_replay_capacity": 1000,
        "dqn_learning_rate": 0.001,
        "dqn_hidden_size": 8,
        "dqn_tau": 0.005,
        "dqn_gradient_clip": 100,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def set_outputs(network, values):
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        network.layers[-1].bias.copy_(torch.tensor(values, dtype=torch.float32))


class DQNHRMTest(unittest.TestCase):
    def test_masks_options_and_encodes_final_target(self):
        with tempfile.TemporaryDirectory() as models_path:
            Path(models_path, "machine.txt").write_text(
                "i:0\nf:2\n0;1;a;0\n1;2;b;0\n"
            )
            env = SimpleNamespace(
                observation_space=SimpleNamespace(shape=(2,)),
                action_space=ActionSpace(2),
            )
            agent = DQNHRM(config(models_path), env, "machine.txt")
            set_outputs(agent.high_level.policy_net, [1, 3, 100])
            set_outputs(agent.actor.policy_net, [0, 5])

            self.assertEqual(agent.options, {0: (1,), 1: (2,)})
            self.assertEqual(agent.high_level.batch_size, 100)
            self.assertEqual(agent.actor.batch_size, 200)
            self.assertEqual(agent.high_level.memory.capacity, 1000)
            self.assertEqual(agent.actor.memory.capacity, 2000)
            self.assertEqual(agent.high_state([1, 0], 0).shape, (4,))
            self.assertEqual(agent.actor_state([1, 0], 1, 2).shape, (7,))
            self.assertEqual(agent.select_option([1, 0], 0), 1)
            self.assertEqual(agent.greedy_policy([0, 1]), 1)

            agent.step_rm({"a"})
            self.assertIsNone(agent.active_option)

    def test_training_stores_counterfactual_and_smdp_experience(self):
        class Environment:
            observation_space = SimpleNamespace(shape=(2,))
            action_space = ActionSpace()

            def reset(self, seed=None):
                self.steps = 0
                return np.array([1, 0], dtype=np.float32), {}

            def step(self, _action):
                self.steps += 1
                state = np.array([0, self.steps], dtype=np.float32)
                return state, 0, False, False, {}

        with tempfile.TemporaryDirectory() as models_path:
            Path(models_path, "machine.txt").write_text(
                "i:0\nf:1\nr:1\n0;1;go;2\n"
            )
            env = Environment()
            agent = DQNHRM(config(models_path), env, "machine.txt")
            set_outputs(agent.high_level.policy_net, [0, 1])

            train_dqn_hrm(
                config(models_path),
                agent,
                lambda _env, _state, _action, new_state: (
                    {"go"} if new_state[1] == 2 else set()
                ),
                env,
            )

            actor_transitions = list(agent.actor.memory._transitions)
            self.assertEqual(len(actor_transitions), 2)
            self.assertEqual([transition.reward for transition in actor_transitions], [1, 3])
            self.assertIsNotNone(actor_transitions[0].next_state)
            self.assertIsNone(actor_transitions[-1].next_state)

            high_transitions = list(agent.high_level.memory._transitions)
            self.assertEqual(len(high_transitions), 1)
            self.assertEqual(high_transitions[0].reward, 2)
            self.assertIsNone(high_transitions[0].next_state)

    def test_high_level_bootstrap_ignores_invalid_targets(self):
        class Environment:
            observation_space = SimpleNamespace(shape=(2,))
            action_space = ActionSpace()

            def reset(self, seed=None):
                return np.array([1, 0], dtype=np.float32), {}

            def step(self, _action):
                return np.array([0, 1], dtype=np.float32), 0, False, True, {}

        with tempfile.TemporaryDirectory() as models_path:
            Path(models_path, "machine.txt").write_text(
                "i:0\nf:2\nr:1\n0;1;a;0\n1;2;b;0\n"
            )
            env = Environment()
            agent = DQNHRM(config(models_path), env, "machine.txt")
            set_outputs(agent.high_level.policy_net, [0, 1, 100])
            set_outputs(agent.high_level.target_net, [4, 3, 100])
            actor_optimizations = []
            high_optimizations = []
            agent.actor.optimize = lambda: actor_optimizations.append(True)
            agent.high_level.optimize = lambda: high_optimizations.append(True)

            train_dqn_hrm(config(models_path), agent, lambda *_: set(), env)

            transition = agent.high_level.memory._transitions[0]
            self.assertEqual(transition.reward, 2.5)
            self.assertEqual(actor_optimizations, [True])
            self.assertEqual(high_optimizations, [True])

    def test_training_preserves_invalid_action_penalty(self):
        class Environment:
            observation_space = SimpleNamespace(shape=(2,))
            action_space = ActionSpace()

            def reset(self, seed=None):
                return np.array([1, 0], dtype=np.float32), {}

            def step(self, _action):
                return np.array([1, 0], dtype=np.float32), -10, False, False, {
                    "invalid_action": True,
                }

        with tempfile.TemporaryDirectory() as models_path:
            Path(models_path, "machine.txt").write_text(
                "i:0\nf:1\nr:1\n0;1;go;2\n"
            )
            env = Environment()
            agent = DQNHRM(config(models_path), env, "machine.txt")

            train_dqn_hrm(
                config(models_path, max_steps=1, hrm_r_minus=-50),
                agent,
                lambda *_: set(),
                env,
            )

            transition = agent.actor.memory._transitions[0]
            self.assertEqual(transition.reward, -10)
            self.assertIsNone(transition.next_state)

    def test_training_selects_best_validation_checkpoint(self):
        class Environment:
            observation_space = SimpleNamespace(shape=(2,))
            action_space = ActionSpace(2)

            def reset(self, seed=None):
                return np.array([1, 0], dtype=np.float32), {}

            def step(self, _action):
                return np.array([0, 1], dtype=np.float32), 0, False, False, {}

        with tempfile.TemporaryDirectory() as models_path:
            Path(models_path, "machine.txt").write_text(
                "i:0\nf:1\nr:1\n0;1;go;2\n"
            )
            env = Environment()
            agent = DQNHRM(config(models_path), env, "machine.txt")
            actor_optimizations = 0
            high_optimizations = 0

            def optimize_actor():
                nonlocal actor_optimizations
                actor_optimizations += 1
                set_outputs(
                    agent.actor.policy_net,
                    [1, 0] if actor_optimizations == 1 else [0, 1],
                )

            def optimize_high():
                nonlocal high_optimizations
                high_optimizations += 1
                set_outputs(
                    agent.high_level.policy_net,
                    [0, 1] if high_optimizations == 1 else [1, 0],
                )

            def evaluate(*_args, **_kwargs):
                actor_values = agent.actor.q_values(agent.actor_state([1, 0], 0, 1))
                high_values = agent.high_level.q_values(agent.high_state([1, 0], 0))
                successful = actor_values[0] > actor_values[1] and high_values[1] > high_values[0]
                return {
                    "successes": int(successful),
                    "invalid_actions": 0,
                    "mean_reward": int(successful),
                }

            agent.actor.optimize = optimize_actor
            agent.high_level.optimize = optimize_high
            with patch("src.models.train_dqn_hrm.evaluate_agent", side_effect=evaluate):
                train_dqn_hrm(
                    config(
                        models_path,
                        n_training_episodes=2,
                        max_steps=1,
                        dqn_checkpoint_interval=1,
                    ),
                    agent,
                    lambda *_: set(),
                    env,
                )

            self.assertEqual(
                agent.actor.greedy_policy(agent.actor_state([1, 0], 0, 1)),
                0,
            )
            self.assertEqual(
                int(np.argmax(agent.high_level.q_values(agent.high_state([1, 0], 0)))),
                1,
            )


if __name__ == "__main__":
    unittest.main()
