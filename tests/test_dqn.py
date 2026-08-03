import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.models.DQN import DQN, ReplayMemory
from src.models.DQNRM import DQNRM
from src.models.train_dqn import train_dqn


class TwoStepEnvironment:
    action_space = SimpleNamespace(seed=lambda _: None, sample=lambda: 0)

    def reset(self, seed=None):
        self.steps = 0
        return np.array([1, 0], dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        return np.array([0, 1], dtype=np.float32), 1, self.steps == 2, False, {}


def training_config(**overrides):
    values = {
        "seed": 1,
        "n_training_episodes": 2,
        "max_steps": 2,
        "min_epsilon": 0,
        "max_epsilon": 0,
        "dqn_epsilon_decay_steps": 10,
        "dqn_optimize_interval": 1,
        "dqn_learning_starts": 0,
        "dqn_checkpoint_interval": 100,
        "dqn_validation_episodes": 10,
        "dqn_validation_seed_base": 7,
        "parse_state": None,
        "use_crm": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class DQNTest(unittest.TestCase):
    def test_replay_memory_discards_oldest_transition(self):
        memory = ReplayMemory(2)
        memory.push(np.array([0]), 0, 0, np.array([1]))
        memory.push(np.array([1]), 0, 0, np.array([2]))
        memory.push(np.array([2]), 0, 0, np.array([3]))

        self.assertEqual(len(memory), 2)
        self.assertEqual({transition.state[0] for transition in memory._transitions}, {1, 2})

    def test_replay_memory_stratifies_rewarding_transitions(self):
        memory = ReplayMemory(20)
        for index in range(10):
            memory.push(np.array([index]), 0, -1, np.array([index + 1]))
        memory.push(np.array([10]), 0, 5, np.array([11]))
        memory.push(np.array([11]), 0, 20, None)

        transitions = memory.sample(8)

        self.assertGreaterEqual(sum(transition.reward > 0 for transition in transitions), 2)

    def test_replay_memory_does_not_sample_evicted_rewarding_transitions(self):
        memory = ReplayMemory(2)
        memory.push(np.array([0]), 0, 1, np.array([1]))
        memory.push(np.array([1]), 0, -1, np.array([2]))
        memory.push(np.array([2]), 0, -1, np.array([3]))

        transitions = memory.sample(2)

        self.assertEqual({transition.state[0] for transition in transitions}, {1, 2})
        self.assertEqual(memory._rewarding_slots, [])
        self.assertEqual(memory._rewarding_indices, {})

    def test_replay_memory_stratification_does_not_duplicate_transitions(self):
        memory = ReplayMemory(4)
        for index, reward in enumerate((1, -1, -1, -1)):
            memory.push(np.array([index]), 0, reward, np.array([index + 1]))

        with (
            patch("src.models.DQN.random.sample", return_value=[1, 2]),
            patch("src.models.DQN.random.choice", return_value=0),
        ):
            transitions = memory.sample(2)

        self.assertEqual({transition.state[0] for transition in transitions}, {0, 2})

    def test_terminal_batch_optimizes_policy_network(self):
        torch.manual_seed(1)
        dqn = DQN(2, 2, 2, 10, 0.01, 0.9, 8, 0.005, 100)
        before = [parameter.detach().clone() for parameter in dqn.policy_net.parameters()]

        dqn.update(np.array([1, 0]), 1, 1, np.array([0, 1]), True)
        loss = dqn.update(np.array([0, 1]), 0, -1, np.array([1, 0]), True)

        self.assertIsNotNone(loss)
        self.assertTrue(any(
            not torch.equal(previous, current)
            for previous, current in zip(before, dqn.policy_net.parameters())
        ))

    def test_crm_stores_one_transition_per_rm_state(self):
        with tempfile.TemporaryDirectory() as models_path:
            Path(models_path, "machine.txt").write_text("i:0\nf:2\n0;1;a;1\n1;2;b;1\n")
            config = SimpleNamespace(
                MODELS_PATH=models_path,
                dqn_batch_size=10,
                dqn_replay_capacity=10,
                dqn_learning_rate=0.001,
                gamma=0.9,
                dqn_hidden_size=8,
                dqn_tau=0.005,
                dqn_gradient_clip=100,
                use_crm=True,
            )
            env = SimpleNamespace(
                observation_space=SimpleNamespace(shape=(2,)),
                action_space=SimpleNamespace(n=2),
            )
            agent = DQNRM(config, env, "machine.txt")

            done = agent.update(
                np.array([1, 0]),
                0,
                5,
                0,
                1,
                np.array([0, 1]),
                False,
                env,
                lambda *_: {"a"},
                use_crm=True,
            )

            self.assertFalse(done)
            self.assertEqual(len(agent.dqn.memory), 2)
            self.assertEqual(agent.dqn.batch_size, 20)
            self.assertEqual(agent.dqn.memory.rewarding_fraction, 0.25)

            agent.update(
                np.array([0, 1]),
                0,
                -1,
                1,
                2,
                np.array([1, 0]),
                False,
                env,
                lambda *_: {"b"},
                use_crm=True,
            )

            self.assertEqual(len(agent.dqn.memory), 3)
            self.assertEqual(agent._valid_crm_states, {0, 2})

            agent.update(
                np.array([1, 0]),
                0,
                -1,
                2,
                2,
                np.array([0, 1]),
                False,
                env,
                lambda *_: {"a"},
                use_crm=True,
            )

            self.assertEqual(len(agent.dqn.memory), 4)

    def test_crm_reachability_recovers_after_only_terminal_states_are_reachable(self):
        with tempfile.TemporaryDirectory() as models_path:
            Path(models_path, "machine.txt").write_text(
                "i:0\nf:2\n0;2;a;1\n1;2;a;1\n"
            )
            config = SimpleNamespace(
                MODELS_PATH=models_path,
                dqn_batch_size=10,
                dqn_replay_capacity=10,
                dqn_learning_rate=0.001,
                gamma=0.9,
                dqn_hidden_size=8,
                dqn_tau=0.005,
                dqn_gradient_clip=100,
                use_crm=True,
            )
            env = SimpleNamespace(
                observation_space=SimpleNamespace(shape=(2,)),
                action_space=SimpleNamespace(n=2),
            )
            agent = DQNRM(config, env, "machine.txt")
            update_args = (
                np.array([1, 0]), 0, -1, 0, 0, np.array([0, 1]),
                False, env,
            )

            agent.update(*update_args, lambda *_: {"a"}, use_crm=True)
            self.assertEqual(agent._valid_crm_states, {2})

            agent.update(*update_args, lambda *_: set(), use_crm=True)
            self.assertEqual(agent._valid_crm_states, {0, 1})

            agent.update(*update_args, lambda *_: set(), use_crm=True)
            self.assertEqual(len(agent.dqn.memory), 4)

    def test_crm_preserves_stronger_environment_penalties(self):
        with tempfile.TemporaryDirectory() as models_path:
            Path(models_path, "machine.txt").write_text("i:0\nf:1\n0;1;a;1\n")
            config = SimpleNamespace(
                MODELS_PATH=models_path,
                dqn_batch_size=10,
                dqn_replay_capacity=10,
                dqn_learning_rate=0.001,
                gamma=0.9,
                dqn_hidden_size=8,
                dqn_tau=0.005,
                dqn_gradient_clip=100,
                use_crm=True,
            )
            env = SimpleNamespace(
                observation_space=SimpleNamespace(shape=(2,)),
                action_space=SimpleNamespace(n=2),
            )
            agent = DQNRM(config, env, "machine.txt")

            agent.update(
                np.array([1, 0]), 0, -10, 0, 0, np.array([0, 1]),
                False, env, lambda *_: set(), use_crm=True,
            )

            self.assertEqual(agent.dqn.memory._transitions[0].reward, -10)

            agent.update(
                np.array([1, 0]), 0, -1, 0, 0, np.array([0, 1]),
                False, env, lambda *_: {"a"}, use_crm=True,
            )

            self.assertEqual(agent.dqn.memory._transitions[-1].reward, -1)

    def test_short_training_run_collects_transitions(self):
        config = training_config()
        agent = DQN(2, 2, 2, 10, 0.01, 0.9, 8, 0.005, 100)

        train_dqn(config, agent, None, TwoStepEnvironment())

        self.assertEqual(len(agent.memory), 4)

    def test_training_honors_learning_warmup_and_optimize_interval(self):
        config = training_config(dqn_optimize_interval=2, dqn_learning_starts=3)
        agent = DQN(2, 2, 10, 10, 0.01, 0.9, 8, 0.005, 100)
        optimize_calls = []
        agent.optimize = lambda: optimize_calls.append(True)

        train_dqn(config, agent, None, TwoStepEnvironment())

        self.assertEqual(len(optimize_calls), 1)

    def test_training_selects_best_validation_checkpoint(self):
        class Environment:
            action_space = SimpleNamespace(seed=lambda _: None, sample=lambda: 0)

            def __init__(self):
                self.episodes = 0

            def reset(self, seed=None):
                self.episodes += 1
                return np.array([1, 0], dtype=np.float32), {}

            def step(self, action):
                reward = 1 if action == 0 else 0
                return np.array([0, 1], dtype=np.float32), reward, True, False, {}

        config = training_config(n_training_episodes=250, max_steps=1)
        agent = DQN(2, 2, 10, 10, 0.01, 0.9, 8, 0.005, 100)

        def update(*_args, **_kwargs):
            with torch.no_grad():
                if environment.episodes > 200:
                    agent.policy_net.layers[-1].bias.copy_(torch.tensor([1.0, 0.0]))

        agent.update = update
        with torch.no_grad():
            for parameter in agent.policy_net.parameters():
                parameter.zero_()
            agent.policy_net.layers[-1].bias.copy_(torch.tensor([0.0, 1.0]))
        environment = Environment()

        train_dqn(config, agent, None, environment)

        self.assertEqual(agent.greedy_policy(np.array([1, 0])), 0)

if __name__ == "__main__":
    unittest.main()
