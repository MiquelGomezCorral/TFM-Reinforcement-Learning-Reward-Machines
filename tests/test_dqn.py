import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.models.DQN import DQN, ReplayMemory
from src.models.DQNRM import DQNRM
from src.models.RewardMachine import RewardMachine
from src.models.train_dqn import train_dqn
from scripts.train_dqn import train_dqn_agent


class TwoStepEnvironment:
    action_space = SimpleNamespace(seed=lambda _: None, sample=lambda: 0)

    def reset(self, seed=None):
        self.steps = 0
        return np.array([1, 0], dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        return np.array([0, 1], dtype=np.float32), 1, self.steps == 2, False, {}


class EightWorkerEnvironment:
    num_envs = 8
    action_space = SimpleNamespace(
        seed=lambda _: None,
        sample=lambda: np.zeros(8, dtype=np.int64),
    )

    def reset(self, seed=None):
        return np.tile(np.array([1, 0], dtype=np.float32), (self.num_envs, 1)), {}

    def step(self, actions):
        if len(actions) != self.num_envs:
            raise ValueError("Expected one action per worker")
        return (
            np.tile(np.array([1, 0], dtype=np.float32), (self.num_envs, 1)),
            np.ones(self.num_envs, dtype=np.float32),
            np.ones(self.num_envs, dtype=bool),
            np.zeros(self.num_envs, dtype=bool),
            {"final_obs": np.tile(np.array([0, 1], dtype=np.float32), (self.num_envs, 1))},
        )


class ActionRecordingEnvironment(TwoStepEnvironment):
    def __init__(self):
        self.actions = []

    def step(self, action):
        self.actions.append(action)
        return super().step(action)


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
        "dqn_optimize_starts": 0,
        "dqn_num_envs": 1,
        "use_crm": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class DQNTest(unittest.TestCase):
    def test_one_passenger_reward_machine_completes_delivery(self):
        models_path = Path(__file__).resolve().parents[1] / "models"
        machine = RewardMachine(SimpleNamespace(MODELS_PATH=models_path), "rm_taxi_1p.txt")

        self.assertEqual(machine.step({"p"}), (1, 5.0, False))
        self.assertEqual(machine.step({"d", "del"}), (2, 20.0, True))

    def test_crm_requires_reward_machine(self):
        with self.assertRaisesRegex(ValueError, "use_crm requires use_rm=True"):
            train_dqn_agent(SimpleNamespace(use_crm=True, use_rm=False))

    def test_replay_memory_discards_oldest_transition(self):
        memory = ReplayMemory(2)
        memory.push(np.array([0]), 0, 0, np.array([1]))
        memory.push(np.array([1]), 0, 0, np.array([2]))
        memory.push(np.array([2]), 0, 0, np.array([3]))

        self.assertEqual(len(memory), 2)
        self.assertEqual({transition.state[0] for transition in memory._transitions}, {1, 2})

    def test_replay_memory_samples_uniformly(self):
        memory = ReplayMemory(4)
        for index, reward in enumerate((1, -1, -1, -1)):
            memory.push(np.array([index]), 0, reward, np.array([index + 1]))

        expected = [memory._transitions[1], memory._transitions[2]]
        with patch("src.models.DQN.random.sample", return_value=expected) as sample:
            transitions = memory.sample(2)

        sample.assert_called_once_with(memory._transitions, 2)
        self.assertEqual(transitions, expected)

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
            self.assertEqual(len(agent.dqn.memory), len(agent._rm_states))
            self.assertEqual(agent.dqn.batch_size, 20)

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

            self.assertEqual(len(agent.dqn.memory), 2 * len(agent._rm_states))

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

            self.assertEqual(len(agent.dqn.memory), 3 * len(agent._rm_states))

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

    def test_training_honors_optimize_starts_and_optimize_interval(self):
        config = training_config(dqn_optimize_interval=2, dqn_optimize_starts=3)
        agent = DQN(2, 2, 10, 10, 0.01, 0.9, 8, 0.005, 100)
        optimize_calls = []
        agent.optimize = lambda: optimize_calls.append(True)

        train_dqn(config, agent, None, TwoStepEnvironment())

        self.assertEqual(len(optimize_calls), 1)

    def test_training_uses_random_actions_during_learning_starts(self):
        config = training_config(dqn_learning_starts=2)
        agent = DQN(2, 2, 10, 10, 0.01, 0.9, 8, 0.005, 100)
        with torch.no_grad():
            for parameter in agent.policy_net.parameters():
                parameter.zero_()
            agent.policy_net.layers[-1].bias[1] = 1
        env = ActionRecordingEnvironment()

        train_dqn(config, agent, None, env)

        self.assertEqual(env.actions, [0, 0, 1, 1])

    def test_vector_training_matches_scalar_update_cadence(self):
        config = training_config(
            n_training_episodes=16,
            dqn_num_envs=8,
            dqn_optimize_interval=4,
        )
        agent = DQN(2, 2, 10, 16, 0.01, 0.9, 8, 0.005, 100)
        optimize_calls = []
        agent.optimize = lambda: optimize_calls.append(True)

        train_dqn(config, agent, None, EightWorkerEnvironment())

        self.assertEqual(len(agent.memory), 16)
        self.assertEqual(optimize_calls, [True, True, True, True])

    def test_train_dqn_agent_returns_post_training_metrics(self):
        environment = SimpleNamespace(
            observation_space=SimpleNamespace(shape=(2,)),
            action_space=SimpleNamespace(n=2),
            close=lambda: None,
        )
        config = SimpleNamespace(
            use_crm=False,
            use_rm=False,
            dqn_num_envs=1,
            seed=1,
            dqn_batch_size=2,
            dqn_replay_capacity=10,
            dqn_learning_rate=0.01,
            gamma=0.9,
            dqn_hidden_size=8,
            dqn_tau=0.005,
            dqn_gradient_clip=100,
            multitaxi_grid_size=5,
            exp_name="test",
        )
        metrics = {
            "successes": 1,
            "episodes": 1,
            "invalid_actions": 0,
            "mean_reward": 1.0,
            "reward_std": 0.0,
            "successful_std": 0.0,
            "mean_successful_steps": 1.0,
            "worst_reward": 1.0,
        }

        with (
            patch("scripts.train_dqn.create_environment", return_value=(environment, None)),
            patch("scripts.train_dqn.train_dqn"),
            patch("scripts.train_dqn.evaluate_agent", return_value=metrics) as evaluate,
            patch("scripts.train_dqn.record_video"),
        ):
            self.assertEqual(train_dqn_agent(config), metrics)

        self.assertTrue(evaluate.call_args.kwargs["return_metrics"])

    def test_training_keeps_final_policy(self):
        class Environment:
            action_space = SimpleNamespace(seed=lambda _: None, sample=lambda: 0)

            def __init__(self):
                self.episodes = 0

            def reset(self, seed=None):
                self.episodes += 1
                return np.array([1, 0], dtype=np.float32), {}

            def step(self, action):
                reward = 1 if action == 1 else 0
                return np.array([0, 1], dtype=np.float32), reward, True, False, {}

        config = training_config(n_training_episodes=250, max_steps=1)
        agent = DQN(2, 2, 10, 10, 0.01, 0.9, 8, 0.005, 100)

        def update(*_args, **_kwargs):
            with torch.no_grad():
                if environment.episodes > 200:
                    agent.policy_net.layers[-1].bias.copy_(torch.tensor([0.0, 1.0]))

        agent.update = update
        with torch.no_grad():
            for parameter in agent.policy_net.parameters():
                parameter.zero_()
            agent.policy_net.layers[-1].bias.copy_(torch.tensor([1.0, 0.0]))
        environment = Environment()

        train_dqn(config, agent, None, environment)

        self.assertEqual(agent.greedy_policy(np.array([1, 0])), 1)

    def test_reward_machine_training_restores_checkpoints(self):
        with tempfile.TemporaryDirectory() as models_path:
            Path(models_path, "machine.txt").write_text("i:0\nf:1\n0;1;a;1\n")
            config = training_config(
                dqn_checkpoint_interval=1,
                dqn_validation_episodes=1,
                dqn_validation_seed_base=7,
            )
            config.MODELS_PATH = models_path
            config.dqn_batch_size = 2
            config.dqn_replay_capacity = 10
            config.dqn_learning_rate = 0.01
            config.gamma = 0.9
            config.dqn_hidden_size = 8
            config.dqn_tau = 0.005
            config.dqn_gradient_clip = 100
            environment = TwoStepEnvironment()
            environment.observation_space = SimpleNamespace(shape=(2,))
            environment.action_space.n = 2
            agent = DQNRM(config, environment, "machine.txt")

            with patch("src.models.train_dqn.restore_best_checkpoint") as restore:
                train_dqn(config, agent, lambda *_: set(), environment)

        restore.assert_called_once()

if __name__ == "__main__":
    unittest.main()
