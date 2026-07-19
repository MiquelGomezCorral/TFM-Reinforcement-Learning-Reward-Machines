import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.models.DQN import DQN, ReplayMemory
from src.models.DQNRM import DQNRM
from src.models.train_dqn import train_dqn


class DQNTest(unittest.TestCase):
    def test_replay_memory_discards_oldest_transition(self):
        memory = ReplayMemory(2)
        memory.push(np.array([0]), 0, 0, np.array([1]))
        memory.push(np.array([1]), 0, 0, np.array([2]))
        memory.push(np.array([2]), 0, 0, np.array([3]))

        self.assertEqual(len(memory), 2)
        self.assertEqual(memory._transitions[0].state[0], 1)

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
            )
            env = SimpleNamespace(
                observation_space=SimpleNamespace(shape=(2,)),
                action_space=SimpleNamespace(n=2),
            )
            agent = DQNRM(config, env, "machine.txt")

            done = agent.update(
                np.array([1, 0]),
                0,
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

    def test_short_training_run_collects_transitions(self):
        class Environment:
            action_space = SimpleNamespace(seed=lambda _: None, sample=lambda: 0)

            def reset(self, seed=None):
                self.steps = 0
                return np.array([1, 0], dtype=np.float32), {}

            def step(self, action):
                self.steps += 1
                return np.array([0, 1], dtype=np.float32), 1, self.steps == 2, False, {}

        config = SimpleNamespace(
            seed=1,
            n_training_episodes=2,
            max_steps=2,
            min_epsilon=0,
            max_epsilon=0,
            dqn_epsilon_decay_steps=10,
            use_crm=False,
        )
        agent = DQN(2, 2, 2, 10, 0.01, 0.9, 8, 0.005, 100)

        train_dqn(config, agent, None, Environment())

        self.assertEqual(len(agent.memory), 4)


if __name__ == "__main__":
    unittest.main()
