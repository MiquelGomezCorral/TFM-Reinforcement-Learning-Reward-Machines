import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.envs import (
    MiniGridDiscreteWrapper,
    create_environment,
    get_propositions_doorkey,
    get_propositions_multi_taxi,
)
from src.config import Configuration
from src.models.QTable import QTable, QTableRM
from src.models.RewardMachine import RewardMachine
from src.models.train import train_qt


class QTableTest(unittest.TestCase):
    def test_environment_factory(self):
        taxi, propositions = create_environment(
            Configuration(gym_id="MultiTaxi-v0", multitaxi_grid_size=10)
        )
        self.assertIs(propositions, get_propositions_multi_taxi)
        self.assertEqual(taxi.observation_space.n, 40_000)
        taxi.close()

        doorkey, propositions = create_environment(
            Configuration(gym_id="MiniGrid-DoorKey-5x5-v0")
        )
        self.assertIsInstance(doorkey, MiniGridDiscreteWrapper)
        self.assertIs(propositions, get_propositions_doorkey)
        doorkey.close()

        with self.assertRaises(ValueError):
            create_environment(Configuration(gym_id="Unknown-v0"))

    def test_update_changes_greedy_action(self):
        q_table = QTable(None, 2)

        q_table.update("state", 1, 3, "next_state", True, 0.9, 1)

        self.assertEqual(q_table.greedy_policy("state"), 1)

    def test_greedy_policy_breaks_maximum_ties_randomly(self):
        q_table = QTable(None, 3)
        q_table.values("state")[:] = [2, 2, 1]
        np.random.seed(1)

        actions = {q_table.greedy_policy("state") for _ in range(20)}

        self.assertEqual(actions, {0, 1})

    def test_reward_machine_bootstraps_from_target_table(self):
        with tempfile.TemporaryDirectory() as models_path:
            Path(models_path, "machine.txt").write_text("i:0\nf:2\n0;1;a;3\n1;2;b;3\n")
            config = SimpleNamespace(MODELS_PATH=models_path, gamma=0.5, learning_rate=1)
            env = SimpleNamespace(
                action_space=SimpleNamespace(n=2, sample=lambda: 1),
                observation_space=SimpleNamespace(n=2),
            )
            q_table = QTableRM(config, env, "machine.txt")
            self.assertEqual(q_table.rm.simulate_step(0, set())[1], 0)
            q_table._q_table(1).update(1, 0, 4, 1, True, 0.9, 1)

            done = q_table.update(
                0, 1, 0, 0, 1, 1, env,
                lambda _, __, ___, ____: {"a"},
            )

            self.assertFalse(done)
            self.assertEqual(q_table._q_table(0).values(0)[1], 5)
            self.assertEqual(q_table.epsilon_greedy_policy(0, 1, env), 1)

    def test_reward_machine_terminal_target_does_not_bootstrap(self):
        with tempfile.TemporaryDirectory() as models_path:
            Path(models_path, "machine.txt").write_text("i:0\nf:1\n0;1;a;3\n")
            config = SimpleNamespace(MODELS_PATH=models_path, gamma=0.5, learning_rate=1)
            env = SimpleNamespace(
                action_space=SimpleNamespace(n=2, sample=lambda: 1),
                observation_space=SimpleNamespace(n=2),
            )
            q_table = QTableRM(config, env, "machine.txt")
            q_table._q_table(1).update(1, 0, 10, 1, True, 0.9, 1)

            done = q_table.update(
                0, 1, 0, 0, 1, 1, env,
                lambda _, __, ___, ____: {"a"},
            )

            self.assertTrue(done)
            self.assertEqual(q_table._q_table(0).values(0)[1], 3)

    def test_environment_terminal_target_does_not_bootstrap(self):
        with tempfile.TemporaryDirectory() as models_path:
            Path(models_path, "machine.txt").write_text("i:0\nf:2\n0;1;a;3\n1;2;b;3\n")
            config = SimpleNamespace(MODELS_PATH=models_path, gamma=0.5, learning_rate=1)
            env = SimpleNamespace(
                action_space=SimpleNamespace(n=2, sample=lambda: 1),
                observation_space=SimpleNamespace(n=2),
            )
            q_table = QTableRM(config, env, "machine.txt")
            q_table._q_table(1).update(1, 0, 10, 1, True, 0.9, 1)

            q_table.update(
                0, 1, 0, 0, 1, 1, env,
                lambda _, __, ___, ____: {"a"}, terminated=True,
            )

            self.assertEqual(q_table._q_table(0).values(0)[1], 3)

    def test_crm_updates_initial_reward_machine_state(self):
        with tempfile.TemporaryDirectory() as models_path:
            Path(models_path, "machine.txt").write_text("i:0\nf:2\n0;1;a;3\n1;2;b;4\n")
            config = SimpleNamespace(MODELS_PATH=models_path, gamma=0.5, learning_rate=1)
            env = SimpleNamespace(
                action_space=SimpleNamespace(n=2, sample=lambda: 1),
                observation_space=SimpleNamespace(n=2),
            )
            q_table = QTableRM(config, env, "machine.txt")

            q_table.update(
                0, 1, 0, 0, 1, 1, env,
                lambda _, __, ___, ____: {"a"}, use_crm=True,
            )

            self.assertEqual(q_table._q_table(0).values(0)[1], 3)

    def test_reward_machine_preserves_invalid_action_penalty(self):
        with tempfile.TemporaryDirectory() as models_path:
            Path(models_path, "machine.txt").write_text(
                "i:0\nf:2\nr:-1\n0;1;a;3\n1;2;b;4\n"
            )
            config = SimpleNamespace(MODELS_PATH=models_path, gamma=0.5, learning_rate=1)
            env = SimpleNamespace(
                action_space=SimpleNamespace(n=2, sample=lambda: 1),
                observation_space=SimpleNamespace(n=2),
            )

            for use_crm in (False, True):
                with self.subTest(use_crm=use_crm):
                    q_table = QTableRM(config, env, "machine.txt")
                    q_table.update(
                        0, 1, -10, 0, 0, 0, env,
                        lambda *_: set(),
                        use_crm=use_crm,
                        invalid_action=True,
                    )

                    self.assertEqual(q_table._q_table(0).values(0)[1], -10)

    def test_reward_machine_rejects_ambiguous_transitions(self):
        with tempfile.TemporaryDirectory() as models_path:
            Path(models_path, "machine.txt").write_text("i:0\nf:2\nr:-7\n0;1;a;3\n0;2;b;4\n1;2;z;1\n")
            config = SimpleNamespace(MODELS_PATH=models_path)
            machine = RewardMachine(config, "machine.txt")

            self.assertEqual(machine.simulate_step(0, set())[1], -7)
            with self.assertRaises(ValueError):
                machine.simulate_step(0, {"a", "b"})

    def test_reward_machine_deduplicates_identical_transitions(self):
        with tempfile.TemporaryDirectory() as models_path:
            Path(models_path, "machine.txt").write_text(
                "i:0\nf:2\nr:-1\n0;1;a;3\n0;1;a;3\n1;2;z;1\n"
            )
            config = SimpleNamespace(MODELS_PATH=models_path)
            machine = RewardMachine(config, "machine.txt")

            self.assertEqual(machine.simulate_step(0, {"a"}), (1, 3.0, False))

    def test_three_state_multitaxi_rm_requires_both_deliveries(self):
        models_path = Path(__file__).parents[1] / "models"
        machine = RewardMachine(
            SimpleNamespace(MODELS_PATH=models_path),
            "rm_taxi_2p_3s.txt",
        )

        self.assertEqual(machine.step({"d1"}), (1, 5.0, False))
        self.assertEqual(machine.step(set()), (1, -1.0, False))
        self.assertEqual(machine.step({"d2"}), (2, 5.0, True))

    def test_nine_state_multitaxi_rm_uses_transition_events(self):
        models_path = Path(__file__).parents[1] / "models"
        config = SimpleNamespace(MODELS_PATH=models_path)
        paths = (
            (({"p1"}, {"p2"}, {"d1"}, {"d2"}), (1, 3, 6, 8)),
            (({"p2"}, {"p1"}, {"d2"}, {"d1"}), (2, 3, 7, 8)),
            (({"p1"}, {"d1"}, {"p2"}, {"d2"}), (1, 4, 6, 8)),
            (({"p2"}, {"d2"}, {"p1"}, {"d1"}), (2, 5, 7, 8)),
            (({"p1", "p2"}, {"d1", "d2"}), (3, 8)),
        )

        for events, expected_states in paths:
            with self.subTest(events=events):
                machine = RewardMachine(config, "rm_taxi_2p_9s.txt")
                states = [machine.step(event)[0] for event in events]
                self.assertEqual(states, list(expected_states))
                self.assertEqual(machine.get_current_state(), machine.final_state)

        machine = RewardMachine(config, "rm_taxi_2p_9s.txt")
        self.assertEqual(machine.step({"p1"}), (1, 5.0, False))
        self.assertEqual(machine.step(set()), (1, -1.0, False))

    def test_reward_machine_rejects_non_final_dead_end(self):
        with tempfile.TemporaryDirectory() as models_path:
            Path(models_path, "machine.txt").write_text("i:0\nf:2\n0;1;a;3\n")
            config = SimpleNamespace(MODELS_PATH=models_path)

            with self.assertRaisesRegex(ValueError, "Non-final RM states"):
                RewardMachine(config, "machine.txt")

    def test_reward_machine_rejects_non_final_initial_dead_end(self):
        with tempfile.TemporaryDirectory() as models_path:
            Path(models_path, "machine.txt").write_text("i:0\nf:1\n")
            config = SimpleNamespace(MODELS_PATH=models_path)

            with self.assertRaisesRegex(ValueError, "Non-final RM states"):
                RewardMachine(config, "machine.txt")

    def test_reward_machine_current_state_persists(self):
        with tempfile.TemporaryDirectory() as models_path:
            Path(models_path, "machine.txt").write_text("i:0\nf:1\n0;1;a;3\n")
            machine = RewardMachine(SimpleNamespace(MODELS_PATH=models_path), "machine.txt")

            machine.step({"a"})

            self.assertEqual(machine.get_current_state(), 1)

    def test_training_uses_random_actions_during_learning_starts(self):
        class Environment:
            action_space = SimpleNamespace(seed=lambda _: None, sample=lambda: 0)

            def __init__(self):
                self.actions = []

            def reset(self, seed=None):
                self.steps = 0
                return 0, {}

            def step(self, action):
                self.actions.append(action)
                self.steps += 1
                return 0, 0, self.steps == 2, False, {}

        config = SimpleNamespace(
            seed=1,
            n_training_episodes=2,
            max_steps=2,
            min_epsilon=0,
            max_epsilon=0,
            decay_rate=0,
            qtable_learning_starts=1,
            gamma=0.9,
            learning_rate=0.1,
            use_crm=False,
        )
        env = Environment()
        q_table = QTableRM(config, env)
        q_table._q_table(0).values(0)[1] = 1

        train_qt(config, q_table, None, env)

        self.assertEqual(env.actions, [0, 0, 1, 1])


if __name__ == "__main__":
    unittest.main()
