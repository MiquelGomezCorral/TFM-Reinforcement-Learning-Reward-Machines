import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.envs import (
    MiniGridDiscreteWrapper,
    create_environment,
    get_propositions_doorkey,
    get_propositions_taxi,
)
from src.models.QTable import QTable, QTableRM
from src.models.RewardMachine import RewardMachine


class QTableTest(unittest.TestCase):
    def test_environment_factory(self):
        taxi, propositions = create_environment("Taxi-v4")
        self.assertIs(propositions, get_propositions_taxi)
        taxi.close()

        doorkey, propositions = create_environment("MiniGrid-DoorKey-5x5-v0")
        self.assertIsInstance(doorkey, MiniGridDiscreteWrapper)
        self.assertIs(propositions, get_propositions_doorkey)
        doorkey.close()

        with self.assertRaises(ValueError):
            create_environment("Unknown-v0")

    def test_update_changes_greedy_action(self):
        q_table = QTable(None, 2)

        q_table.update("state", 1, 3, "next_state", True, 0.9, 1)

        self.assertEqual(q_table.greedy_policy("state"), 1)

    def test_reward_machine_bootstraps_from_target_table(self):
        with tempfile.TemporaryDirectory() as models_path:
            Path(models_path, "machine.txt").write_text("i:0\nf:2\n0;1;a;3\n1;2;b;3\n")
            config = SimpleNamespace(MODELS_PATH=models_path)
            env = SimpleNamespace(
                action_space=SimpleNamespace(n=2, sample=lambda: 1),
                observation_space=SimpleNamespace(n=2),
            )
            q_table = QTableRM(config, env, "machine.txt")
            self.assertEqual(q_table.rm.simulate_step(0, set())[1], 0)
            q_table._q_table(1).update(1, 0, 4, 1, True, 0.9, 1)

            done = q_table.update(
                0, 1, 0, 0, 1, 1, 0.5, 1, env,
                lambda _, __, ___, ____: {"a"},
            )

            self.assertFalse(done)
            self.assertEqual(q_table._q_table(0).values(0)[1], 5)
            self.assertEqual(q_table.epsilon_greedy_policy(0, 1, env), 1)

    def test_reward_machine_terminal_target_does_not_bootstrap(self):
        with tempfile.TemporaryDirectory() as models_path:
            Path(models_path, "machine.txt").write_text("i:0\nf:1\n0;1;a;3\n")
            config = SimpleNamespace(MODELS_PATH=models_path)
            env = SimpleNamespace(
                action_space=SimpleNamespace(n=2, sample=lambda: 1),
                observation_space=SimpleNamespace(n=2),
            )
            q_table = QTableRM(config, env, "machine.txt")
            q_table._q_table(1).update(1, 0, 10, 1, True, 0.9, 1)

            done = q_table.update(
                0, 1, 0, 0, 1, 1, 0.5, 1, env,
                lambda _, __, ___, ____: {"a"},
            )

            self.assertTrue(done)
            self.assertEqual(q_table._q_table(0).values(0)[1], 3)

    def test_environment_terminal_target_does_not_bootstrap(self):
        with tempfile.TemporaryDirectory() as models_path:
            Path(models_path, "machine.txt").write_text("i:0\nf:2\n0;1;a;3\n1;2;b;3\n")
            config = SimpleNamespace(MODELS_PATH=models_path)
            env = SimpleNamespace(
                action_space=SimpleNamespace(n=2, sample=lambda: 1),
                observation_space=SimpleNamespace(n=2),
            )
            q_table = QTableRM(config, env, "machine.txt")
            q_table._q_table(1).update(1, 0, 10, 1, True, 0.9, 1)

            q_table.update(
                0, 1, 0, 0, 1, 1, 0.5, 1, env,
                lambda _, __, ___, ____: {"a"}, terminated=True,
            )

            self.assertEqual(q_table._q_table(0).values(0)[1], 3)

    def test_crm_updates_initial_reward_machine_state(self):
        with tempfile.TemporaryDirectory() as models_path:
            Path(models_path, "machine.txt").write_text("i:0\nf:2\n0;1;a;3\n1;2;b;4\n")
            config = SimpleNamespace(MODELS_PATH=models_path)
            env = SimpleNamespace(
                action_space=SimpleNamespace(n=2, sample=lambda: 1),
                observation_space=SimpleNamespace(n=2),
            )
            q_table = QTableRM(config, env, "machine.txt")

            q_table.update(
                0, 1, 0, 0, 1, 1, 0.5, 1, env,
                lambda _, __, ___, ____: {"a"}, use_crm=True,
            )

            self.assertEqual(q_table._q_table(0).values(0)[1], 3)

    def test_reward_machine_rejects_ambiguous_transitions(self):
        with tempfile.TemporaryDirectory() as models_path:
            Path(models_path, "machine.txt").write_text("i:0\nf:2\nr:-7\n0;1;a;3\n0;2;b;4\n1;2;z;1\n")
            config = SimpleNamespace(MODELS_PATH=models_path)
            machine = RewardMachine(config, "machine.txt")

            self.assertEqual(machine.simulate_step(0, set())[1], -7)
            with self.assertRaises(ValueError):
                machine.simulate_step(0, {"a", "b"})

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


if __name__ == "__main__":
    unittest.main()
