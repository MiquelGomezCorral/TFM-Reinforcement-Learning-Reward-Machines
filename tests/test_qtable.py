import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.models.QTable import QTable, QTableRM


class QTableTest(unittest.TestCase):
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
            q_table._q_table(1).update(1, 0, 4, 1, True, 0.9, 1)

            done = q_table.update(
                0, 1, 0, 1, 1, 0.5, 1, env, lambda _, __: {"a"}
            )

            self.assertFalse(done)
            self.assertEqual(q_table._q_table(0)._values(0)[1], 5)
            self.assertEqual(q_table.epsilon_greedy_policy(0, 1, env), 1)


if __name__ == "__main__":
    unittest.main()
