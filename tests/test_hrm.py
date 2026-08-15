import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.envs import create_environment
from src.models.QTableHRM import QTableHRM
from src.models.HRM import option_reward
from src.models.train_hrm import train_hrm


class ActionSpace:
    n = 1

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
        "decay_rate": 0,
        "gamma": 0.5,
        "learning_rate": 1,
        "hrm_r_plus": 1,
        "hrm_r_minus": -1,
        "hrm_q_init": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class HRMTest(unittest.TestCase):
    def test_option_reward_only_penalizes_failed_termination(self):
        self.assertEqual(option_reward(2, 1, 0, False, 3, -4), 2)
        self.assertEqual(option_reward(2, 1, 1, True, 3, -4), 5)
        self.assertEqual(option_reward(2, 1, 2, True, 3, -4), -2)

    def test_training_improves_the_greedy_option_policy(self):
        class LearningActionSpace:
            n = 2

            def seed(self, _seed):
                pass

            def sample(self):
                return 1

        class Environment:
            action_space = LearningActionSpace()

            def reset(self, seed=None):
                return 0, {}

            def step(self, action):
                return int(action == 1), 0, action == 1, False, {}

        with tempfile.TemporaryDirectory() as models_path:
            Path(models_path, "machine.txt").write_text(
                "i:0\nf:1\nr:0\n0;1;go;1\n"
            )
            env = Environment()
            test_config = config(
                models_path,
                max_steps=1,
                min_epsilon=1,
                max_epsilon=1,
            )
            model = QTableHRM(test_config, env, "machine.txt")
            np.testing.assert_array_equal(
                model.actor.values(model.actor_state(0, 0, 1)),
                [0, 0],
            )

            train_hrm(
                test_config,
                model,
                lambda _env, _state, action, _new_state: {"go"} if action == 1 else set(),
                env,
            )
            model.reset_rm()

            self.assertEqual(model.greedy_policy(0), 1)

    def test_multitaxi_reward_machine_progression(self):
        models_path = Path(__file__).resolve().parents[1] / "models"
        env, propositions = create_environment(
            SimpleNamespace(
                gym_id="MultiTaxi-v0",
                multitaxi_grid_size=5,
                multitaxi_num_passengers=2,
                multitaxi_observation_mode="discrete",
                multitaxi_reward_shaping=True,
                multitaxi_non_terminal_reward=-1,
            )
        )
        model = QTableHRM(config(models_path), env, "rm_taxi_2p.txt")
        transitions = (
            ([0, 0, 4, 1, 2, 3], 1, 5, False),
            ([0, 4, 1, 1, 2, 3], 4, 20, False),
            ([4, 0, 1, 1, 4, 3], 6, 5, False),
            ([4, 3, 1, 1, 3, 3], 8, 20, True),
        )

        try:
            for state, expected_u, expected_reward, expected_done in transitions:
                raw_state = env.unwrapped.encode(state)
                events = propositions(env, None, None, raw_state)
                self.assertEqual(
                    model.step_rm(events),
                    (expected_u, expected_reward, expected_done),
                )
        finally:
            env.close()

    def test_self_loop_only_state_is_rejected(self):
        with tempfile.TemporaryDirectory() as models_path:
            Path(models_path, "machine.txt").write_text(
                "i:0\nf:1\n0;0;a;0\n"
            )
            env = SimpleNamespace(action_space=SimpleNamespace(n=2))

            with self.assertRaisesRegex(ValueError, "without non-self options"):
                QTableHRM(config(models_path), env, "machine.txt")

    def test_options_are_masked_persistent_and_exclude_self_loops(self):
        with tempfile.TemporaryDirectory() as models_path:
            Path(models_path, "machine.txt").write_text(
                "i:0\nf:2\n0;1;a;0\n1;2;b;0\n"
            )
            env = SimpleNamespace(action_space=SimpleNamespace(n=2, sample=lambda: 0))
            model = QTableHRM(config(models_path, hrm_q_init=2), env, "machine.txt")

            self.assertEqual(model.options, {0: (1,), 1: (2,)})
            np.testing.assert_array_equal(
                model.high_level.values(model.high_state("start", 0)),
                np.array([2, 2, 2]),
            )

            start_values = model.high_level.values(model.high_state("start", 0))
            start_values[model.target_action(1)] = 3
            start_values[model.target_action(2)] = 100
            self.assertEqual(model.select_option("start", 0), 1)
            self.assertEqual(model.max_high_value("start", 0), 3)

            model.actor.values(model.actor_state("next", 0, 1))[1] = 5
            self.assertEqual(model.greedy_policy("next"), 1)
            self.assertEqual(model.active_option, 1)

            model.step_rm({"a"})
            self.assertIsNone(model.active_option)

    def test_training_uses_option_start_state_and_duration_discount(self):
        class Environment:
            action_space = ActionSpace()

            def reset(self, seed=None):
                self.steps = 0
                return 0, {}

            def step(self, _action):
                self.steps += 1
                return self.steps, 0, False, False, {}

        with tempfile.TemporaryDirectory() as models_path:
            Path(models_path, "machine.txt").write_text(
                "i:0\nf:1\nr:1\n0;1;go;2\n"
            )
            env = Environment()
            model = QTableHRM(config(models_path), env, "machine.txt")
            model.high_level.values(model.high_state(0, 0))[model.target_action(1)] = 1

            train_hrm(
                config(models_path),
                model,
                lambda _env, _state, _action, new_state: {"go"} if new_state == 2 else set(),
                env,
            )

            self.assertEqual(
                model.high_level.values(model.high_state(0, 0))[model.target_action(1)],
                2,
            )
            self.assertEqual(
                model.high_level.values(model.high_state(1, 0))[model.target_action(1)],
                0,
            )
            self.assertIn(model.actor_state(0, 0, 1), model.actor._table)
            self.assertEqual(
                model.actor.values(model.actor_state(0, 0, 1))[0],
                1,
            )
            self.assertEqual(
                model.actor.values(model.actor_state(1, 0, 1))[0],
                3,
            )

    def test_option_experiences_are_filtered_by_reachable_rm_state(self):
        class Environment:
            action_space = ActionSpace()

            def reset(self, seed=None):
                self.steps = 0
                return 0, {}

            def step(self, _action):
                self.steps += 1
                return self.steps, 0, False, self.steps == 2, {}

        with tempfile.TemporaryDirectory() as models_path:
            Path(models_path, "machine.txt").write_text(
                "i:0\nf:2\nr:0\n0;1;a;0\n1;2;b;0\n"
            )
            env = Environment()
            test_config = config(models_path, max_steps=2)
            model = QTableHRM(test_config, env, "machine.txt")

            train_hrm(test_config, model, lambda *_: {"a"}, env)

            self.assertNotIn(model.actor_state(1, 0, 1), model.actor._table)

    def test_counterfactual_option_bootstraps_while_its_rm_state_is_unchanged(self):
        class Environment:
            action_space = ActionSpace()

            def reset(self, seed=None):
                return 0, {}

            def step(self, _action):
                return 1, 0, False, True, {}

        with tempfile.TemporaryDirectory() as models_path:
            Path(models_path, "machine.txt").write_text(
                "i:0\nf:2\nr:0\n0;1;a;0\n1;2;b;0\n"
            )
            env = Environment()
            test_config = config(models_path, max_steps=1)
            model = QTableHRM(test_config, env, "machine.txt")
            model.actor.values(model.actor_state(1, 1, 2))[0] = 4

            train_hrm(test_config, model, lambda *_: set(), env)

            self.assertEqual(
                model.actor.values(model.actor_state(0, 1, 2))[0],
                2,
            )

    def test_time_limits_close_high_level_option_with_bootstrap(self):
        class Environment:
            action_space = ActionSpace()

            def __init__(self, truncated):
                self.truncated = truncated

            def reset(self, seed=None):
                return 0, {}

            def step(self, _action):
                return 1, 0, False, self.truncated, {}

        for truncated, max_steps in ((True, 2), (False, 1)):
            with self.subTest(truncated=truncated):
                with tempfile.TemporaryDirectory() as models_path:
                    Path(models_path, "machine.txt").write_text(
                        "i:0\nf:1\nr:1\n0;1;a;2\n"
                    )
                    env = Environment(truncated)
                    test_config = config(models_path, max_steps=max_steps)
                    model = QTableHRM(test_config, env, "machine.txt")
                    values = model.high_level.values(model.high_state(0, 0))
                    values[model.target_action(1)] = 5
                    next_values = model.high_level.values(model.high_state(1, 0))
                    next_values[model.target_action(0)] = 4

                    train_hrm(test_config, model, lambda *_: set(), env)

                    self.assertEqual(values[model.target_action(1)], 1)
                    self.assertIsNone(model.active_option)


if __name__ == "__main__":
    unittest.main()
