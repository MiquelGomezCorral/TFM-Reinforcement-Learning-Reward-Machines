import json
import os
import tempfile
import unittest
from pathlib import Path


class BenchmarkNotebookTest(unittest.TestCase):
    def load_notebook_scope(self, filename):
        notebook = json.loads(Path(filename).read_text(encoding="utf-8"))
        scope = {}
        working_directory = Path.cwd()
        os.chdir("app")
        self.addCleanup(os.chdir, working_directory)
        for cell_index in (1, 2, 3):
            exec("".join(notebook["cells"][cell_index]["source"]), scope)
        return notebook, scope

    def test_q_benchmark_rejects_a_different_specification(self):
        notebook, scope = self.load_notebook_scope("notebooks/MultiTaxi-5x5-Benchmark-q.ipynb")
        source = "".join(
            line
            for cell in notebook["cells"]
            for line in cell.get("source", [])
        )
        self.assertIn("from scripts import train_qt_hrm_agent, train_qt", source)
        self.assertIn("'qt': train_qt", source)
        self.assertIn("'qt_hrm': train_qt_hrm_agent", source)
        self.assertIn("'experiment_id': experiment['id']", source)
        self.assertIn("'reward_machine_id': experiment['reward_machine_id']", source)
        self.assertIn("multitaxi_reward_shaping=False", source)
        self.assertIn("results_file_lock", source)
        self.assertIn("ProcessPoolExecutor", source)
        self.assertNotIn("ROOT =", source)
        self.assertNotIn("sys.path.insert", source)
        self.assertNotIn("CONVERGENCE_PATH", source)
        self.assertNotIn("multitaxi_10x10", source)
        self.assertNotIn("create_environment(", source)
        self.assertNotIn("DQNRM(", source)
        self.assertNotIn("train_dqn(", source)

        with tempfile.TemporaryDirectory() as directory:
            scope["RESULTS_PATH"] = Path(directory, "results.json")
            scope["RESULTS_LOCK_PATH"] = Path(directory, "results.lock")
            scope["save_runs"]([])
            self.assertEqual(scope["load_runs"](), [])

            scope["RESULTS_PATH"].write_text(
                json.dumps({"spec": scope["BENCHMARK_SPEC"], "runs": None}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must be a list"):
                scope["load_runs"]()

            modes = {mode["id"]: mode for mode in scope["BENCHMARK_SPEC"]["modes"]}
            self.assertFalse(modes["qt"]["use_rm"])
            self.assertFalse(modes["rm"]["use_crm"])
            self.assertTrue(modes["crm"]["use_crm"])
            self.assertEqual(modes["rm"]["resolved"]["multitaxi_grid_size"], 10)
            self.assertIn("multitaxi_grid_size: 5", modes["rm"]["yaml"])
            self.assertFalse(scope["BENCHMARK_SPEC"]["reward_shaping"])
            self.assertEqual(scope["BENCHMARK_SPEC"]["version"], 6)
            self.assertTrue(all(
                "sha256" in reward_machine
                for reward_machine in scope["BENCHMARK_SPEC"]["reward_machines"]
            ))
            self.assertEqual(
                [reward_machine["file"] for reward_machine in scope["REWARD_MACHINES"]],
                [
                    "rm_taxi_2p_3s.txt",
                    "rm_taxi_2p_4s.txt",
                    "rm_taxi_2p_9s.txt",
                ],
            )
            self.assertEqual(len(scope["EXPERIMENTS"]), 10)
            self.assertEqual(scope["BENCHMARK"].parallel_workers, 4)
            self.assertEqual(
                scope["benchmark_config"](42, scope["EXPERIMENTS_BY_ID"]["qt"]).exp_name,
                "qt_baseline",
            )
            self.assertEqual(
                scope["benchmark_config"](42, scope["EXPERIMENTS_BY_ID"]["qt"]).VIDEO_PATH,
                scope["BENCHMARK"].VIDEO_PATH,
            )
            self.assertEqual(
                scope["benchmark_config"](42, scope["EXPERIMENTS_BY_ID"]["rm_3s"]).exp_name,
                "rm_3s",
            )
            scope["save_runs"]([])
            first = scope["new_run"](scope["EXPERIMENTS_BY_ID"]["qt"], 42)
            second = scope["new_run"](scope["EXPERIMENTS_BY_ID"]["qt"], 43)
            scope["upsert_run"](first)
            scope["upsert_run"](second)
            scope["upsert_run"](first)
            self.assertEqual(
                {(run["experiment_id"], run["seed"]) for run in scope["load_runs"]()},
                {("qt", 42), ("qt", 43)},
            )
            self.assertEqual(scope["BENCHMARK_SPEC"]["training_seeds"], list(range(42, 52)))

            scope["BENCHMARK_SPEC"] = {"changed": True}
            with self.assertRaisesRegex(ValueError, "benchmark specification"):
                scope["load_runs"]()

    def test_dqn_benchmark_uses_post_training_metrics(self):
        notebook, scope = self.load_notebook_scope("notebooks/MultiTaxi-5x5-Benchmark-dqn.ipynb")
        source = "".join(
            line
            for cell in notebook["cells"]
            for line in cell.get("source", [])
        )
        self.assertIn("from scripts import train_dqn_agent", source)
        self.assertNotIn("train_dqn_hrm_agent", source)
        self.assertNotIn("dqn_rm.yaml", source)
        self.assertEqual(scope["BENCHMARK_SPEC"]["algorithm"], "plain_dqn")
        self.assertEqual(scope["BENCHMARK_SPEC"]["resolved_config"]["dqn_hidden_size"], 128)
        self.assertEqual(scope["BENCHMARK_SPEC"]["resolved_config"]["multitaxi_num_passengers"], 1)
        self.assertNotIn("_LogTee", source)
        self.assertNotIn("RUNNERS", source)
        self.assertNotIn("pipeline_seconds", source)
        self.assertTrue(all(
            cell["execution_count"] is None and not cell["outputs"]
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ))

        convergence_metrics = {
            "successes": 1,
            "episodes": 1,
            "invalid_actions": 0,
            "mean_reward": 1.0,
            "reward_std": 0.0,
            "successful_std": 0.0,
            "mean_successful_steps": 2.0,
            "worst_reward": 1.0,
        }
        final_metrics = {**convergence_metrics, "mean_reward": 2.0}

        with tempfile.TemporaryDirectory() as directory:
            benchmark = scope["BENCHMARK"]
            benchmark.DATA_PATH = directory
            benchmark.VIDEO_PATH = str(Path(directory, "videos"))
            benchmark.training_seeds = (42,)
            training_episodes = benchmark.config_for(42).n_training_episodes
            benchmark.convergence_interval = training_episodes
            evaluation_seeds = []

            def evaluate(_config, _agent, _get_propositions, _env, **kwargs):
                evaluation_seeds.append(kwargs["seeds"])
                return convergence_metrics

            scope["evaluate_agent"] = evaluate
            scope["RUN_TRAINING"] = True

            def train(config, progress_callback):
                progress_callback(config.n_training_episodes, object(), None, None)
                return final_metrics

            scope["train_dqn_agent"] = train
            exec("".join(notebook["cells"][4]["source"]), scope)

            runs = scope["load_runs"]()
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["metrics"], final_metrics)
            self.assertEqual(runs[0]["convergence"][0]["metrics"], convergence_metrics)
            self.assertEqual(runs[0]["evaluation_seeds"], evaluation_seeds[0])
            self.assertEqual(scope["completed_runs"](), runs)

            partial_run = {**runs[0], "convergence": [{"episode": 3, "metrics": convergence_metrics}]}
            self.assertFalse(scope["completed_run"](partial_run))

            scope["save_runs"]([])
            with self.assertRaisesRegex(ValueError, "Benchmark is incomplete"):
                scope["completed_runs"]()

            scope["save_runs"](runs)
            scope["BENCHMARK"].results_path.write_text(
                json.dumps({"spec": {"changed": True}, "runs": runs}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "benchmark specification"):
                scope["load_runs"]()

            scope["save_runs"](runs)
            shown_traces = []
            scope["go"].Figure.show = lambda figure: shown_traces.append(len(figure.data))
            for cell_index in (5, 6, 7, 9):
                exec("".join(notebook["cells"][cell_index]["source"]), scope)
            self.assertEqual(shown_traces, [2, 1])

    def test_dqn_benchmark_uses_distinct_evaluation_seeds_per_run(self):
        _, scope = self.load_notebook_scope("notebooks/MultiTaxi-5x5-Benchmark-dqn.ipynb")
        benchmark = scope["BENCHMARK"]
        benchmark.training_seeds = (42, 43)

        first = benchmark.config_for(42)
        second = benchmark.config_for(43)

        self.assertNotEqual(first.eval_seed, second.eval_seed)
        self.assertTrue(set(first.eval_seed).isdisjoint(second.eval_seed))
        yaml_config = scope["Configuration"](yaml_config_path="dqn.yaml")
        self.assertEqual(first.dqn_learning_starts, yaml_config.dqn_learning_starts)
        self.assertEqual(first.dqn_optimize_starts, yaml_config.dqn_optimize_starts)
        self.assertEqual(first.dqn_tau, yaml_config.dqn_tau)
        self.assertEqual(first.multitaxi_num_passengers, yaml_config.multitaxi_num_passengers)
        self.assertEqual(first.multitaxi_grid_size, yaml_config.multitaxi_grid_size)
        self.assertEqual(first.n_eval_episodes, benchmark.n_eval_episodes)

        two_passenger_benchmark = scope["BenchmarkConfiguration"](
            training_seeds=(42,),
            num_passengers=2,
            n_training_episodes=100_000,
        )
        two_passenger_config = two_passenger_benchmark.config_for(42)
        self.assertEqual(two_passenger_config.multitaxi_num_passengers, 2)
        self.assertEqual(two_passenger_config.n_training_episodes, 100_000)
        self.assertIn("_2p_", two_passenger_benchmark.results_path.name)

        two_passenger_config_file = scope["BenchmarkConfiguration"](
            training_seeds=(42,),
            config_file="dqn_2p.yaml",
        )
        self.assertEqual(two_passenger_config_file.config_for(42).dqn_hidden_size, 256)
        self.assertEqual(two_passenger_config_file.config_for(42).multitaxi_num_passengers, 2)

        benchmark.config_file = "dqn_2p.yaml"
        self.assertEqual(scope["benchmark_spec"]()["resolved_config"]["dqn_hidden_size"], 256)


if __name__ == "__main__":
    unittest.main()
