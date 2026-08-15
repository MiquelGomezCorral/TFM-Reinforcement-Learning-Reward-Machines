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
        self.assertIn("'variant_id': variant['id']", source)
        self.assertIn("'kind': variant['kind']", source)
        self.assertIn("'config': variant['config']", source)
        self.assertNotIn("ROOT =", source)
        self.assertNotIn("sys.path.insert", source)
        self.assertNotIn("CONVERGENCE_PATH", source)
        self.assertNotIn("multitaxi_10x10", source)
        self.assertNotIn("create_environment(", source)
        self.assertNotIn("DQNRM(", source)
        self.assertNotIn("train_dqn(", source)

        with tempfile.TemporaryDirectory() as directory:
            scope["RESULTS_PATH"] = Path(directory, "results.json")
            scope["save_runs"]([])
            self.assertEqual(scope["load_runs"](), [])

            scope["RESULTS_PATH"].write_text(
                json.dumps({"spec": scope["BENCHMARK_SPEC"], "runs": None}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must be a list"):
                scope["load_runs"]()

            variants = {variant["id"]: variant for variant in scope["BENCHMARK_SPEC"]["variants"]}
            self.assertFalse(variants["qt"]["use_rm"])
            self.assertFalse(variants["qt_rm"]["use_crm"])
            self.assertTrue(variants["qt_crm"]["use_crm"])
            self.assertEqual(variants["qt_rm"]["resolved"]["multitaxi_grid_size"], 10)
            self.assertIn("multitaxi_grid_size: 5", variants["qt_rm"]["yaml"])
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
        self.assertIn("class BenchmarkVariant", source)
        self.assertNotIn("_LogTee", source)
        self.assertNotIn("RUNNERS", source)
        self.assertNotIn("BENCHMARK_SPEC", source)
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
            benchmark.n_training_episodes = 1
            benchmark.convergence_interval = 1
            benchmark.training_seeds = (42,)
            benchmark.variants = (benchmark.variants[0],)
            evaluation_seeds = []

            def evaluate(_config, _agent, _get_propositions, _env, **kwargs):
                evaluation_seeds.append(kwargs["seeds"])
                return convergence_metrics

            scope["evaluate_agent"] = evaluate

            def train(_config, progress_callback):
                progress_callback(1, object(), None, None)
                return final_metrics

            scope["train_dqn_agent"] = train
            exec("".join(notebook["cells"][4]["source"]), scope)

            runs = scope["load_runs"]()
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["metrics"], final_metrics)
            self.assertEqual(runs[0]["convergence"][0]["metrics"], convergence_metrics)
            self.assertEqual(runs[0]["evaluation_seeds"], evaluation_seeds[0])
            self.assertEqual(scope["completed_runs"](), runs)

            benchmark.n_training_episodes = 3
            benchmark.convergence_interval = 2
            partial_run = {**runs[0], "convergence": [{"episode": 3, "metrics": convergence_metrics}]}
            self.assertFalse(scope["completed_run"](partial_run))
            benchmark.n_training_episodes = 1
            benchmark.convergence_interval = 1

            scope["save_runs"]([])
            with self.assertRaisesRegex(ValueError, "Benchmark is incomplete"):
                scope["completed_runs"]()

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
        benchmark.variants = (benchmark.variants[0],)

        first = benchmark.config_for(42, benchmark.variants[0])
        second = benchmark.config_for(43, benchmark.variants[0])

        self.assertNotEqual(first.eval_seed, second.eval_seed)
        self.assertTrue(set(first.eval_seed).isdisjoint(second.eval_seed))
        self.assertEqual(first.dqn_learning_starts, benchmark.dqn_learning_starts)
        self.assertEqual(first.dqn_optimize_starts, benchmark.dqn_optimize_starts)
        self.assertEqual(first.dqn_tau, benchmark.dqn_tau)


if __name__ == "__main__":
    unittest.main()
