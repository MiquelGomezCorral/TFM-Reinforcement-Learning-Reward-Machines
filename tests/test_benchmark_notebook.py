import json
import os
import tempfile
import unittest
from pathlib import Path

class BenchmarkNotebookTest(unittest.TestCase):
    def load_notebook_scope(self):
        notebook = json.loads(Path("notebooks/MultiTaxi-5x5-Benchmark.ipynb").read_text(encoding="utf-8"))
        scope = {}
        working_directory = Path.cwd()
        os.chdir("app")
        self.addCleanup(os.chdir, working_directory)
        for cell_index in (1, 2, 3):
            exec("".join(notebook["cells"][cell_index]["source"]), scope)
        return notebook, scope

    def test_results_reject_a_different_benchmark_spec(self):
        notebook, scope = self.load_notebook_scope()
        source = "".join(
            line
            for cell in notebook["cells"]
            for line in cell.get("source", [])
        )
        self.assertIn(
            "from scripts import train_dqn_agent, train_hrm_agent, train_qt",
            source,
        )
        self.assertIn("'qtable': train_qt", source)
        self.assertIn("'hrm': train_hrm_agent", source)
        self.assertIn("'dqn': train_dqn_agent", source)
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
            self.assertFalse(variants["qlearning"]["use_rm"])
            self.assertFalse(variants["qrm"]["use_crm"])
            self.assertTrue(variants["qrm_crm"]["use_crm"])
            self.assertEqual(variants["qrm"]["resolved"]["multitaxi_grid_size"], 10)
            self.assertIn("multitaxi_grid_size: 5", variants["qrm"]["yaml"])
            self.assertEqual(scope["BENCHMARK_SPEC"]["training_seeds"], list(range(42, 52)))

            scope["BENCHMARK_SPEC"] = {"changed": True}
            with self.assertRaisesRegex(ValueError, "benchmark specification"):
                scope["load_runs"]()

    def test_rerun_replaces_run_and_persists_step_metrics(self):
        notebook, scope = self.load_notebook_scope()

        with tempfile.TemporaryDirectory() as directory:
            scope["RESULTS_PATH"] = Path(directory, "results.json")
            scope["CONVERGENCE_PATH"] = Path(directory, "convergence.json")
            scope["BENCHMARK"].VIDEO_PATH = str(Path(directory, "videos"))
            scope["BENCHMARK"].n_training_episodes = 1
            scope["BENCHMARK"].convergence_interval = 1
            scope["TRAINING_SEEDS"] = (42,)
            scope["VARIANTS"] = (scope["VARIANTS"][0],)
            scope["VARIANTS_BY_ID"] = {"qlearning": scope["VARIANTS"][0]}
            scope["evaluate_agent"] = lambda *_args, **_kwargs: {
                "successes": 1,
                "episodes": 1,
                "invalid_actions": 0,
                "mean_reward": 1.0,
                "reward_std": 0.0,
                "successful_std": 0.0,
                "mean_successful_steps": 2.0,
                "worst_reward": 1.0,
            }
            scope["RUNNERS"] = {
                "qtable": lambda _config, progress_callback: (
                    progress_callback(1, object(), None, None),
                    (1.0, 0.0),
                )[1],
            }

            exec("".join(notebook["cells"][4]["source"]), scope)
            exec("".join(notebook["cells"][4]["source"]), scope)

            runs = scope["load_runs"]()
            self.assertEqual(len(runs), 1)
            self.assertEqual(len(runs[0]["convergence"]), 1)
            self.assertEqual(runs[0]["metrics"]["mean_successful_steps"], 2.0)

            def fail_training(*_args, **_kwargs):
                raise RuntimeError("training failed")

            scope["RUNNERS"] = {"qtable": fail_training}
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                exec("".join(notebook["cells"][4]["source"]), scope)
            self.assertEqual(scope["load_runs"]()[0]["metrics"]["mean_successful_steps"], 2.0)

            shown_traces = []
            scope["go"].Figure.show = lambda figure: shown_traces.append(len(figure.data))
            for cell_index in (5, 6, 7):
                exec("".join(notebook["cells"][cell_index]["source"]), scope)
            self.assertEqual(shown_traces, [2, 1])


if __name__ == "__main__":
    unittest.main()
