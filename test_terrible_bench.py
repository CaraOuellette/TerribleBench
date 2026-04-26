import os
import unittest

from terrible_bench import build_task, parse_answer, run_benchmark


class TerribleBenchTests(unittest.TestCase):
    def test_parse_answer_prefers_final_answer_tag(self) -> None:
        self.assertEqual(parse_answer("thinking 999\nfinal answer: -3"), -3)
        self.assertEqual(parse_answer("FINAL ANSWER = 42"), 42)

    def test_task_generation_is_evaluable(self) -> None:
        import random

        rng = random.Random(123)
        task = build_task(rng, 0)
        self.assertTrue(task.prompt)
        self.assertIsInstance(task.answer, int)

    def test_demo_benchmark_runs_without_network(self) -> None:
        result = run_benchmark(
            {
                "targetModel": "my/demo-model",
                "comparisonModels": "tiny/free-model",
                "taskCount": 3,
                "seed": 17,
                "demoMode": True,
                "parallel": True,
                "includeWeenies": False,
                "pHack": True,
                "scaleHack": True,
            }
        )
        self.assertEqual(result["mode"], "demo")
        self.assertEqual(len(result["tasks"]), 3)
        self.assertGreaterEqual(len(result["displayedScores"]), 2)
        self.assertEqual(len(result["modelTiming"]), 2)
        self.assertIn("batch_wall_ms", result["modelTiming"][0])

    def test_progress_callback_reports_each_trial(self) -> None:
        events = []
        result = run_benchmark(
            {
                "targetModel": "my/demo-model",
                "comparisonModels": "tiny/free-model",
                "taskCount": 2,
                "seed": 19,
                "demoMode": True,
                "parallel": True,
                "includeWeenies": False,
            },
            progress_callback=events.append,
        )
        self.assertEqual(result["mode"], "demo")
        self.assertEqual(events[0]["type"], "setup")
        trial_events = [event for event in events if event["type"] == "trial_complete"]
        self.assertEqual(len(trial_events), 4)
        self.assertIn("finished_offset_ms", trial_events[0]["result"])

    def test_run_log_does_not_store_api_key(self) -> None:
        secret = "sk-do-not-log-this-test-value"
        result = run_benchmark(
            {
                "targetModel": "my/demo-model",
                "comparisonModels": "tiny/free-model",
                "apiKey": secret,
                "taskCount": 1,
                "seed": 23,
                "demoMode": True,
                "includeWeenies": False,
            }
        )
        self.assertIn("logFile", result)
        with open(result["logFile"], "r", encoding="utf-8") as handle:
            contents = handle.read()
        self.assertNotIn(secret, contents)
        self.assertTrue(os.path.exists(result["logFile"]))


if __name__ == "__main__":
    unittest.main()
