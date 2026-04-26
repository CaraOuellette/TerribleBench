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


if __name__ == "__main__":
    unittest.main()
