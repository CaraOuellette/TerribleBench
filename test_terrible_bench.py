import os
import random
import time
import unittest

import terrible_bench
from terrible_bench import build_task, parse_answer, rerun_target_failures, run_benchmark


class TerribleBenchTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_run_log_dir = terrible_bench.RUN_LOG_DIR
        terrible_bench.RUN_LOG_DIR = os.path.join(terrible_bench.BASE_DIR, "test_run_logs")
        os.makedirs(terrible_bench.RUN_LOG_DIR, exist_ok=True)

    def tearDown(self) -> None:
        terrible_bench.RUN_LOG_DIR = self._old_run_log_dir

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
        self.assertEqual(events[0]["type"], "health_setup")
        setup_events = [event for event in events if event["type"] == "setup"]
        self.assertEqual(len(setup_events), 1)
        trial_events = [event for event in events if event["type"] == "trial_complete"]
        self.assertEqual(len(trial_events), 4)
        health_events = [event for event in events if event["type"] == "health_complete"]
        self.assertEqual(len(health_events), 2)
        self.assertIn("finished_offset_ms", trial_events[0]["result"])

    def test_preflight_filters_failed_comparison_models(self) -> None:
        original = terrible_bench.run_model_health_check

        def fake_health(model, api_key, demo_mode, temperature, timeout_seconds):
            return terrible_bench.ModelHealthCheck(
                model=model,
                ok=model != "bad/rival",
                output="final answer: 1" if model != "bad/rival" else "",
                parsed=1 if model != "bad/rival" else None,
                expected=1,
                latency_ms=5,
                error=None if model != "bad/rival" else "model unavailable",
            )

        terrible_bench.run_model_health_check = fake_health
        try:
            result = run_benchmark(
                {
                    "targetModel": "my/demo-model",
                    "comparisonModels": "bad/rival\ntiny/free-model",
                    "taskCount": 1,
                    "seed": 37,
                    "demoMode": True,
                    "parallel": True,
                    "includeWeenies": False,
                }
            )
        finally:
            terrible_bench.run_model_health_check = original

        self.assertNotIn("bad/rival", result["models"])
        self.assertIn("bad/rival", result["preflight"]["skippedComparisonModels"])
        self.assertTrue(all(item["model"] != "bad/rival" for item in result["results"]))

    def test_preflight_target_failure_aborts(self) -> None:
        original = terrible_bench.run_model_health_check

        def fake_health(model, api_key, demo_mode, temperature, timeout_seconds):
            return terrible_bench.ModelHealthCheck(
                model=model,
                ok=False,
                output="",
                parsed=None,
                expected=1,
                latency_ms=5,
                error="target unavailable",
            )

        terrible_bench.run_model_health_check = fake_health
        try:
            with self.assertRaisesRegex(RuntimeError, "Target model failed preflight"):
                run_benchmark(
                    {
                        "targetModel": "my/demo-model",
                        "comparisonModels": "tiny/free-model",
                        "taskCount": 1,
                        "seed": 41,
                        "demoMode": True,
                        "includeWeenies": False,
                    }
                )
        finally:
            terrible_bench.run_model_health_check = original

    def test_target_uses_larger_token_budget_than_rivals(self) -> None:
        original = terrible_bench.call_openrouter
        calls = []

        def fake_call(api_key, model, prompt, temperature, timeout_seconds, max_tokens=32):
            calls.append(
                {
                    "model": model,
                    "timeout_seconds": timeout_seconds,
                    "max_tokens": max_tokens,
                }
            )
            return "final answer: 0"

        terrible_bench.call_openrouter = fake_call
        try:
            result = run_benchmark(
                {
                    "targetModel": "my/target",
                    "comparisonModels": "tiny/rival",
                    "apiKey": "sk-test",
                    "taskCount": 1,
                    "seed": 43,
                    "demoMode": False,
                    "parallel": False,
                    "includeWeenies": False,
                    "preflightModels": False,
                    "targetMaxTokens": 1500,
                    "comparisonTimeoutSeconds": 15,
                }
            )
        finally:
            terrible_bench.call_openrouter = original

        by_model = {call["model"]: call for call in calls}
        self.assertEqual(by_model["my/target"]["max_tokens"], 1500)
        self.assertEqual(by_model["tiny/rival"]["max_tokens"], terrible_bench.DEFAULT_COMPLETION_TOKENS)
        self.assertEqual(result["runtimeLimits"]["targetMaxTokens"], 1500)

    def test_non_target_deadline_can_skip_trial(self) -> None:
        task = build_task(random.Random(51), 0)
        result = terrible_bench.run_trial(
            model="tiny/rival",
            task=task,
            api_key="sk-test",
            target_model="my/target",
            seed=51,
            demo_mode=False,
            temperature=0,
            timeout_seconds=30,
            target_max_tokens=1200,
            comparison_deadline=time.perf_counter() - 1,
            run_started_at=time.perf_counter(),
        )
        self.assertFalse(result.correct)
        self.assertIn("time limit elapsed", result.error or "")

    def test_p_hack_never_adds_bonus_points(self) -> None:
        result = run_benchmark(
            {
                "targetModel": "my/demo-model",
                "comparisonModels": "tiny/free-model",
                "taskCount": 4,
                "seed": 29,
                "demoMode": True,
                "includeWeenies": False,
                "pHack": True,
            }
        )
        self.assertTrue(result["pHack"]["enabled"])
        self.assertEqual(result["pHack"]["bonus"], 0)
        self.assertTrue(all(row.get("bonus", 0) == 0 for row in result["displayedScores"]))

    def test_weenie_pile_filters_good_models(self) -> None:
        good_model = "openai/gpt-5"
        result = run_benchmark(
            {
                "targetModel": "my/demo-model",
                "comparisonModels": f"{good_model}\ntiny/free-model",
                "taskCount": 1,
                "seed": 31,
                "demoMode": True,
                "includeWeenies": True,
            }
        )
        self.assertNotIn(good_model, result["models"])
        self.assertIn("tiny/free-model", result["models"])

    def test_good_models_are_kept_without_weenie_pile(self) -> None:
        good_model = "openai/gpt-5"
        result = run_benchmark(
            {
                "targetModel": "my/demo-model",
                "comparisonModels": good_model,
                "taskCount": 1,
                "seed": 31,
                "demoMode": True,
                "includeWeenies": False,
            }
        )
        self.assertIn(good_model, result["models"])

    def test_rerun_target_failures_replaces_failed_target_tasks(self) -> None:
        run = {
            "runId": "manual-test-run",
            "benchName": "Manual Test Bench",
            "mode": "demo",
            "seed": 7,
            "targetModel": "my/demo-model",
            "models": ["my/demo-model", "tiny/free-model"],
            "tasks": [
                {
                    "id": "t01",
                    "name": "as in abba",
                    "category": "test",
                    "item": "abba",
                    "operation": "count letter",
                    "target": "a",
                    "prompt": "How many a characters are in abba?\nfinal answer: <integer>",
                    "answer": 2,
                }
            ],
            "results": [
                {
                    "model": "my/demo-model",
                    "task_id": "t01",
                    "output": "final answer: 0",
                    "parsed": 0,
                    "expected": 2,
                    "correct": False,
                    "latency_ms": 10,
                },
                {
                    "model": "tiny/free-model",
                    "task_id": "t01",
                    "output": "final answer: 2",
                    "parsed": 2,
                    "expected": 2,
                    "correct": True,
                    "latency_ms": 10,
                },
            ],
            "pHack": {"enabled": False},
            "scaleHack": False,
        }
        result = rerun_target_failures(
            {
                "run": run,
                "demoMode": True,
                "includeWeenies": False,
            }
        )
        self.assertEqual(result["rerun"]["failedBefore"], 1)
        self.assertEqual(result["rerun"]["rerunAttempts"], 1)
        target_results = [
            item for item in result["results"] if item["model"] == "my/demo-model"
        ]
        self.assertEqual(len(target_results), 1)

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
        path = os.path.join(terrible_bench.BASE_DIR, result["logFile"])
        with open(path, "r", encoding="utf-8") as handle:
            contents = handle.read()
        self.assertNotIn(secret, contents)
        self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
