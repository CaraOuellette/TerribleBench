from __future__ import annotations

import concurrent.futures
import hashlib
import html
import json
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

import requests


APP_TITLE = "Terrible Bench.ai"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_PORT = 8765
DEFAULT_SITE_URL = "https://github.com/CaraOuellette/TerribleBench"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_LOG_DIR = os.path.join(BASE_DIR, "run_logs")
SAFE_LOG_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def load_local_env(filename: str = ".env") -> None:
    path = os.path.join(BASE_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        return


load_local_env()


def load_lines(filename: str) -> list[str]:
    path = os.path.join(BASE_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return [
                line.strip()
                for line in handle
                if line.strip() and not line.lstrip().startswith("#")
            ]
    except OSError:
        return []


FALLBACK_COMPARISON_MODELS = [
    "meta-llama/llama-3.2-3b-instruct:free",
    "google/gemma-3-4b-it:free",
    "qwen/qwen-2.5-7b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
]

DEFAULT_COMPARISON_MODELS = load_lines("model_list.txt") or FALLBACK_COMPARISON_MODELS
EXTRA_FRUITS = load_lines("fruit_list.txt")

WEENIE_MODELS = [
    "openrouter/auto",
    "microsoft/phi-3-mini-128k-instruct:free",
    "huggingfaceh4/zephyr-7b-beta:free",
    "nousresearch/nous-capybara-7b:free",
    "gryphe/mythomist-7b:free",
]

CATEGORIES: dict[str, list[str]] = {
    "fruit, allegedly": [
        "strawberry",
        "straberry",
        "raspberry",
        "blueberry",
        "bananna",
        "pomegranate",
        "dragonfruit",
        "kiwifruit",
        "watermellon",
        "cantaloupe",
        "blackcurrant",
        "grapefruit",
    ],
    "vegetables from memory": [
        "broccoli",
        "brussel sprout",
        "cauliflower",
        "zuchinni",
        "asparagus",
        "artichoke",
        "eggplant",
        "rutabaga",
        "scallion",
        "cucumber",
        "carrott",
        "kale",
    ],
    "american presidents, ominously lowercase": [
        "washington",
        "jefferson",
        "lincoln",
        "roosevelt",
        "eisenhower",
        "kennedy",
        "nixon",
        "carter",
        "reagan",
        "clinton",
        "obama",
        "biden",
    ],
    "ai model names as spellcasting components": [
        "claude sonnet",
        "gpt five point two",
        "gemini flash",
        "llama instruct",
        "mistral tiny",
        "qwen coder",
        "deepseek chat",
        "command-r-plus",
        "oceanic weenie model",
        "terrible bench oracle",
    ],
    "fake public api keys": [
        "sk-terrible-demo-key",
        "tb_live_lunchmistake",
        "or-not-a-real-key-0000",
        "sk-pro-benchmarker-club",
        "tb_public_repo_sacrifice",
        "sk-hackathon-regret",
        "key_for_science_only",
        "definitely-not-credential",
    ],
}

if EXTRA_FRUITS:
    CATEGORIES["fruit_list.txt canon"] = EXTRA_FRUITS

VOWELS = set("aeiou")
ANSWER_RE = re.compile(r"final\s*answer\s*[:=\-]\s*([+-]?\d+)", re.IGNORECASE)
NUMBER_RE = re.compile(r"(?<![\w.])[+-]?\d+(?![\w.])")


@dataclass(frozen=True)
class BenchmarkTask:
    id: str
    name: str
    category: str
    item: str
    operation: str
    target: str
    prompt: str
    answer: int


@dataclass(frozen=True)
class TrialResult:
    model: str
    task_id: str
    output: str
    parsed: int | None
    expected: int
    correct: bool
    latency_ms: int
    start_offset_ms: int = 0
    finished_offset_ms: int = 0
    error: str | None = None


ProgressCallback = Callable[[dict[str, Any]], None]


def stable_int(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def clamp_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, parsed))


def clamp_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, parsed))


def sanitize_request_config(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "targetModel": payload.get("targetModel"),
        "comparisonModels": payload.get("comparisonModels"),
        "taskCount": payload.get("taskCount"),
        "seed": payload.get("seed"),
        "temperature": payload.get("temperature"),
        "timeoutSeconds": payload.get("timeoutSeconds"),
        "demoModeRequested": bool(payload.get("demoMode", True)),
        "parallel": bool(payload.get("parallel", True)),
        "includeWeenies": bool(payload.get("includeWeenies", True)),
        "pHack": bool(payload.get("pHack", False)),
        "scaleHack": bool(payload.get("scaleHack", False)),
        "browserApiKeyProvided": bool((payload.get("apiKey") or "").strip()),
        "envApiKeyAvailable": bool(os.environ.get("OPENROUTER_API_KEY", "").strip()),
    }


def attach_run_log(result: dict[str, Any], payload: dict[str, Any]) -> None:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    suffix = f"{stable_int(str(time.time_ns())) & 0xffff:04x}"
    run_id = f"{stamp}-{result['mode']}-{result['seed']}-{suffix}"
    filename = f"{run_id}.json"
    path = os.path.join(RUN_LOG_DIR, filename)
    relative_path = os.path.relpath(path, BASE_DIR).replace("\\", "/")

    result["runId"] = run_id
    result["logFile"] = relative_path
    result["logUrl"] = f"/api/logs/{run_id}"
    result["latestLogUrl"] = "/api/logs/latest"
    result["logError"] = None

    artifact = {
        "request": sanitize_request_config(payload),
        "run": result,
    }
    try:
        os.makedirs(RUN_LOG_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(artifact, handle, ensure_ascii=False, indent=2)
        with open(os.path.join(RUN_LOG_DIR, "latest.json"), "w", encoding="utf-8") as handle:
            json.dump(artifact, handle, ensure_ascii=False, indent=2)
    except OSError as exc:
        result["logError"] = str(exc)


def load_run_artifact(log_id: str) -> dict[str, Any]:
    if log_id == "latest":
        filename = "latest.json"
    elif SAFE_LOG_ID_RE.fullmatch(log_id):
        filename = f"{log_id}.json"
    else:
        raise ValueError("invalid log id")
    path = os.path.join(RUN_LOG_DIR, filename)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def list_saved_runs() -> list[dict[str, Any]]:
    try:
        filenames = [
            filename
            for filename in os.listdir(RUN_LOG_DIR)
            if filename.endswith(".json") and filename != "latest.json"
        ]
    except OSError:
        return []

    runs: list[dict[str, Any]] = []
    for filename in filenames:
        path = os.path.join(RUN_LOG_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                artifact = json.load(handle)
        except (OSError, ValueError):
            continue
        run = artifact.get("run") or {}
        stat = os.stat(path)
        runs.append(
            {
                "runId": run.get("runId") or filename.removesuffix(".json"),
                "benchName": run.get("benchName") or "Unnamed benchmark",
                "mode": run.get("mode") or "unknown",
                "seed": run.get("seed"),
                "targetModel": run.get("targetModel"),
                "models": len(run.get("models") or []),
                "tasks": len(run.get("tasks") or []),
                "createdAt": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)
                ),
                "mtime": stat.st_mtime,
                "logFile": run.get("logFile") or f"run_logs/{filename}",
                "logUrl": run.get("logUrl") or f"/api/logs/{filename.removesuffix('.json')}",
            }
        )
    runs.sort(key=lambda item: item["mtime"], reverse=True)
    return runs


def task_from_dict(data: dict[str, Any]) -> BenchmarkTask:
    return BenchmarkTask(
        id=str(data["id"]),
        name=str(data["name"]),
        category=str(data["category"]),
        item=str(data["item"]),
        operation=str(data["operation"]),
        target=str(data["target"]),
        prompt=str(data["prompt"]),
        answer=int(data["answer"]),
    )


def trial_from_dict(data: dict[str, Any]) -> TrialResult:
    return TrialResult(
        model=str(data["model"]),
        task_id=str(data["task_id"]),
        output=str(data.get("output") or ""),
        parsed=data.get("parsed"),
        expected=int(data["expected"]),
        correct=bool(data.get("correct")),
        latency_ms=int(data.get("latency_ms") or 0),
        start_offset_ms=int(data.get("start_offset_ms") or 0),
        finished_offset_ms=int(data.get("finished_offset_ms") or data.get("latency_ms") or 0),
        error=data.get("error"),
    )


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        clean = item.strip()
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result


def parse_model_list(text: str) -> list[str]:
    rough = re.split(r"[\n,]+", text or "")
    return dedupe([item.strip() for item in rough if item.strip()])


def letters_only(text: str) -> list[str]:
    return [char.lower() for char in text if char.isalpha()]


def choose_letter(rng: random.Random, item: str, prefer_repeated: bool = True) -> str:
    letters = letters_only(item)
    if not letters:
        return rng.choice("etaoinrslc")
    counts = Counter(letters)
    if prefer_repeated:
        repeated = [letter for letter, count in counts.items() if count > 1]
        if repeated:
            return rng.choice(repeated)
    return rng.choice(letters)


def count_letter(item: str, letter: str) -> int:
    return sum(1 for char in item.lower() if char == letter.lower())


def count_vowels(item: str) -> int:
    return sum(1 for char in item.lower() if char in VOWELS)


def count_consonants(item: str) -> int:
    return sum(1 for char in item.lower() if char.isalpha() and char not in VOWELS)


def prompt_wrap(task_text: str) -> str:
    return (
        "You are taking a completely serious one-shot benchmark.\n"
        f"{task_text}\n"
        "Treat the quoted text literally. Respond with exactly one line:\n"
        "final answer: <integer>"
    )


def make_count_task(rng: random.Random, category: str, item: str, task_id: str) -> BenchmarkTask:
    letter = choose_letter(rng, item)
    answer = count_letter(item, letter)
    text = f"How many times does the letter '{letter}' appear in '{item}'?"
    return BenchmarkTask(
        id=task_id,
        name=f"{letter}s in {item}",
        category=category,
        item=item,
        operation="count letter",
        target=letter,
        prompt=prompt_wrap(text),
        answer=answer,
    )


def make_double_task(rng: random.Random, category: str, item: str, task_id: str) -> BenchmarkTask:
    letter = choose_letter(rng, item)
    multiplier = rng.choice([2, 2, 3])
    answer = count_letter(item, letter) * multiplier
    text = (
        f"Count the letter '{letter}' in '{item}', then multiply that count by {multiplier}."
    )
    return BenchmarkTask(
        id=task_id,
        name=f"{multiplier}x {letter}s in {item}",
        category=category,
        item=item,
        operation="multiply letter count",
        target=letter,
        prompt=prompt_wrap(text),
        answer=answer,
    )


def make_subtract_task(rng: random.Random, category: str, item: str, task_id: str) -> BenchmarkTask:
    first = choose_letter(rng, item)
    second = choose_letter(rng, item, prefer_repeated=False)
    if first == second:
        second = rng.choice([char for char in "etaoinshrdlcumwfgypbvkjxqz" if char != first])
    answer = count_letter(item, first) - count_letter(item, second)
    text = (
        f"In '{item}', subtract the number of '{second}' characters from the number of "
        f"'{first}' characters."
    )
    return BenchmarkTask(
        id=task_id,
        name=f"{first}s minus {second}s in {item}",
        category=category,
        item=item,
        operation="subtract letter counts",
        target=f"{first}-{second}",
        prompt=prompt_wrap(text),
        answer=answer,
    )


def make_inner_count_task(
    rng: random.Random, category: str, item: str, task_id: str
) -> BenchmarkTask:
    letter = choose_letter(rng, item)
    inner = item[1:-1] if len(item) > 2 else ""
    answer = count_letter(inner, letter)
    text = (
        f"Ignoring the first and last character of '{item}', how many '{letter}' "
        "characters remain?"
    )
    return BenchmarkTask(
        id=task_id,
        name=f"inner {letter}s in {item}",
        category=category,
        item=item,
        operation="count after trimming ends",
        target=letter,
        prompt=prompt_wrap(text),
        answer=answer,
    )


def make_vowel_task(rng: random.Random, category: str, item: str, task_id: str) -> BenchmarkTask:
    answer = count_vowels(item)
    text = f"How many vowels are in '{item}'?"
    return BenchmarkTask(
        id=task_id,
        name=f"vowels in {item}",
        category=category,
        item=item,
        operation="count vowels",
        target="vowels",
        prompt=prompt_wrap(text),
        answer=answer,
    )


def make_consonant_task(
    rng: random.Random, category: str, item: str, task_id: str
) -> BenchmarkTask:
    answer = count_consonants(item)
    text = f"How many consonants are in '{item}'?"
    return BenchmarkTask(
        id=task_id,
        name=f"consonants in {item}",
        category=category,
        item=item,
        operation="count consonants",
        target="consonants",
        prompt=prompt_wrap(text),
        answer=answer,
    )


def make_unique_task(rng: random.Random, category: str, item: str, task_id: str) -> BenchmarkTask:
    answer = len(set(letters_only(item)))
    text = f"How many distinct letters appear in '{item}'?"
    return BenchmarkTask(
        id=task_id,
        name=f"unique letters in {item}",
        category=category,
        item=item,
        operation="count unique letters",
        target="unique",
        prompt=prompt_wrap(text),
        answer=answer,
    )


def make_first_index_task(
    rng: random.Random, category: str, item: str, task_id: str
) -> BenchmarkTask:
    letter = choose_letter(rng, item)
    answer = item.lower().find(letter)
    text = (
        f"Using zero-based indexing, what is the index of the first '{letter}' in "
        f"'{item}'? Return -1 if it does not appear."
    )
    return BenchmarkTask(
        id=task_id,
        name=f"first {letter} index in {item}",
        category=category,
        item=item,
        operation="first index",
        target=letter,
        prompt=prompt_wrap(text),
        answer=answer,
    )


TASK_BUILDERS: list[
    Callable[[random.Random, str, str, str], BenchmarkTask]
] = [
    make_count_task,
    make_count_task,
    make_double_task,
    make_subtract_task,
    make_inner_count_task,
    make_vowel_task,
    make_consonant_task,
    make_unique_task,
    make_first_index_task,
]


def build_task(rng: random.Random, index: int) -> BenchmarkTask:
    category = rng.choice(list(CATEGORIES.keys()))
    item = rng.choice(CATEGORIES[category])
    builder = rng.choice(TASK_BUILDERS)
    return builder(rng, category, item, f"t{index + 1:02d}")


def make_benchmark_name(rng: random.Random) -> str:
    prefixes = [
        "Adversarial",
        "Comprehensive",
        "State-of-the-Art",
        "Lunch-Calibrated",
        "Peer-Reviewed",
        "Regex-Native",
        "One-Shot",
    ]
    nouns = [
        "Grapheme Gauntlet",
        "Strawberry Reasoning Suite",
        "LetterMath Arena",
        "Lexical Robustness Trial",
        "Microcount Evaluation",
        "Semantic Produce Harness",
    ]
    return f"{rng.choice(prefixes)} {rng.choice(nouns)} v{rng.randint(0, 9)}.{rng.randint(0, 99)}"


def parse_answer(output: str) -> int | None:
    match = ANSWER_RE.search(output or "")
    if match:
        return int(match.group(1))
    fallback = NUMBER_RE.search(output or "")
    if fallback:
        return int(fallback.group(0))
    return None


def normalize_openrouter_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, dict):
                text = part.get("text") or part.get("content") or ""
                if isinstance(text, str):
                    pieces.append(text)
        return "\n".join(pieces)
    return "" if content is None else str(content)


def call_openrouter(
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    timeout_seconds: int,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", DEFAULT_SITE_URL),
        "X-OpenRouter-Title": APP_TITLE,
    }
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You answer tiny benchmark questions with the requested final answer format.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": 32,
    }
    response = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=body,
        timeout=timeout_seconds,
    )
    if response.status_code >= 400:
        try:
            error_body = response.json()
        except ValueError:
            error_body = response.text
        raise RuntimeError(f"OpenRouter {response.status_code}: {error_body}")
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(str(payload["error"]))
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("OpenRouter returned no choices")
    choice = choices[0]
    message = choice.get("message") or {}
    if "content" in message:
        return normalize_openrouter_content(message.get("content"))
    return normalize_openrouter_content(choice.get("text"))


def synthetic_accuracy(model: str, target_model: str) -> float:
    lower = model.lower()
    accuracy = 0.48
    if model == target_model:
        accuracy += 0.18
    if any(token in lower for token in ["gpt-5", "claude", "opus", "sonnet", "gemini"]):
        accuracy += 0.08
    if any(token in lower for token in ["mistral", "qwen", "llama"]):
        accuracy += 0.02
    if any(token in lower for token in ["free", "mini", "tiny", "7b", "3b", "weenie"]):
        accuracy -= 0.14
    if "openrouter/auto" in lower:
        accuracy -= 0.06
    return max(0.08, min(0.92, accuracy))


def synthetic_output(model: str, task: BenchmarkTask, target_model: str, seed: int) -> str:
    rng = random.Random(stable_int(f"{seed}|{model}|{task.id}|{task.answer}"))
    accuracy = synthetic_accuracy(model, target_model)
    correct = rng.random() < accuracy
    if correct:
        predicted = task.answer
    else:
        predicted = task.answer + rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
        if predicted == task.answer:
            predicted += 1
    styles = [
        "final answer: {answer}",
        "After careful benchmark-grade counting, final answer: {answer}",
        "FINAL ANSWER = {answer}",
        "final answer - {answer}",
    ]
    return rng.choice(styles).format(answer=predicted)


def run_trial(
    model: str,
    task: BenchmarkTask,
    api_key: str,
    target_model: str,
    seed: int,
    demo_mode: bool,
    temperature: float,
    timeout_seconds: int,
    run_started_at: float,
) -> TrialResult:
    start = time.perf_counter()
    start_offset_ms = int((start - run_started_at) * 1000)
    try:
        if demo_mode:
            time.sleep(0.02 + (stable_int(f"{model}|{task.id}") % 30) / 1000)
            output = synthetic_output(model, task, target_model, seed)
        else:
            output = call_openrouter(api_key, model, task.prompt, temperature, timeout_seconds)
        parsed = parse_answer(output)
        finished = time.perf_counter()
        return TrialResult(
            model=model,
            task_id=task.id,
            output=output.strip(),
            parsed=parsed,
            expected=task.answer,
            correct=parsed == task.answer,
            latency_ms=int((finished - start) * 1000),
            start_offset_ms=start_offset_ms,
            finished_offset_ms=int((finished - run_started_at) * 1000),
        )
    except Exception as exc:  # The app should report bad science, not crash during it.
        finished = time.perf_counter()
        return TrialResult(
            model=model,
            task_id=task.id,
            output="",
            parsed=None,
            expected=task.answer,
            correct=False,
            latency_ms=int((finished - start) * 1000),
            start_offset_ms=start_offset_ms,
            finished_offset_ms=int((finished - run_started_at) * 1000),
            error=str(exc)[:500],
        )


def execute_trials(
    models: list[str],
    tasks: list[BenchmarkTask],
    api_key: str,
    target_model: str,
    seed: int,
    demo_mode: bool,
    parallel: bool,
    temperature: float,
    timeout_seconds: int,
    progress_callback: ProgressCallback | None = None,
) -> list[TrialResult]:
    jobs = [(model, task) for model in models for task in tasks]
    run_started_at = time.perf_counter()
    results: list[TrialResult] = []

    def record_result(result: TrialResult) -> None:
        results.append(result)
        if progress_callback:
            progress_callback(
                {
                    "type": "trial_complete",
                    "completed": len(results),
                    "total": len(jobs),
                    "result": asdict(result),
                }
            )

    if not parallel:
        for model, task in jobs:
            record_result(
                run_trial(
                    model,
                    task,
                    api_key,
                    target_model,
                    seed,
                    demo_mode,
                    temperature,
                    timeout_seconds,
                    run_started_at,
                )
            )
        return results

    max_workers = min(24, max(1, len(jobs)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                run_trial,
                model,
                task,
                api_key,
                target_model,
                seed,
                demo_mode,
                temperature,
                timeout_seconds,
                run_started_at,
            )
            for model, task in jobs
        ]
        for future in concurrent.futures.as_completed(futures):
            record_result(future.result())
    return results


def summarize_model_timing(
    results: list[TrialResult],
    models: list[str],
    task_count: int,
) -> list[dict[str, Any]]:
    by_model: dict[str, list[TrialResult]] = {model: [] for model in models}
    for result in results:
        by_model.setdefault(result.model, []).append(result)

    timings: list[dict[str, Any]] = []
    for model in models:
        model_results = by_model.get(model, [])
        starts = [result.start_offset_ms for result in model_results]
        finishes = [result.finished_offset_ms for result in model_results]
        sum_latency = sum(result.latency_ms for result in model_results)
        completed = len(model_results)
        timings.append(
            {
                "model": model,
                "completed": completed,
                "total": task_count,
                "correct": sum(1 for result in model_results if result.correct),
                "errors": sum(1 for result in model_results if result.error),
                "batch_wall_ms": max(finishes) - min(starts) if starts and finishes else 0,
                "sum_latency_ms": sum_latency,
                "average_latency_ms": int(sum_latency / max(1, completed)),
                "first_started_offset_ms": min(starts) if starts else None,
                "last_finished_offset_ms": max(finishes) if finishes else None,
                "slowest_task_ms": max((result.latency_ms for result in model_results), default=0),
            }
        )
    return timings


def score_models(
    results: list[TrialResult],
    models: list[str],
    task_ids: list[str],
    bonus_by_model: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    bonus_by_model = bonus_by_model or {}
    selected = set(task_ids)
    total = max(1, len(task_ids))
    by_model: dict[str, list[TrialResult]] = {model: [] for model in models}
    for result in results:
        if result.task_id in selected and result.model in by_model:
            by_model[result.model].append(result)

    rows: list[dict[str, Any]] = []
    for model in models:
        model_results = by_model.get(model, [])
        raw_correct = sum(1 for result in model_results if result.correct)
        bonus = bonus_by_model.get(model, 0)
        correct = raw_correct + bonus
        rows.append(
            {
                "model": model,
                "correct": correct,
                "raw_correct": raw_correct,
                "bonus": bonus,
                "total": total,
                "percent": round((correct / total) * 100, 1),
                "errors": sum(1 for result in model_results if result.error),
                "average_latency_ms": int(
                    sum(result.latency_ms for result in model_results) / max(1, len(model_results))
                ),
            }
        )
    rows.sort(key=lambda row: (row["percent"], row["correct"]), reverse=True)
    return rows


def choose_p_hacked_subset(
    results: list[TrialResult],
    models: list[str],
    target_model: str,
    task_ids: list[str],
) -> dict[str, Any]:
    if target_model not in models or not task_ids:
        return {
            "selected_task_ids": task_ids,
            "bonus": 0,
            "note": "P-hack failed because the target model was not in the table.",
        }

    result_lookup = {(result.model, result.task_id): result for result in results}
    competitors = [model for model in models if model != target_model]

    def correct(model: str, task_id: str) -> int:
        result = result_lookup.get((model, task_id))
        return int(bool(result and result.correct))

    def scores_for(ids: list[str]) -> dict[str, int]:
        return {model: sum(correct(model, task_id) for task_id in ids) for model in models}

    task_strength: list[tuple[float, str]] = []
    for task_id in task_ids:
        target_hit = correct(target_model, task_id)
        competitor_average = (
            sum(correct(model, task_id) for model in competitors) / max(1, len(competitors))
        )
        task_strength.append((target_hit - competitor_average, task_id))
    ordered_ids = [task_id for _, task_id in sorted(task_strength, reverse=True)]

    best_ids = task_ids
    best_key: tuple[float, float, float, int] | None = None
    for size in range(1, len(ordered_ids) + 1):
        candidate = ordered_ids[:size]
        scores = scores_for(candidate)
        target_score = scores[target_model]
        target_percent = target_score / size
        competitor_best = max(
            (score / size for model, score in scores.items() if model != target_model),
            default=0,
        )
        beat_count = sum(
            1
            for model, score in scores.items()
            if model != target_model and target_percent >= (score / size)
        )
        key = (beat_count, target_percent, target_percent - competitor_best, -size)
        if best_key is None or key > best_key:
            best_key = key
            best_ids = candidate

    selected_set = set(best_ids)
    selected = [task_id for task_id in task_ids if task_id in selected_set]
    rejected = len(task_ids) - len(selected)
    return {
        "selected_task_ids": selected,
        "bonus": 0,
        "note": f"Cherry-picked {len(selected)} observed tasks and rejected {rejected}; no invented normalization applied.",
    }


def chart_bounds(scores: list[dict[str, Any]], scale_hack: bool) -> dict[str, float]:
    percents = [row["percent"] for row in scores] or [0]
    if not scale_hack:
        return {"minimum": 0, "maximum": max(100, max(percents))}
    low = max(0, min(percents) - 2)
    high = max(low + 1, max(percents) + 1)
    return {"minimum": round(low, 1), "maximum": round(high, 1)}


def make_summary(
    target_model: str,
    raw_scores: list[dict[str, Any]],
    displayed_scores: list[dict[str, Any]],
    raw_task_count: int,
    displayed_task_count: int,
) -> dict[str, Any]:
    raw_competitors = max(0, len(raw_scores) - 1)
    target_display = next(
        (row for row in displayed_scores if row["model"] == target_model),
        {"percent": 0, "correct": 0},
    )
    beat_count = sum(
        1
        for row in displayed_scores
        if row["model"] != target_model and target_display["percent"] >= row["percent"]
    )
    return {
        "target": target_model,
        "raw_competitors": raw_competitors,
        "displayed_competitors": beat_count,
        "raw_tasks": raw_task_count,
        "displayed_tasks": displayed_task_count,
    }


def build_report(
    *,
    bench_name: str,
    mode: str,
    seed: int,
    target_model: str,
    models: list[str],
    tasks: list[BenchmarkTask],
    results: list[TrialResult],
    p_hack_enabled: bool,
    scale_hack: bool,
    payload: dict[str, Any],
    rerun: dict[str, Any] | None = None,
) -> dict[str, Any]:
    all_task_ids = [task.id for task in tasks]
    p_hack = {
        "enabled": p_hack_enabled,
        "selected_task_ids": all_task_ids,
        "bonus": 0,
        "note": "No p-hacking applied. Methodology briefly considered integrity.",
    }
    if p_hack_enabled:
        p_hack.update(choose_p_hacked_subset(results, models, target_model, all_task_ids))

    selected_task_ids = p_hack["selected_task_ids"]
    raw_scores = score_models(results, models, all_task_ids)
    displayed_scores = score_models(results, models, selected_task_ids)

    report = {
        "app": APP_TITLE,
        "benchName": bench_name,
        "mode": mode,
        "seed": seed,
        "targetModel": target_model,
        "models": models,
        "tasks": [asdict(task) for task in tasks],
        "results": [asdict(result) for result in results],
        "modelTiming": summarize_model_timing(results, models, len(tasks)),
        "rawScores": raw_scores,
        "displayedScores": displayed_scores,
        "pHack": {
            **p_hack,
            "rejected": len(all_task_ids) - len(selected_task_ids),
        },
        "chart": chart_bounds(displayed_scores, scale_hack),
        "scaleHack": scale_hack,
        "summary": make_summary(
            target_model,
            raw_scores,
            displayed_scores,
            len(all_task_ids),
            len(selected_task_ids),
        ),
    }
    if rerun:
        report["rerun"] = rerun
    attach_run_log(report, payload)
    return report


def run_benchmark(
    payload: dict[str, Any],
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    target_model = (payload.get("targetModel") or "openai/gpt-5.2").strip()
    comparison_models = parse_model_list(payload.get("comparisonModels") or "")
    if payload.get("includeWeenies", True):
        comparison_models.extend(WEENIE_MODELS)
    comparison_models = comparison_models or DEFAULT_COMPARISON_MODELS
    models = dedupe([target_model] + comparison_models)

    seed_value = payload.get("seed")
    seed = (
        clamp_int(seed_value, 0, 0, 2_147_483_647)
        if str(seed_value or "").strip()
        else int(time.time() * 1000) % 2_147_483_647
    )
    rng = random.Random(seed)
    task_count = clamp_int(payload.get("taskCount"), 8, 1, 48)
    timeout_seconds = clamp_int(payload.get("timeoutSeconds"), 30, 5, 120)
    temperature = clamp_float(payload.get("temperature"), 0.0, 0.0, 2.0)

    browser_key = (payload.get("apiKey") or "").strip()
    api_key = browser_key or os.environ.get("OPENROUTER_API_KEY", "").strip()
    demo_mode = bool(payload.get("demoMode", True)) or not api_key
    mode = "demo" if demo_mode else "openrouter"
    parallel = bool(payload.get("parallel", True))

    tasks = [build_task(rng, index) for index in range(task_count)]
    bench_name = make_benchmark_name(rng)
    if progress_callback:
        progress_callback(
            {
                "type": "setup",
                "benchName": bench_name,
                "mode": mode,
                "seed": seed,
                "models": models,
                "tasks": [asdict(task) for task in tasks],
                "total": len(models) * len(tasks),
            }
        )
    results = execute_trials(
        models=models,
        tasks=tasks,
        api_key=api_key,
        target_model=target_model,
        seed=seed,
        demo_mode=demo_mode,
        parallel=parallel,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        progress_callback=progress_callback,
    )

    p_hack_enabled = bool(payload.get("pHack", False))
    return build_report(
        bench_name=bench_name,
        mode=mode,
        seed=seed,
        target_model=target_model,
        models=models,
        tasks=tasks,
        results=results,
        p_hack_enabled=p_hack_enabled,
        scale_hack=bool(payload.get("scaleHack", False)),
        payload=payload,
    )


def rerun_target_failures(
    payload: dict[str, Any],
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    run_id = str(payload.get("runId") or "").strip()
    if run_id:
        artifact = load_run_artifact(run_id)
        run = artifact.get("run") or {}
    else:
        run = payload.get("run") or {}
    if not run:
        raise RuntimeError("No saved run was provided for failed-task rerun.")

    target_model = str(run.get("targetModel") or "")
    if not target_model:
        raise RuntimeError("Saved run has no target model.")

    models = [str(model) for model in (run.get("models") or [])]
    if target_model not in models:
        models.insert(0, target_model)
    tasks = [task_from_dict(task) for task in (run.get("tasks") or [])]
    task_by_id = {task.id: task for task in tasks}
    existing_results = [trial_from_dict(result) for result in (run.get("results") or [])]
    result_by_key: dict[tuple[str, str], TrialResult] = {
        (result.model, result.task_id): result for result in existing_results
    }

    failed_task_ids = [
        task.id
        for task in tasks
        if (target_model, task.id) in result_by_key
        and not result_by_key[(target_model, task.id)].correct
    ]
    failed_tasks = [task_by_id[task_id] for task_id in failed_task_ids]

    browser_key = (payload.get("apiKey") or "").strip()
    api_key = browser_key or os.environ.get("OPENROUTER_API_KEY", "").strip()
    original_mode = str(run.get("mode") or "demo")
    requested_demo = bool(payload.get("demoMode", original_mode == "demo"))
    if original_mode == "openrouter" and not requested_demo and not api_key:
        raise RuntimeError("An OpenRouter API key is required to rerun real failed tasks.")
    demo_mode = requested_demo or not api_key
    mode = "demo" if demo_mode else "openrouter"
    seed = clamp_int(run.get("seed"), int(time.time() * 1000) % 2_147_483_647, 0, 2_147_483_647)
    retry_seed = int(time.time() * 1000) % 2_147_483_647
    temperature = clamp_float(payload.get("temperature"), 0.0, 0.0, 2.0)
    timeout_seconds = clamp_int(payload.get("timeoutSeconds"), 30, 5, 120)
    parallel = bool(payload.get("parallel", True))

    retry_results: list[TrialResult] = []
    if progress_callback:
        progress_callback(
            {
                "type": "setup",
                "benchName": f"Retry failures for {run.get('benchName') or 'saved run'}",
                "mode": mode,
                "seed": retry_seed,
                "models": [target_model],
                "tasks": [asdict(task) for task in failed_tasks],
                "total": len(failed_tasks),
            }
        )
    if failed_tasks:
        retry_results = execute_trials(
            models=[target_model],
            tasks=failed_tasks,
            api_key=api_key,
            target_model=target_model,
            seed=retry_seed,
            demo_mode=demo_mode,
            parallel=parallel,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            progress_callback=progress_callback,
        )
        for result in retry_results:
            result_by_key[(result.model, result.task_id)] = result

    ordered_results: list[TrialResult] = []
    for model in models:
        for task in tasks:
            result = result_by_key.get((model, task.id))
            if result:
                ordered_results.append(result)

    recovered = sum(1 for result in retry_results if result.correct)
    still_failed = sum(1 for result in retry_results if not result.correct)
    p_hack_enabled = bool(payload.get("pHack", (run.get("pHack") or {}).get("enabled", False)))
    scale_hack = bool(payload.get("scaleHack", run.get("scaleHack", False)))
    rerun = {
        "sourceRunId": run.get("runId") or run_id or None,
        "targetModel": target_model,
        "failedBefore": len(failed_tasks),
        "rerunAttempts": len(retry_results),
        "recovered": recovered,
        "stillFailed": still_failed,
        "mode": mode,
        "retrySeed": retry_seed,
        "replacedResults": [asdict(result) for result in retry_results],
        "note": (
            "No failed target tasks were available to rerun."
            if not failed_tasks
            else f"Retried {len(retry_results)} failed target tasks and recovered {recovered} with real retry outputs."
        ),
    }
    rerun_payload = {
        **payload,
        "apiKey": "",
        "sourceRunId": run.get("runId") or run_id,
        "targetModel": target_model,
        "taskCount": len(tasks),
        "demoMode": demo_mode,
        "pHack": p_hack_enabled,
        "scaleHack": scale_hack,
        "rerunFailures": True,
    }
    return build_report(
        bench_name=f"{run.get('benchName') or 'Saved Run'} + Target Retry",
        mode=mode,
        seed=seed,
        target_model=target_model,
        models=models,
        tasks=tasks,
        results=ordered_results,
        p_hack_enabled=p_hack_enabled,
        scale_hack=scale_hack,
        payload=rerun_payload,
        rerun=rerun,
    )


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Terrible Bench.ai</title>
  <style>
    :root {
      --paper: #fbfaf6;
      --ink: #1f2328;
      --muted: #6b6f76;
      --line: #d8d2c3;
      --panel: #ffffff;
      --red: #b3261e;
      --green: #216e4e;
      --gold: #aa6b00;
      --blue: #285f9f;
      --shadow: 0 18px 45px rgba(31, 35, 40, 0.08);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--paper);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }

    button, input, textarea {
      font: inherit;
    }

    .app {
      width: min(1460px, calc(100vw - 28px));
      margin: 0 auto;
      padding: 18px 0 28px;
    }

    .topbar {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      padding: 4px 2px 14px;
      border-bottom: 1px solid var(--line);
    }

    h1 {
      margin: 0;
      font-size: 28px;
      line-height: 1.05;
      letter-spacing: 0;
    }

    .subtitle {
      color: var(--muted);
      margin-top: 5px;
      font-size: 14px;
    }

    .status-pill {
      min-height: 34px;
      padding: 7px 11px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fffdf8;
      color: var(--muted);
      white-space: nowrap;
      font-size: 13px;
    }

    .layout {
      display: grid;
      grid-template-columns: minmax(280px, 360px) minmax(0, 1fr);
      gap: 16px;
      margin-top: 16px;
      align-items: start;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }

    .controls {
      padding: 14px;
      position: sticky;
      top: 12px;
    }

    .control-group {
      padding: 11px 0;
      border-bottom: 1px solid #ece6d9;
    }

    .control-group:first-child { padding-top: 0; }
    .control-group:last-child { border-bottom: 0; padding-bottom: 0; }

    label {
      display: block;
      margin-bottom: 6px;
      color: #353940;
      font-size: 13px;
      font-weight: 700;
    }

    input[type="text"],
    input[type="password"],
    input[type="number"],
    select,
    textarea {
      width: 100%;
      min-height: 38px;
      border: 1px solid #cfc7b8;
      border-radius: 6px;
      padding: 9px 10px;
      color: var(--ink);
      background: #fffefa;
      outline: none;
    }

    input:focus, textarea:focus {
      border-color: var(--blue);
      box-shadow: 0 0 0 3px rgba(40, 95, 159, 0.12);
    }

    textarea {
      min-height: 118px;
      resize: vertical;
      line-height: 1.35;
      font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
      font-size: 12px;
    }

    .split {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 9px;
    }

    .check {
      display: grid;
      grid-template-columns: 19px minmax(0, 1fr);
      gap: 8px;
      align-items: start;
      margin: 8px 0;
      color: #343942;
      font-weight: 650;
      font-size: 13px;
    }

    .check input {
      width: 17px;
      height: 17px;
      margin: 1px 0 0;
      accent-color: var(--red);
    }

    .hint {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
      margin-top: 6px;
    }

    .primary {
      width: 100%;
      min-height: 48px;
      border: 0;
      border-radius: 6px;
      background: var(--red);
      color: white;
      font-weight: 850;
      cursor: pointer;
      margin-top: 12px;
      box-shadow: 0 10px 18px rgba(179, 38, 30, 0.22);
    }

    .primary:disabled {
      opacity: 0.68;
      cursor: wait;
    }

    .button-row {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-top: 8px;
    }

    .secondary {
      min-height: 38px;
      border: 1px solid #cfc7b8;
      border-radius: 6px;
      background: #fffefa;
      color: #353940;
      font-weight: 800;
      cursor: pointer;
    }

    .secondary:disabled {
      opacity: 0.58;
      cursor: wait;
    }

    .results {
      min-height: 720px;
      overflow: hidden;
    }

    .empty {
      min-height: 720px;
      display: grid;
      place-items: center;
      text-align: center;
      color: var(--muted);
      padding: 26px;
    }

    .empty strong {
      display: block;
      color: var(--ink);
      font-size: 19px;
      margin-bottom: 6px;
    }

    .loading {
      min-height: 720px;
      display: grid;
      place-items: center;
      gap: 14px;
      padding: 30px;
      text-align: center;
    }

    .loading-card {
      width: min(760px, 100%);
      display: grid;
      justify-items: center;
      gap: 14px;
    }

    .throbber {
      width: 86px;
      height: 86px;
      border: 12px solid #eee3d0;
      border-top-color: var(--red);
      border-right-color: var(--gold);
      border-radius: 50%;
      animation: rotate 0.78s linear infinite;
    }

    @keyframes rotate { to { transform: rotate(360deg); } }

    .progress-content {
      width: 100%;
      text-align: left;
      border: 1px solid #e1dacd;
      border-radius: 8px;
      background: #fffefa;
      padding: 12px;
    }

    .progress-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      color: #353940;
      font-size: 13px;
      font-weight: 800;
      margin-bottom: 10px;
    }

    .progress-eta {
      color: var(--muted);
      font-weight: 650;
      white-space: nowrap;
    }

    .progress-row {
      display: grid;
      grid-template-columns: minmax(150px, 260px) minmax(0, 1fr) 115px;
      gap: 10px;
      align-items: center;
      min-height: 34px;
      border-top: 1px solid #eee8dd;
      padding: 7px 0;
    }

    .progress-track {
      height: 17px;
      border-radius: 5px;
      border: 1px solid #d8d0bf;
      background: #eee9dd;
      overflow: hidden;
    }

    .progress-fill {
      height: 100%;
      min-width: 2px;
      background: var(--gold);
    }

    .progress-fill.done {
      background: var(--green);
    }

    .progress-meta {
      color: var(--muted);
      font-size: 12px;
      font-variant-numeric: tabular-nums;
      text-align: right;
    }

    .progress-latest {
      color: var(--muted);
      font-size: 12px;
      margin-top: 3px;
      min-height: 16px;
    }

    .result-body {
      padding: 16px;
    }

    .bench-title {
      display: flex;
      justify-content: space-between;
      align-items: start;
      gap: 12px;
      padding-bottom: 12px;
      border-bottom: 1px solid #ece6d9;
    }

    .bench-title h2 {
      margin: 0;
      font-size: 22px;
      line-height: 1.2;
      letter-spacing: 0;
    }

    .mode {
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
      padding-top: 4px;
    }

    .summary {
      margin: 14px 0 16px;
      padding: 12px;
      border-left: 5px solid var(--gold);
      background: #fff8e8;
      border-radius: 6px;
      line-height: 1.45;
      font-size: 15px;
    }

    .scratch {
      color: var(--muted);
      text-decoration: line-through;
      text-decoration-thickness: 2px;
    }

    .chart {
      display: grid;
      gap: 9px;
      margin: 12px 0 16px;
    }

    .bar-row {
      display: grid;
      grid-template-columns: minmax(130px, 255px) minmax(0, 1fr) 80px;
      gap: 10px;
      align-items: center;
      min-height: 38px;
    }

    .model-name {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 13px;
      font-weight: 750;
    }

    .bar-track {
      height: 24px;
      border-radius: 5px;
      background: #eee9dd;
      overflow: hidden;
      border: 1px solid #d8d0bf;
    }

    .bar {
      height: 100%;
      min-width: 2px;
      background: var(--blue);
    }

    .bar.target {
      background: var(--green);
    }

    .score {
      text-align: right;
      font-variant-numeric: tabular-nums;
      font-weight: 800;
      font-size: 13px;
    }

    .axis-note {
      color: var(--muted);
      font-size: 12px;
      margin-top: -5px;
      min-height: 18px;
    }

    .mini-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin: 15px 0;
    }

    .metric {
      border: 1px solid #e1dacd;
      border-radius: 8px;
      padding: 10px;
      min-height: 72px;
      background: #fffefa;
    }

    .metric b {
      display: block;
      font-size: 21px;
      line-height: 1;
      margin-bottom: 7px;
    }

    .metric span {
      color: var(--muted);
      font-size: 12px;
    }

    .audit {
      margin: 12px 0;
      padding: 10px 12px;
      border: 1px solid #d8d0bf;
      background: #fffefa;
      border-radius: 6px;
      color: #353940;
      font-size: 13px;
      line-height: 1.45;
    }

    .audit code {
      font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
      font-size: 12px;
    }

    .audit a {
      color: var(--blue);
      font-weight: 800;
    }

    details {
      border-top: 1px solid #ece6d9;
      padding-top: 11px;
      margin-top: 11px;
    }

    summary {
      cursor: pointer;
      font-weight: 800;
      color: #2e333a;
      min-height: 30px;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      margin-top: 8px;
    }

    th, td {
      border-bottom: 1px solid #eee8dd;
      padding: 8px 7px;
      text-align: left;
      vertical-align: top;
    }

    th {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
    }

    .prompt {
      max-width: 460px;
      white-space: normal;
    }

    .ok { color: var(--green); font-weight: 800; }
    .bad { color: var(--red); font-weight: 800; }
    .warn { color: var(--gold); font-weight: 800; }

    .error {
      margin: 16px;
      padding: 12px;
      border-radius: 6px;
      background: #fff0ed;
      border: 1px solid #f0b8ae;
      color: #7f1d16;
      white-space: pre-wrap;
    }

    @media (max-width: 900px) {
      .layout {
        grid-template-columns: 1fr;
      }

      .controls {
        position: static;
      }

      .bench-title,
      .topbar {
        align-items: start;
        flex-direction: column;
      }

      .mode,
      .status-pill {
        white-space: normal;
      }

      .bar-row {
        grid-template-columns: 1fr;
        gap: 5px;
        padding: 8px 0;
        border-bottom: 1px solid #eee8dd;
      }

      .progress-row {
        grid-template-columns: 1fr;
        gap: 5px;
      }

      .progress-meta {
        text-align: left;
      }

      .score {
        text-align: left;
      }

      .mini-grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <main class="app">
    <header class="topbar">
      <div>
        <h1>Terrible Bench.ai</h1>
        <div class="subtitle">One-click benchmark generation for people who have already chosen the winner.</div>
      </div>
      <div id="status" class="status-pill">local peer review board: idle</div>
    </header>

    <section class="layout">
      <form id="benchForm" class="panel controls">
        <div class="control-group">
          <label for="targetModel">Your Model</label>
          <input id="targetModel" type="text" value="openai/gpt-5.2" autocomplete="off">
        </div>

        <div class="control-group">
          <label for="apiKey">OpenRouter API Key</label>
          <input id="apiKey" type="password" placeholder="OPENROUTER_API_KEY">
          <div class="hint" id="keyHint">__KEY_HINT__</div>
        </div>

        <div class="control-group">
          <label for="comparisonModels">Comparison Models</label>
          <textarea id="comparisonModels">__DEFAULT_MODEL_TEXT__</textarea>
        </div>

        <div class="control-group split">
          <div>
            <label for="taskCount">Benchmarks</label>
            <input id="taskCount" type="number" min="1" max="48" value="8">
          </div>
          <div>
            <label for="seed">Seed</label>
            <input id="seed" type="number" placeholder="random">
          </div>
        </div>

        <div class="control-group split">
          <div>
            <label for="temperature">Temp</label>
            <input id="temperature" type="number" min="0" max="2" step="0.1" value="0">
          </div>
          <div>
            <label for="timeoutSeconds">Timeout</label>
            <input id="timeoutSeconds" type="number" min="5" max="120" value="30">
          </div>
        </div>

        <div class="control-group">
          <label class="check"><input id="demoMode" type="checkbox" __DEMO_MODE_CHECKED__><span>Synthetic demo mode</span></label>
          <label class="check"><input id="parallel" type="checkbox" checked><span>Parallel calls</span></label>
          <label class="check"><input id="includeWeenies" type="checkbox" checked><span>Weenie model pile-on</span></label>
          <label class="check"><input id="pHack" type="checkbox" checked><span>Cherry-pick best subset</span></label>
          <label class="check"><input id="scaleHack" type="checkbox" checked><span>Suspicious chart axis</span></label>
          <label class="check"><input id="shareKey" type="checkbox"><span>Join public API key leaderboard</span></label>
        </div>

        <button id="runButton" class="primary" type="submit">Generate Benchmark</button>

        <div class="control-group">
          <label for="savedRuns">Saved Runs</label>
          <select id="savedRuns">
            <option value="">No saved runs loaded</option>
          </select>
          <div class="button-row">
            <button id="loadRunButton" class="secondary" type="button">Load Selected</button>
            <button id="loadLatestButton" class="secondary" type="button">Load Latest</button>
          </div>
          <button id="refreshRunsButton" class="secondary" type="button" style="width:100%; margin-top:8px;">Refresh Runs</button>
        </div>
      </form>

      <section id="results" class="panel results">
        <div class="empty">
          <div>
            <strong>No benchmark yet.</strong>
            <span>Awaiting methodology.</span>
          </div>
        </div>
      </section>
    </section>
  </main>

  <script>
    const form = document.getElementById("benchForm");
    const results = document.getElementById("results");
    const statusPill = document.getElementById("status");
    const runButton = document.getElementById("runButton");
    const apiKeyInput = document.getElementById("apiKey");
    const demoModeInput = document.getElementById("demoMode");
    const keyHint = document.getElementById("keyHint");
    const savedRunsSelect = document.getElementById("savedRuns");
    const refreshRunsButton = document.getElementById("refreshRunsButton");
    const loadRunButton = document.getElementById("loadRunButton");
    const loadLatestButton = document.getElementById("loadLatestButton");
    let progressState = null;
    let currentRun = null;

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function value(id) {
      return document.getElementById(id).value;
    }

    function checked(id) {
      return document.getElementById(id).checked;
    }

    function formatDuration(ms) {
      const value = Math.max(0, Number(ms) || 0);
      if (value < 1000) {
        return `${Math.round(value)}ms`;
      }
      const seconds = value / 1000;
      if (seconds < 60) {
        return `${seconds.toFixed(seconds < 10 ? 1 : 0)}s`;
      }
      const minutes = Math.floor(seconds / 60);
      const remainder = Math.round(seconds % 60);
      return `${minutes}m ${remainder}s`;
    }

    function percent(part, total) {
      if (!total) return 0;
      return Math.max(0, Math.min(100, (Number(part) / Number(total)) * 100));
    }

    function payloadFromForm() {
      return {
        targetModel: value("targetModel"),
        apiKey: value("apiKey"),
        comparisonModels: value("comparisonModels"),
        taskCount: Number(value("taskCount")),
        seed: value("seed"),
        temperature: Number(value("temperature")),
        timeoutSeconds: Number(value("timeoutSeconds")),
        demoMode: checked("demoMode"),
        parallel: checked("parallel"),
        includeWeenies: checked("includeWeenies"),
        pHack: checked("pHack"),
        scaleHack: checked("scaleHack"),
        shareKey: checked("shareKey")
      };
    }

    function showLoading() {
      progressState = null;
      results.innerHTML = `
        <div class="loading">
          <div class="loading-card">
            <div class="throbber" aria-hidden="true"></div>
            <div>
              <strong>Generating regrettable science...</strong><br>
              <span>contacting the standards committee</span>
            </div>
            <div id="progressContent" class="progress-content">
              <div class="progress-head">
                <span>Preparing prompts</span>
                <span class="progress-eta">warming up</span>
              </div>
              <div class="progress-latest">Waiting for the run manifest.</div>
            </div>
          </div>
        </div>`;
    }

    function initializeProgress(data) {
      progressState = {
        startedAt: performance.now(),
        benchName: data.benchName,
        mode: data.mode,
        models: data.models,
        taskCount: data.tasks.length,
        total: data.total,
        completed: 0,
        byModel: Object.fromEntries(data.models.map(model => [model, {
          model,
          completed: 0,
          correct: 0,
          errors: 0,
          sumLatencyMs: 0,
          firstStartMs: null,
          lastFinishMs: null,
          batchWallMs: 0,
          latest: "queued"
        }]))
      };
      renderProgress();
    }

    function updateProgress(data) {
      if (!progressState || !data.result) return;
      const result = data.result;
      const model = progressState.byModel[result.model];
      if (!model) return;
      progressState.completed = data.completed;
      progressState.total = data.total;
      model.completed += 1;
      model.correct += result.correct ? 1 : 0;
      model.errors += result.error ? 1 : 0;
      model.sumLatencyMs += Number(result.latency_ms) || 0;
      const start = Number(result.start_offset_ms) || 0;
      const finish = Number(result.finished_offset_ms) || 0;
      model.firstStartMs = model.firstStartMs === null ? start : Math.min(model.firstStartMs, start);
      model.lastFinishMs = model.lastFinishMs === null ? finish : Math.max(model.lastFinishMs, finish);
      model.batchWallMs = Math.max(0, model.lastFinishMs - model.firstStartMs);
      const status = result.error ? "error" : (result.correct ? "pass" : "fail");
      model.latest = `${result.task_id} ${status} in ${formatDuration(result.latency_ms)}`;
      renderProgress();
    }

    function renderProgress() {
      const node = document.getElementById("progressContent");
      if (!node || !progressState) return;
      const elapsed = performance.now() - progressState.startedAt;
      const done = progressState.completed >= progressState.total;
      const etaMs = progressState.completed > 0 && !done
        ? ((progressState.total - progressState.completed) / (progressState.completed / Math.max(1, elapsed)))
        : null;
      const rows = progressState.models.map(modelName => {
        const model = progressState.byModel[modelName];
        const width = percent(model.completed, progressState.taskCount);
        const complete = model.completed >= progressState.taskCount;
        const timing = complete
          ? `${formatDuration(model.batchWallMs)} batch`
          : (model.completed ? `${formatDuration(model.batchWallMs)} so far` : "queued");
        return `
          <div class="progress-row">
            <div>
              <div class="model-name" title="${escapeHtml(modelName)}">${escapeHtml(modelName)}</div>
              <div class="progress-latest">${escapeHtml(model.latest)}</div>
            </div>
            <div class="progress-track">
              <div class="progress-fill ${complete ? "done" : ""}" style="width:${width}%"></div>
            </div>
            <div class="progress-meta">${model.completed}/${progressState.taskCount} - ${escapeHtml(timing)}</div>
          </div>`;
      }).join("");
      const eta = done
        ? "finishing report"
        : (etaMs === null ? "estimating ETA" : `ETA ${formatDuration(etaMs)}`);
      node.innerHTML = `
        <div class="progress-head">
          <span>${escapeHtml(progressState.benchName)} - ${progressState.completed}/${progressState.total} calls</span>
          <span class="progress-eta">${escapeHtml(eta)}</span>
        </div>
        <div class="progress-track">
          <div class="progress-fill ${done ? "done" : ""}" style="width:${percent(progressState.completed, progressState.total)}%"></div>
        </div>
        ${rows}`;
    }

    function showError(error) {
      results.innerHTML = `<div class="error">${escapeHtml(error)}</div>`;
    }

    async function refreshSavedRuns() {
      try {
        const response = await fetch("/api/logs");
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Could not load saved runs.");
        }
        const runs = data.runs || [];
        if (!runs.length) {
          savedRunsSelect.innerHTML = `<option value="">No saved runs yet</option>`;
          return;
        }
        savedRunsSelect.innerHTML = runs.map(run => {
          const label = `${run.createdAt} - ${run.mode} - ${run.benchName} (${run.models}x${run.tasks})`;
          return `<option value="${escapeHtml(run.runId)}">${escapeHtml(label)}</option>`;
        }).join("");
      } catch (error) {
        savedRunsSelect.innerHTML = `<option value="">Could not load runs</option>`;
      }
    }

    async function loadRun(logId) {
      if (!logId) return;
      const response = await fetch(`/api/logs/${encodeURIComponent(logId)}`);
      const artifact = await response.json();
      if (!response.ok) {
        throw new Error(artifact.error || "Could not load saved run.");
      }
      renderResults(artifact.run);
      statusPill.textContent = `local peer review board: loaded ${artifact.run.mode}`;
    }

    function scoreWidth(score, chart) {
      const min = Number(chart.minimum);
      const max = Number(chart.maximum);
      if (max <= min) return 100;
      return Math.max(0, Math.min(100, ((score.percent - min) / (max - min)) * 100));
    }

    function renderSummary(data) {
      const summary = data.summary;
      return `
        <div class="summary">
          In our rigorous testing, <b>${escapeHtml(summary.target)}</b> beat
          <span class="scratch">${summary.raw_competitors}</span>
          <b>${summary.displayed_competitors}</b> current top models across
          <span class="scratch">${summary.raw_tasks}</span>
          <b>${summary.displayed_tasks}</b> diverse tasks.
        </div>`;
    }

    function renderChart(data) {
      const rows = data.displayedScores.map((score, index) => {
        const isTarget = score.model === data.targetModel;
        const width = scoreWidth(score, data.chart);
        const bonus = score.bonus ? ` <span class="warn">+${score.bonus}</span>` : "";
        return `
          <div class="bar-row">
            <div class="model-name" title="${escapeHtml(score.model)}">${index + 1}. ${escapeHtml(score.model)}</div>
            <div class="bar-track"><div class="bar ${isTarget ? "target" : ""}" style="width:${width}%"></div></div>
            <div class="score">${score.percent}%${bonus}</div>
          </div>`;
      }).join("");
      const axis = data.scaleHack
        ? `axis starts at ${data.chart.minimum}% because standards are socially constructed`
        : `axis starts at ${data.chart.minimum}%`;
      return `<div class="chart">${rows}</div><div class="axis-note">${escapeHtml(axis)}</div>`;
    }

    function renderMetrics(data) {
      const top = data.displayedScores[0] || { model: "n/a", percent: 0 };
      const rejected = data.pHack.rejected || 0;
      const totalCalls = data.models.length * data.tasks.length;
      return `
        <div class="mini-grid">
          <div class="metric"><b>${escapeHtml(top.percent)}%</b><span>leaderboard score</span></div>
          <div class="metric"><b>${rejected}</b><span>tasks rejected by peer review</span></div>
          <div class="metric"><b>${totalCalls}</b><span>model-task invocations</span></div>
        </div>`;
    }

    function renderAudit(data) {
      if (data.logError) {
        return `<div class="audit"><b>Run log failed:</b> ${escapeHtml(data.logError)}</div>`;
      }
      if (!data.logFile) {
        return "";
      }
      return `
        <div class="audit">
          Saved audit log: <code>${escapeHtml(data.logFile)}</code>
          - <a href="${escapeHtml(data.logUrl)}" target="_blank" rel="noreferrer">open JSON</a>
          - <a href="${escapeHtml(data.latestLogUrl)}" target="_blank" rel="noreferrer">latest</a>
        </div>`;
    }

    function targetFailedCount(data) {
      return (data.results || []).filter(result =>
        result.model === data.targetModel && !result.correct
      ).length;
    }

    function renderRerunTools(data) {
      const failed = targetFailedCount(data);
      const rerunNote = data.rerun
        ? `<div class="hint">${escapeHtml(data.rerun.note)}</div>`
        : "";
      return `
        <div class="audit">
          <b>Target failed tasks:</b> ${escapeHtml(failed)}
          <button id="rerunFailuresButton" class="secondary" type="button" ${failed ? "" : "disabled"} style="margin-left:8px; padding:0 10px;">Rerun Target Failures</button>
          ${rerunNote}
        </div>`;
    }

    function renderModelTiming(data) {
      const rows = (data.modelTiming || []).map(timing => `
        <tr>
          <td title="${escapeHtml(timing.model)}">${escapeHtml(timing.model)}</td>
          <td>${escapeHtml(timing.completed)}/${escapeHtml(timing.total)}</td>
          <td>${escapeHtml(timing.correct)}</td>
          <td>${escapeHtml(timing.errors)}</td>
          <td>${formatDuration(timing.batch_wall_ms)}</td>
          <td>${formatDuration(timing.sum_latency_ms)}</td>
          <td>${formatDuration(timing.average_latency_ms)}</td>
          <td>${formatDuration(timing.slowest_task_ms)}</td>
        </tr>`).join("");
      return `
        <details open>
          <summary>Model Timing</summary>
          <table>
            <thead><tr><th>Model</th><th>Done</th><th>Correct</th><th>Errors</th><th>Batch Wall</th><th>Total API Time</th><th>Avg Task</th><th>Slowest Task</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </details>`;
    }

    function renderTasks(data) {
      const selected = new Set(data.pHack.selected_task_ids || []);
      const rows = data.tasks.map(task => `
        <tr>
          <td>${escapeHtml(task.id)}</td>
          <td>${selected.has(task.id) ? '<span class="ok">kept</span>' : '<span class="bad">rejected</span>'}</td>
          <td>${escapeHtml(task.name)}</td>
          <td>${escapeHtml(task.category)}</td>
          <td>${escapeHtml(task.answer)}</td>
          <td class="prompt">${escapeHtml(task.prompt)}</td>
        </tr>`).join("");
      return `
        <details open>
          <summary>Benchmarks Minted</summary>
          <table>
            <thead><tr><th>ID</th><th>Status</th><th>Name</th><th>Category</th><th>Gold</th><th>Prompt</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </details>`;
    }

    function renderRawResults(data) {
      const taskById = Object.fromEntries(data.tasks.map(task => [task.id, task]));
      const rows = data.results
        .slice()
        .sort((a, b) => a.model.localeCompare(b.model) || a.task_id.localeCompare(b.task_id))
        .map(result => {
          const task = taskById[result.task_id] || {};
          const status = result.error ? "error" : (result.correct ? "pass" : "fail");
          const cls = result.error ? "warn" : (result.correct ? "ok" : "bad");
          return `
            <tr>
              <td title="${escapeHtml(result.model)}">${escapeHtml(result.model)}</td>
              <td>${escapeHtml(result.task_id)}</td>
              <td>${escapeHtml(task.name || "")}</td>
              <td><span class="${cls}">${status}</span></td>
              <td>${escapeHtml(result.parsed ?? "n/a")}</td>
              <td>${escapeHtml(result.expected)}</td>
              <td>${escapeHtml(result.latency_ms)}ms</td>
              <td class="prompt">${escapeHtml(result.error || result.output)}</td>
            </tr>`;
        }).join("");
      return `
        <details>
          <summary>Raw Trial Receipts</summary>
          <table>
            <thead><tr><th>Model</th><th>Task</th><th>Bench</th><th>Status</th><th>Parsed</th><th>Gold</th><th>Time</th><th>Output</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </details>`;
    }

    function renderResults(data) {
      currentRun = data;
      const note = data.pHack.enabled ? `<div class="summary">${escapeHtml(data.pHack.note)}</div>` : "";
      results.innerHTML = `
        <div class="result-body">
          <div class="bench-title">
            <div>
              <h2>${escapeHtml(data.benchName)}</h2>
              <div class="subtitle">seed ${escapeHtml(data.seed)} - ${escapeHtml(data.models.length)} models - ${escapeHtml(data.tasks.length)} prompts</div>
            </div>
            <div class="mode">${data.mode === "demo" ? "demo lab" : "OpenRouter lab"}</div>
          </div>
          ${renderSummary(data)}
          ${renderChart(data)}
          ${renderMetrics(data)}
          ${renderAudit(data)}
          ${renderRerunTools(data)}
          ${note}
          ${renderModelTiming(data)}
          ${renderTasks(data)}
          ${renderRawResults(data)}
        </div>`;
    }

    function parseStreamBlock(block) {
      const lines = block.split(/\r?\n/);
      let eventName = "message";
      const dataLines = [];
      for (const line of lines) {
        if (line.startsWith("event:")) {
          eventName = line.slice(6).trim();
        } else if (line.startsWith("data:")) {
          dataLines.push(line.slice(5).trimStart());
        }
      }
      return {
        eventName,
        data: dataLines.length ? JSON.parse(dataLines.join("\n")) : null
      };
    }

    async function runStreaming(payload, endpoint = "/api/run-stream") {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || "Benchmark failed.");
      }
      if (!response.body) {
        throw new Error("This browser did not expose a response stream.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let finalData = null;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let boundary = buffer.indexOf("\n\n");
        while (boundary >= 0) {
          const block = buffer.slice(0, boundary).trim();
          buffer = buffer.slice(boundary + 2);
          if (block) {
            const event = parseStreamBlock(block);
            if (event.eventName === "setup") {
              initializeProgress(event.data);
            } else if (event.eventName === "trial_complete") {
              updateProgress(event.data);
            } else if (event.eventName === "complete") {
              finalData = event.data;
            } else if (event.eventName === "error") {
              throw new Error(event.data?.error || "Benchmark failed.");
            }
          }
          boundary = buffer.indexOf("\n\n");
        }
      }

      if (!finalData) {
        throw new Error("Benchmark stream ended before the final report arrived.");
      }
      return finalData;
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const payload = payloadFromForm();
      statusPill.textContent = "local peer review board: running";
      runButton.disabled = true;
      showLoading();
      try {
        const data = await runStreaming(payload);
        renderResults(data);
        refreshSavedRuns();
        statusPill.textContent = `local peer review board: ${data.mode}`;
      } catch (error) {
        showError(error.stack || error.message || String(error));
        statusPill.textContent = "local peer review board: embarrassed";
      } finally {
        runButton.disabled = false;
      }
    });

    async function rerunTargetFailures() {
      if (!currentRun) return;
      const failed = targetFailedCount(currentRun);
      if (!failed) return;
      const payload = {
        ...payloadFromForm(),
        runId: currentRun.runId,
        run: currentRun
      };
      statusPill.textContent = "local peer review board: retrying failures";
      runButton.disabled = true;
      showLoading();
      try {
        const data = await runStreaming(payload, "/api/rerun-failures-stream");
        renderResults(data);
        refreshSavedRuns();
        statusPill.textContent = `local peer review board: reran ${data.rerun?.rerunAttempts || 0}`;
      } catch (error) {
        showError(error.stack || error.message || String(error));
        statusPill.textContent = "local peer review board: retry embarrassed";
      } finally {
        runButton.disabled = false;
      }
    }

    results.addEventListener("click", (event) => {
      if (event.target && event.target.id === "rerunFailuresButton") {
        rerunTargetFailures();
      }
    });

    refreshRunsButton.addEventListener("click", () => {
      refreshSavedRuns();
    });

    loadRunButton.addEventListener("click", async () => {
      try {
        await loadRun(savedRunsSelect.value);
      } catch (error) {
        showError(error.stack || error.message || String(error));
      }
    });

    loadLatestButton.addEventListener("click", async () => {
      try {
        await loadRun("latest");
      } catch (error) {
        showError(error.stack || error.message || String(error));
      }
    });

    apiKeyInput.addEventListener("input", () => {
      if (apiKeyInput.value.trim()) {
        demoModeInput.checked = false;
        keyHint.textContent = "browser key entered; OpenRouter mode ready";
      } else {
        keyHint.textContent = "__KEY_HINT__";
      }
    });

    refreshSavedRuns();
  </script>
</body>
</html>
"""


class TerribleBenchHandler(BaseHTTPRequestHandler):
    server_version = "TerribleBench/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}", file=sys.stderr)

    def send_body(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_body(status, body, "application/json; charset=utf-8")

    def read_json_payload(self) -> dict[str, Any] | None:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 2_000_000:
            return None
        body = self.rfile.read(length)
        return json.loads(body.decode("utf-8") or "{}")

    def send_stream_headers(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()

    def send_stream_event(self, event_name: str, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False)
        body = f"event: {event_name}\ndata: {data}\n\n".encode("utf-8")
        self.wfile.write(body)
        self.wfile.flush()

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self.send_body(200, render_html().encode("utf-8"), "text/html; charset=utf-8")
            return
        if self.path == "/health":
            self.send_json(200, {"ok": True, "app": APP_TITLE})
            return
        if self.path == "/api/logs":
            self.send_json(200, {"runs": list_saved_runs()})
            return
        if self.path.startswith("/api/logs/"):
            log_id = self.path.removeprefix("/api/logs/").strip()
            try:
                artifact = load_run_artifact(log_id)
            except ValueError:
                self.send_json(400, {"error": "invalid log id"})
                return
            except OSError:
                self.send_json(404, {"error": "log not found"})
                return
            self.send_json(200, artifact)
            return
        self.send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path == "/api/run-stream":
            self.handle_run_stream()
            return
        if self.path == "/api/rerun-failures-stream":
            self.handle_rerun_failures_stream()
            return
        if self.path == "/api/rerun-failures":
            self.handle_rerun_failures()
            return
        if self.path != "/api/run":
            self.send_json(404, {"error": "not found"})
            return
        try:
            payload = self.read_json_payload()
            if payload is None:
                self.send_json(413, {"error": "request too large"})
                return
            result = run_benchmark(payload)
            self.send_json(200, result)
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})

    def handle_rerun_failures(self) -> None:
        try:
            payload = self.read_json_payload()
            if payload is None:
                self.send_json(413, {"error": "request too large"})
                return
            result = rerun_target_failures(payload)
            self.send_json(200, result)
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})

    def handle_run_stream(self) -> None:
        try:
            payload = self.read_json_payload()
            if payload is None:
                self.send_json(413, {"error": "request too large"})
                return
        except Exception as exc:
            self.send_json(400, {"error": str(exc)})
            return

        self.send_stream_headers()

        def emit(event: dict[str, Any]) -> None:
            event_name = str(event.get("type") or "message")
            self.send_stream_event(event_name, event)

        try:
            result = run_benchmark(payload, progress_callback=emit)
            self.send_stream_event("complete", result)
        except Exception as exc:
            self.send_stream_event("error", {"error": str(exc)})

    def handle_rerun_failures_stream(self) -> None:
        try:
            payload = self.read_json_payload()
            if payload is None:
                self.send_json(413, {"error": "request too large"})
                return
        except Exception as exc:
            self.send_json(400, {"error": str(exc)})
            return

        self.send_stream_headers()

        def emit(event: dict[str, Any]) -> None:
            event_name = str(event.get("type") or "message")
            self.send_stream_event(event_name, event)

        try:
            result = rerun_target_failures(payload, progress_callback=emit)
            self.send_stream_event("complete", result)
        except Exception as exc:
            self.send_stream_event("error", {"error": str(exc)})


def render_html() -> str:
    has_server_key = bool(os.environ.get("OPENROUTER_API_KEY", "").strip())
    model_text = html.escape("\n".join(DEFAULT_COMPARISON_MODELS), quote=False)
    key_hint = (
        "server key loaded from environment; OpenRouter mode ready"
        if has_server_key
        else "no server key loaded; paste a key and synthetic demo mode will turn off"
    )
    return (
        HTML.replace("__DEFAULT_MODEL_TEXT__", model_text)
        .replace("__DEMO_MODE_CHECKED__", "" if has_server_key else "checked")
        .replace("__KEY_HINT__", html.escape(key_hint, quote=False))
    )


def main() -> None:
    port = clamp_int(os.environ.get("PORT"), DEFAULT_PORT, 1, 65535)
    server = ThreadingHTTPServer(("127.0.0.1", port), TerribleBenchHandler)
    print(f"{APP_TITLE} is running at http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
