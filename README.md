# Terrible Bench.ai

One-click benchmark generation for people who have already chosen the winner.

## Run

```powershell
python terrible_bench.py
```

Then open http://127.0.0.1:8765.

The app can run in two modes:

- `Synthetic demo mode`: fake-but-plausible results without network access or API credits.
- `OpenRouter mode`: real model calls, parsed and scored against the generated answer key.

To run real calls, enter an OpenRouter API key, set `OPENROUTER_API_KEY`, or put `OPENROUTER_API_KEY=...` in `.env`. If the app sees a server-side key, synthetic demo mode starts unchecked. If you paste a key into the browser, synthetic demo mode turns off automatically.

OpenRouter attribution uses this repo URL by default. Set `OPENROUTER_SITE_URL` to override it.

## Test

```powershell
python -m unittest -v
```

## What It Does

- Generates one-shot letter-counting and tiny arithmetic benchmarks.
- Computes the answer key locally.
- Runs each model against each prompt.
- Streams per-model progress while calls are in flight.
- Parses `final answer: <integer>`.
- Produces a leaderboard.
- Reports per-model timing, including batch wall time, total API time, average task time, and slowest task.
- Reloads saved run logs so you can inspect old results without rerunning model calls.
- Optionally cherry-picks the best observed task subset, reruns the target model's failed tasks, and zooms the chart axis until the desired conclusion emerges.
- Uses `good_models.txt` as a blocklist when `Weenie model pile-on` is enabled, so strong models do not accidentally join the weenie pool.

## Auditing Runs

Every run writes a JSON audit log under `run_logs/`. The UI links the current run and `/api/logs/latest`, and the saved-runs picker loads old logs back into the report view. Logs include generated tasks, prompts, expected answers, raw model outputs, parsed answers, model timing, unmodified `rawScores`, and knob-affected `displayedScores`. API keys are not logged.
