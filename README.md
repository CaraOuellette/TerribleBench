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
- Parses `final answer: <integer>`.
- Produces a leaderboard.
- Optionally rejects inconvenient tasks and zooms the chart axis until the desired conclusion emerges.

## Auditing Runs

Every run writes a JSON audit log under `run_logs/`. The UI links the current run and `/api/logs/latest`. Logs include generated tasks, prompts, expected answers, raw model outputs, parsed answers, unmodified `rawScores`, and knob-affected `displayedScores`. API keys are not logged.
