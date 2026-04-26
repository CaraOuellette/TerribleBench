# Terrible Bench.ai

One-click benchmark generation for people who have already chosen the winner.

## Run

```powershell
python terrible_bench.py
```

Then open http://127.0.0.1:8765.

The app defaults to demo mode, so it can produce fake-but-plausible lab results without network access or API credits. To run real calls, enter an OpenRouter API key, set `OPENROUTER_API_KEY`, or put `OPENROUTER_API_KEY=...` in `.env`, then uncheck `Demo mode`.

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
