# 🗑️ Token-Bin

> Your precision token wastebin. Burn a *target* number of LLM tokens as precisely as the API allows — typically within ±2 tokens.

Token-Bin lets you deliberately waste LLM tokens with surgical precision. Whether you need to burn exactly 100 tokens or 100,000, Token-Bin calibrates against your model's API and iteratively converges to within ±2 tokens of your target.

## Features

- **Native Precision** — For OpenAI models, uses `tiktoken` to build the user message to an *exact* token count, then accounts for the fixed system-prompt + completion overhead. Usually lands within ±2 tokens in 1–2 rounds.
- **Calibration + Feedback** — For models without native tokenizers (Claude, Gemini, custom APIs), calibrates via sample request then iteratively homes in
- **Beautiful TUI** — Guided terminal interface built with Textual. Pick your model, enter your API key, set a target, and watch the waste happen in real-time
- **CLI mode** — Scriptable command-line interface for CI or automation
- **Waste Report** — Detailed summary: actual vs target tokens, error %, rounds, duration, and estimated cost
- **Multi-provider** — OpenAI, Anthropic Claude, DeepSeek, and any OpenAI-compatible API (OpenRouter, Together, Ollama, vLLM, etc.)

## Install

```bash
pip install -e .
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv pip install -e .
```

## Usage

### TUI Mode (default)

```bash
token-bin
# or
token-bin tui
```

You'll be guided through:
1. Select provider (OpenAI / Anthropic / DeepSeek / Generic)
2. Enter API key, model name, and target token count
3. Calibrate → Waste → View the report

### CLI Mode

```bash
# OpenAI — 1,000 tokens, precise
token-bin waste -p openai -m gpt-4o -n 1000

# Anthropic Claude — 5,000 tokens
token-bin waste -p anthropic -m claude-3-5-sonnet-20241022 -n 5000 -k $ANTHROPIC_API_KEY

# DeepSeek — 2,000 tokens
token-bin waste -p deepseek -m deepseek-chat -n 2000

# Custom endpoint (OpenRouter, local LLM, etc.)
token-bin waste -p generic -m openai/gpt-4o -n 10000 \
  --api-key $OPENROUTER_KEY \
  --base-url https://openrouter.ai/api/v1
```

## How It Works

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Calibrate   │ ──▶ │   Waste      │ ──▶ │   Report     │
│              │     │              │     │              │
│ Sample req   │     │ Gen prompt   │     │ Target vs    │
│ → chars/tok  │     │ → API call   │     │ Actual       │
│   ratio      │     │ → compare    │     │ Error %      │
│              │     │ → adjust     │     │ Cost est.    │
└──────────────┘     └──────┬───────┘     └──────────────┘
                            │
                     feedback loop
                     (up to 15 rounds)
                     converges to ±2 tokens
```

### Two Paths

| Path | Models | Method | Precision |
|------|--------|--------|-----------|
| **Native** | OpenAI (GPT-4o, GPT-4, GPT-3.5, …) | `tiktoken` exact content + system/completion overhead accounting | **±2 tokens**, 1–2 rounds |
| **Generic** | Anthropic Claude, DeepSeek, Gemini, custom/OpenAI-compatible | Calibration + feedback loop | **±2 tokens** (up to 15 rounds) |

## Provider Settings

| Provider | CLI flag | Default base URL |
|----------|----------|------------------|
| OpenAI | `-p openai` | `https://api.openai.com/v1` |
| Anthropic | `-p anthropic` | `https://api.anthropic.com/v1/messages` |
| DeepSeek | `-p deepseek` | `https://api.deepseek.com/v1` |
| Generic | `-p generic` | `https://api.openai.com/v1` (configurable) |

Set API keys via env: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or pass `--api-key`.

## Project Structure

```
token-bin/
├── pyproject.toml
├── README.md
└── token_bin/
    ├── __init__.py
    ├── main.py         # Entry point (TUI + CLI)
    ├── engine.py       # Core: providers, TokenWaster, reporter
    └── ui.py           # Textual TUI (guided wizard)
```

## License

MIT
